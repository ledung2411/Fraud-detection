"""Scoring logic behind the API, independent of FastAPI so it can be tested directly."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import Bundle, load_bundle
from ..data import ID_COL, add_time_fields
from ..explain import reason_codes

MIN_ROWS_FOR_DRIFT = 200


class ScoringService:
    """Turns raw transactions into scores, decisions and reason codes.

    Features are built exactly as in training: velocity features come from the
    online store (state as of the end of the data the model was trained/validated on,
    then updated with every scored transaction), the rest from the frozen
    ``FeatureBuilder``. Recently scored rows are kept for drift monitoring.
    """

    def __init__(self, bundle: Bundle, monitor_window: int = 5000):
        if bundle.booster.feature_name() != bundle.builder.feature_names:
            raise ValueError("model and feature builder disagree on the feature list")
        self.bundle = bundle
        self.meta = bundle.metadata
        self.threshold = float(self.meta["threshold"])
        self.top_k = int(self.meta.get("top_k_reasons", 5))
        self._lock = threading.Lock()
        self._recent: deque[np.ndarray] = deque(maxlen=monitor_window)
        self.n_scored = 0

    @classmethod
    def from_dir(cls, path: str | Path, **kwargs: Any) -> ScoringService:
        return cls(load_bundle(path), **kwargs)

    def score(
        self,
        transactions: Sequence[dict[str, Any]],
        update_state: bool = True,
        explain: str = "review",
    ) -> list[dict]:
        """Score transactions given in stream order.

        With ``update_state`` (the default) every transaction is added to the velocity
        state after it is scored, as in production. Set it to False for what-if calls
        that must not count as real traffic.

        ``explain`` chooses which rows get SHAP reason codes: "review" (only those sent
        to analysts - TreeSHAP costs a few ms per row, the score alone well under one),
        "all" or "none".
        """
        b = self.bundle
        records = [add_time_fields(t) for t in transactions]
        with self._lock:
            for r in records:
                r.update(b.store.compute_and_update(r) if update_state else b.store.compute(r))
            X = b.builder.transform_records(records)
            scores = b.booster.predict(X)
            self._recent.extend(np.c_[X.astype(float), scores])
            self.n_scored += len(X)

        flagged = scores >= self.threshold
        wanted = {"all": np.ones_like(flagged), "none": np.zeros_like(flagged)}.get(
            explain, flagged
        )
        reasons: list[list[dict]] = [[] for _ in records]
        idx = np.flatnonzero(wanted)
        if len(idx):
            for i, rc in zip(idx, reason_codes(b.booster, X[idx], self.top_k), strict=True):
                reasons[i] = rc
        cats = set(b.builder.cat_cols)
        out = []
        for i, rec in enumerate(records):
            for r in reasons[i]:
                if r["feature"] in cats:  # show "gmail.com", not the internal category code
                    v = rec.get(r["feature"])
                    r["value"] = (
                        None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
                    )
            out.append(
                {
                    "TransactionID": int(rec[ID_COL]),
                    "fraud_score": float(scores[i]),
                    "decision": "review" if flagged[i] else "approve",
                    "threshold": self.threshold,
                    "reasons": reasons[i],
                    "model_version": self.meta["model_version"],
                }
            )
        return out

    def drift(self) -> dict:
        """PSI of recent traffic (features and score) against the validation month."""
        with self._lock:
            rows = np.array(self._recent)
        n = len(rows)
        if n < MIN_ROWS_FOR_DRIFT:
            return {"status": "insufficient_data", "n_rows": n, "min_rows": MIN_ROWS_FOR_DRIFT}
        cols = [*self.bundle.builder.feature_names, "score"]
        cur = pd.DataFrame(rows, columns=cols)
        summary = self.bundle.monitor.summary(cur)
        alert_rate = float((cur["score"] >= self.threshold).mean())
        budget = float(self.meta["review_budget"])
        summary["alert_rate"] = alert_rate
        summary["review_budget"] = budget
        # The queue is sized for the budget; a large deviation means over/under-staffing.
        summary["alert_rate_vs_budget"] = alert_rate / budget
        if not 0.5 <= alert_rate / budget <= 1.5 and summary["status"] == "stable":
            summary["status"] = "moderate"
        return summary

    def info(self) -> dict:
        m = self.meta
        return {
            "model_version": m["model_version"],
            "data_source": m["data_source"],
            "n_features": len(m["features"]),
            "n_velocity_features": len(m["velocity_features"]),
            "entities": m["entities"],
            "windows": m["windows"],
            "review_budget": m["review_budget"],
            "threshold": self.threshold,
            "split_days": m["split_days"],
            "test_metrics": m["test_metrics"],
            "n_scored_since_start": self.n_scored,
            "entities_in_state": {e: len(s) for e, s in self.bundle.store.state.items()},
        }
