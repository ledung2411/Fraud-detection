"""Data and score drift with the Population Stability Index (PSI).

The reference distribution is the validation month: it is the most recent period
before deployment and the one the alert threshold was calibrated on. Each feature is
binned on reference deciles (plus a missing bin; categoricals use their top
categories plus "other"), and current traffic is compared bin by bin:

    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

Rule of thumb used in credit risk: < 0.1 stable, 0.1-0.25 moderate shift worth a
look, > 0.25 significant shift (investigate, consider retraining). PSI needs no
labels, which matters because fraud labels (chargebacks) arrive weeks late; the
score distribution and the alert rate at the frozen threshold are the earliest
warning that the review queue is about to over- or under-fill.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref, dtype=float), EPS, None)
    cur = np.clip(np.asarray(cur, dtype=float), EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@dataclass
class FeatureProfile:
    kind: str  # "numeric" | "categorical"
    cuts: list[float]  # numeric: interior bin edges; categorical: kept category codes
    ref: list[float]  # reference share per bin; last bin = missing

    def bin_shares(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        missing = np.isnan(x)
        k = len(self.cuts)
        if self.kind == "numeric":
            idx = np.searchsorted(np.asarray(self.cuts), x, side="right")  # 0..k
        else:
            lookup = {c: i for i, c in enumerate(self.cuts)}
            idx = pd.Series(x).map(lookup).fillna(k).to_numpy(dtype=int)  # k = "other"
        idx = np.where(missing, k + 1, idx)
        return np.bincount(idx, minlength=k + 2) / max(len(x), 1)


class DriftMonitor:
    def __init__(self, profiles: dict[str, FeatureProfile], warn: float = 0.1, alert: float = 0.25):
        self.profiles = profiles
        self.warn = warn
        self.alert = alert

    @classmethod
    def fit(
        cls,
        ref: pd.DataFrame,
        cat_cols: list[str] = (),
        n_bins: int = 10,
        max_categories: int = 20,
        warn: float = 0.1,
        alert: float = 0.25,
    ) -> DriftMonitor:
        profiles = {}
        cats = set(cat_cols)
        for col in ref.columns:
            x = ref[col].to_numpy(dtype=float)
            present = x[~np.isnan(x)]
            if col in cats:
                vals, counts = np.unique(present, return_counts=True)
                cuts = vals[np.argsort(-counts)][:max_categories].tolist()
                prof = FeatureProfile("categorical", cuts, [])
            else:
                qs = (
                    np.quantile(present, np.linspace(0, 1, n_bins + 1)[1:-1])
                    if len(present)
                    else []
                )
                prof = FeatureProfile("numeric", np.unique(qs).tolist(), [])
            prof.ref = prof.bin_shares(x).tolist()
            profiles[col] = prof
        return cls(profiles, warn, alert)

    def status(self, value: float) -> str:
        if value >= self.alert:
            return "significant"
        return "moderate" if value >= self.warn else "stable"

    def report(self, cur: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col, prof in self.profiles.items():
            if col not in cur:
                continue
            shares = prof.bin_shares(cur[col].to_numpy(dtype=float))
            value = psi(prof.ref, shares)
            rows.append(
                {
                    "feature": col,
                    "psi": value,
                    "status": self.status(value),
                    "missing_ref": prof.ref[-1],
                    "missing_cur": float(shares[-1]),
                }
            )
        return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)

    def summary(self, cur: pd.DataFrame, top: int = 10) -> dict:
        rep = self.report(cur)
        score_row = rep[rep["feature"] == "score"]
        feats = rep[rep["feature"] != "score"]
        worst = "stable"
        for s in ("moderate", "significant"):
            if (rep["status"] == s).any():
                worst = s
        return {
            "status": worst,
            "n_rows": int(len(cur)),
            "score_psi": float(score_row["psi"].iloc[0]) if len(score_row) else None,
            "n_features_moderate": int((feats["status"] == "moderate").sum()),
            "n_features_significant": int((feats["status"] == "significant").sum()),
            "top_features": feats.head(top).to_dict(orient="records"),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "warn": self.warn,
            "alert": self.alert,
            "profiles": {k: asdict(v) for k, v in self.profiles.items()},
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> DriftMonitor:
        payload = json.loads(Path(path).read_text())
        profiles = {k: FeatureProfile(**v) for k, v in payload["profiles"].items()}
        return cls(profiles, payload["warn"], payload["alert"])
