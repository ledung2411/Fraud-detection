"""Replay the test month through the API as if it were live traffic.

    python -m fraud_detection.replay --config configs/synthetic.yaml --limit 5000
    python -m fraud_detection.replay --config configs/ieee.yaml --url http://localhost:8000

The API starts from the velocity state at the beginning of the test month and
builds every feature itself, one transaction at a time. Comparing its scores with
the offline (batch) scores of the same rows proves there is no training/serving
skew; the run also reports latency, the live alert rate and the drift check.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_bundle
from .config import Config
from .data import TARGET, TIME_COL
from .train import build_features


def raw_columns(df: pd.DataFrame, velocity_cols: list[str]) -> list[str]:
    """IEEE-CIS input columns only (what a caller would send), never derived ones."""
    derived = {"TransactionDay", "D1n", TARGET, *velocity_cols}
    return [c for c in df.columns if c not in derived]


def to_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for rec in df.to_dict("records"):
        out.append(
            {
                k: (
                    None
                    if isinstance(v, float) and math.isnan(v)
                    else (v.item() if isinstance(v, np.generic) else v)
                )
                for k, v in rec.items()
                if v is not None
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/synthetic.yaml")
    p.add_argument("--url", default=None, help="running API; default is an in-process app")
    p.add_argument("--limit", type=int, default=5000, help="number of test transactions (0 = all)")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load(a.config)
    print(json.dumps(replay(cfg, a.url, a.limit, a.batch_size), indent=2))


def replay(cfg: Config, url: str | None = None, limit: int = 5000, batch_size: int = 1) -> dict:
    bundle = load_bundle(cfg.output.artifact_dir)
    df, velocity_cols = build_features(cfg)
    start = bundle.metadata["state_as_of_transaction_dt"]
    test = df[df[TIME_COL] >= start]
    if limit:
        test = test.iloc[:limit]

    offline = bundle.booster.predict(bundle.builder.transform(test)[bundle.booster.feature_name()])
    payload = to_payload(test[raw_columns(test, velocity_cols)])

    if url:
        import httpx

        client = httpx.Client(base_url=url, timeout=30)
    else:
        from fastapi.testclient import TestClient

        from .api.app import create_app

        client = TestClient(create_app(cfg.output.artifact_dir)).__enter__()

    online, latency = [], []
    for i in range(0, len(payload), batch_size):
        chunk = payload[i : i + batch_size]
        t = time.perf_counter()
        if batch_size == 1:
            r = client.post("/score", json=chunk[0])
            r.raise_for_status()
            online.append(r.json()["fraud_score"])
        else:
            r = client.post("/score/batch", json={"transactions": chunk})
            r.raise_for_status()
            online += [x["fraud_score"] for x in r.json()["results"]]
        latency.append((time.perf_counter() - t) * 1000 / len(chunk))
    drift = client.get("/monitoring/drift").json()

    online = np.array(online)
    y = test[TARGET].to_numpy()
    thr = bundle.metadata["threshold"]
    alert = online >= thr
    return {
        "transactions": len(test),
        "max_abs_score_diff_online_vs_offline": float(np.max(np.abs(online - offline))),
        "latency_ms_per_txn": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
        },
        "alert_rate": float(alert.mean()),
        "review_budget": bundle.metadata["review_budget"],
        "recall_at_threshold": float(y[alert].sum() / max(y.sum(), 1)),
        "precision_at_threshold": float(y[alert].mean()) if alert.any() else None,
        "drift": {
            k: drift.get(k)
            for k in (
                "status",
                "n_rows",
                "score_psi",
                "alert_rate",
                "n_features_moderate",
                "n_features_significant",
            )
        },
    }


if __name__ == "__main__":
    main()
