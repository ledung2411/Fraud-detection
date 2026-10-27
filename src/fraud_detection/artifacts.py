"""Everything the scoring service needs, saved together and loaded together."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from .drift import DriftMonitor
from .features import FeatureBuilder
from .velocity import OnlineVelocityStore

MODEL = "model.txt"
BUILDER = "feature_builder.joblib"
STORE = "velocity_store.joblib"
DRIFT = "drift_reference.json"
META = "metadata.json"


@dataclass
class Bundle:
    booster: lgb.Booster
    builder: FeatureBuilder
    store: OnlineVelocityStore
    monitor: DriftMonitor
    metadata: dict[str, Any]


def save_bundle(path: str | Path, bundle: Bundle) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    bundle.booster.save_model(out / MODEL)
    joblib.dump(bundle.builder, out / BUILDER)
    joblib.dump(bundle.store, out / STORE, compress=3)
    bundle.monitor.save(out / DRIFT)
    (out / META).write_text(json.dumps(bundle.metadata, indent=2))
    return out


def load_bundle(path: str | Path) -> Bundle:
    p = Path(path)
    if not (p / MODEL).exists():
        raise FileNotFoundError(f"no trained model in {p}; run `make train` first")
    return Bundle(
        booster=lgb.Booster(model_file=str(p / MODEL)),
        builder=joblib.load(p / BUILDER),
        store=joblib.load(p / STORE),
        monitor=DriftMonitor.load(p / DRIFT),
        metadata=json.loads((p / META).read_text()),
    )
