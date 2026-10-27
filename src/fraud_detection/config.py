"""Typed configuration loaded from YAML (see configs/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "ieee"  # "ieee" | "synthetic"
    transaction_path: str = "data/raw/ieee-fraud-detection/train_transaction.csv"
    identity_path: str = "data/raw/ieee-fraud-detection/train_identity.csv"
    cache_path: str | None = "data/processed/ieee.parquet"
    synthetic: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    valid_days: int = 30
    test_days: int = 30
    # Days dropped between train and valid/test, to mimic label maturity (chargebacks
    # arrive weeks after the transaction, so the freshest labels are not usable yet).
    gap_days: int = 0


@dataclass
class FeatureConfig:
    # name -> list of columns whose concatenation identifies the entity.
    # "uid" = card1 + addr1 + D1n is the client proxy used by the IEEE-CIS winners
    # (D1n = transaction day - D1 = the day the client relationship started).
    entities: dict[str, list[str]] = field(
        default_factory=lambda: {"card1": ["card1"], "uid": ["card1", "addr1", "D1n"]}
    )
    windows: dict[str, int] = field(
        default_factory=lambda: {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
    )
    freq_encode: list[str] = field(
        default_factory=lambda: ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain"]
    )
    min_category_count: int = 20
    drop: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    review_budget: float = 0.01
    budgets: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05])
    bootstrap: int = 200
    seed: int = 42


@dataclass
class ModelConfig:
    logreg: dict[str, Any] = field(default_factory=dict)
    lightgbm: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplainConfig:
    sample_size: int = 5000
    top_k_reasons: int = 5


@dataclass
class DriftConfig:
    n_bins: int = 10
    max_categories: int = 20
    warn: float = 0.1
    alert: float = 0.25


@dataclass
class OutputConfig:
    artifact_dir: str = "artifacts/ieee"
    report_dir: str = "reports/ieee"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        kwargs = {}
        for f in fields(cls):
            section = raw.get(f.name) or {}
            sub_cls = f.default_factory().__class__  # type: ignore[misc]
            known = {sf.name for sf in fields(sub_cls)}
            unknown = set(section) - known
            if unknown:
                raise ValueError(f"Unknown keys in config section '{f.name}': {sorted(unknown)}")
            kwargs[f.name] = sub_cls(**section)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})
