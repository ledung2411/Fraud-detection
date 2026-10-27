import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.drift import DriftMonitor, psi
from fraud_detection.split import time_split


def test_split_is_strictly_chronological_with_gap():
    df = pd.DataFrame({"TransactionDay": np.repeat(np.arange(1, 101), 10)})
    sp = time_split(df, SplitConfig(valid_days=20, test_days=10, gap_days=5))
    day = df["TransactionDay"]
    assert day[sp.train].max() < day[sp.valid].min() - 5
    assert day[sp.valid].max() < day[sp.test].min()
    assert sp.test_days == (91, 100) and sp.valid_days == (71, 90) and sp.train_days == (1, 65)
    assert not (sp.train & sp.valid).any() and not (sp.valid & sp.test).any()


def test_split_rejects_too_little_history():
    df = pd.DataFrame({"TransactionDay": np.arange(1, 31)})
    with pytest.raises(ValueError):
        time_split(df, SplitConfig(valid_days=20, test_days=15))


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert psi([0.9, 0.1], [0.1, 0.9]) > 1


def test_monitor_flags_shift_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "amount": rng.lognormal(3, 1, 20_000),
            "email": rng.integers(0, 5, 20_000).astype(float),
            "stable": rng.normal(0, 1, 20_000),
        }
    )
    cur = pd.DataFrame(
        {
            "amount": rng.lognormal(3.6, 1, 5000),
            "email": np.full(5000, 7.0),
            "stable": rng.normal(0, 1, 5000),
        }
    )
    mon = DriftMonitor.fit(ref, cat_cols=["email"])
    rep = mon.report(cur).set_index("feature")
    assert rep.loc["amount", "status"] == "significant"
    assert rep.loc["email", "status"] == "significant"  # an unseen category lands in "other"
    assert rep.loc["stable", "status"] == "stable"
    mon.save(tmp_path / "ref.json")
    again = DriftMonitor.load(tmp_path / "ref.json").report(cur).set_index("feature")
    pd.testing.assert_series_equal(rep["psi"], again["psi"])
