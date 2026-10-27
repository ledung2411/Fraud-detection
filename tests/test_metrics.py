import numpy as np
import pytest

from fraud_detection.metrics import (
    amount_recall_at_budget,
    block_bootstrap,
    evaluate,
    precision_at_budget,
    recall_at_budget,
    threshold_for_budget,
)


def test_budget_metrics_on_a_known_ranking():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    amt = np.array([100, 1, 50, 1, 1, 1, 1, 1, 1, 850])
    assert recall_at_budget(y, s, 0.2) == pytest.approx(1 / 3)  # top 2 rows: 1 of 3 frauds
    assert precision_at_budget(y, s, 0.3) == pytest.approx(2 / 3)
    assert amount_recall_at_budget(y, s, amt, 0.3) == pytest.approx(150 / 1000)


def test_perfect_model_hits_the_ceiling():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.03).astype(int)
    budget = 0.01
    assert recall_at_budget(y, y + rng.random(len(y)) * 0.1, budget) == pytest.approx(100 / y.sum())
    assert precision_at_budget(y, y.astype(float), budget) == 1.0


def test_threshold_flags_the_budget_share():
    s = np.random.default_rng(1).random(100_000)
    thr = threshold_for_budget(s, 0.01)
    assert (s >= thr).mean() == pytest.approx(0.01, abs=1e-3)
    res = evaluate((s > 0.99).astype(int), s, np.ones_like(s), [0.01], threshold=thr)
    assert res["alert_rate_at_threshold"] == pytest.approx(0.01, abs=1e-3)
    assert res["recall@1pct"] > 0.95


def test_block_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.1).astype(int)
    s = y + rng.random(3000)
    days = np.repeat(np.arange(30), 100)
    a = block_bootstrap(
        y, {"m": s, "same": s.copy()}, days, lambda yy, ss: recall_at_budget(yy, ss, 0.05), 30
    )
    b = block_bootstrap(
        y, {"m": s, "same": s.copy()}, days, lambda yy, ss: recall_at_budget(yy, ss, 0.05), 30
    )
    np.testing.assert_array_equal(a["m"], b["m"])
    np.testing.assert_array_equal(a["m"], a["same"])  # identical models -> zero paired difference
