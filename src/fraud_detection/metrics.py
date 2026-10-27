"""Metrics that match how a fraud team works: a fixed manual-review capacity.

With ~3% fraud, ROC-AUC looks great for almost any model and accuracy is meaningless.
What matters operationally is: if analysts can review only ``budget`` (e.g. 1%) of
transactions, how much fraud (count and money) lands in that queue? PR-AUC (average
precision) summarises the same precision/recall trade-off over all budgets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _top_k(score: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(round(budget * len(score))))
    return np.argsort(-score, kind="stable")[:k]


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Share of all frauds that are in the top ``budget`` fraction of scores."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(y[_top_k(np.asarray(score), budget)].sum() / y.sum())


def precision_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Share of reviewed transactions that are fraud (analyst hit rate)."""
    return float(np.asarray(y)[_top_k(np.asarray(score), budget)].mean())


def amount_recall_at_budget(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, budget: float
) -> float:
    """Share of fraudulent *money* caught in the review queue."""
    y, amount = np.asarray(y), np.asarray(amount, dtype=float)
    total = amount[y == 1].sum()
    if total == 0:
        return float("nan")
    top = _top_k(np.asarray(score), budget)
    return float(amount[top][y[top] == 1].sum() / total)


def threshold_for_budget(score: np.ndarray, budget: float) -> float:
    """Score cut-off that flags ``budget`` of the given (validation) population."""
    return float(np.quantile(np.asarray(score), 1.0 - budget))


def evaluate(
    y: np.ndarray,
    score: np.ndarray,
    amount: np.ndarray,
    budgets: Sequence[float],
    threshold: float | None = None,
) -> dict[str, float]:
    """All headline metrics for one model on one period.

    ``threshold`` (chosen on validation for the target budget) gives the *operational*
    view: on the test month the same cut-off may flag more or less than the budget,
    which is what would actually happen in production.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    res = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "fraud_rate": float(y.mean()),
        "n": int(len(y)),
    }
    for b in budgets:
        tag = f"{b * 100:g}pct"
        res[f"recall@{tag}"] = recall_at_budget(y, score, b)
        res[f"precision@{tag}"] = precision_at_budget(y, score, b)
        res[f"amount_recall@{tag}"] = amount_recall_at_budget(y, score, amount, b)
    if threshold is not None:
        alert = score >= threshold
        res["threshold"] = float(threshold)
        res["alert_rate_at_threshold"] = float(alert.mean())
        res["recall_at_threshold"] = float(y[alert].sum() / max(y.sum(), 1))
        res["precision_at_threshold"] = float(y[alert].mean()) if alert.any() else float("nan")
    return res


def block_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    blocks: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Bootstrap a metric for several models, resampling whole days.

    Fraud comes in bursts (one compromised card = many rows on the same day), so rows
    are not independent; resampling days instead of rows gives honest (wider) intervals.
    The same resamples are used for every model, so differences are paired.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        for k, s in scores.items():
            out[k][b] = metric(y[idx], s[idx])
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
