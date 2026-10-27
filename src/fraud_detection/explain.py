"""SHAP explanations: global importance for the report, reason codes for each alert.

LightGBM ships an exact TreeSHAP implementation (``pred_contrib=True``), the same
algorithm as ``shap.TreeExplainer``. Using it keeps the scoring API free of the
``shap`` dependency; the ``shap`` package is only used for its plots.
"""

from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd

_WINDOW_TEXT = {"1h": "last hour", "24h": "last 24h", "7d": "last 7 days"}
_ENTITY_TEXT = {"card1": "card BIN (card1)", "uid": "client (card1+addr1+D1n)"}


def _matrix(booster: lgb.Booster, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Rows in the booster's feature order (an ndarray is assumed to be in that order)."""
    return X[booster.feature_name()].to_numpy() if isinstance(X, pd.DataFrame) else X


def shap_values(
    booster: lgb.Booster, X: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature SHAP values in log-odds space and the expected value per row."""
    contrib = booster.predict(_matrix(booster, X), pred_contrib=True)
    return contrib[:, :-1], contrib[:, -1]


def global_importance(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    values, _ = shap_values(booster, X)
    imp = pd.DataFrame(
        {"feature": booster.feature_name(), "mean_abs_shap": np.abs(values).mean(axis=0)}
    )
    return imp.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def describe(feature: str) -> str:
    """Plain-language description of a feature, for analysts reading reason codes."""
    m = re.fullmatch(r"(\w+?)_(cnt|amt_sum|amt_ratio)_(\w+)", feature)
    if m and m.group(1) in _ENTITY_TEXT:
        ent, kind, w = (
            _ENTITY_TEXT[m.group(1)],
            m.group(2),
            _WINDOW_TEXT.get(m.group(3), m.group(3)),
        )
        return {
            "cnt": f"number of prior transactions of this {ent} in the {w}",
            "amt_sum": f"amount spent by this {ent} in the {w}",
            "amt_ratio": f"this amount vs. the {ent}'s average amount in the {w}",
        }[kind]
    m = re.fullmatch(r"(\w+?)_(n_prior|secs_since_prev|amt_mean_hist|amt_z_hist)", feature)
    if m and m.group(1) in _ENTITY_TEXT:
        ent = _ENTITY_TEXT[m.group(1)]
        return {
            "n_prior": f"number of earlier transactions of this {ent}",
            "secs_since_prev": f"seconds since the previous transaction of this {ent}",
            "amt_mean_hist": f"average historical amount of this {ent}",
            "amt_z_hist": f"how unusual this amount is for this {ent} (std. deviations)",
        }[m.group(2)]
    fixed = {
        "TransactionAmt": "transaction amount",
        "amt_log": "transaction amount (log)",
        "amt_cents": "cents part of the amount (non-round = currency conversion)",
        "hour": "hour of day",
        "dow": "day of week",
        "email_match": "purchaser and recipient e-mail domains match",
        "ProductCD": "product code",
        "P_emaildomain": "purchaser e-mail domain",
        "R_emaildomain": "recipient e-mail domain",
        "dist1": "distance between billing and other address",
        "DeviceInfo": "device",
        "D1": "days since the client relationship started",
    }
    if feature in fixed:
        return fixed[feature]
    if feature.endswith("_freq"):
        return f"how common this {feature[:-5]} value is in history"
    return feature


def reason_codes(
    booster: lgb.Booster, X: pd.DataFrame | np.ndarray, top_k: int = 5
) -> list[list[dict[str, float | str | None]]]:
    """Top features pushing each row's score *up*, with their value and SHAP contribution."""
    data = _matrix(booster, X)
    values, _ = shap_values(booster, data)
    names = booster.feature_name()
    out = []
    for i in range(len(X)):
        order = np.argsort(-values[i])[:top_k]
        out.append(
            [
                {
                    "feature": names[j],
                    "description": describe(names[j]),
                    # 6 significant digits: the matrix is float32 (365.43 -> 365.42999...)
                    "value": None if np.isnan(data[i, j]) else float(f"{data[i, j]:.6g}"),
                    "contribution": float(values[i, j]),
                }
                for j in order
                if values[i, j] > 0
            ]
        )
    return out
