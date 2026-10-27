"""Baseline logistic regression and the LightGBM model."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)

LGBM_DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 20,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
}


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress heavy tails (amounts, counts) so a linear model is not dominated by them."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_logreg(num_cols: list[str], cat_cols: list[str], **params: Any) -> Pipeline:
    """A fair baseline: robust scaling, missing-value indicators and one-hot categoricals."""
    params = {"C": 0.1, "max_iter": 1000, **params}
    numeric = Pipeline(
        [
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
            # Cap the width: DeviceInfo / id_31 have hundreds of rare values on IEEE-CIS.
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", max_categories=30, dtype=np.float32
                ),
            ),
        ]
    )
    pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(**params))])


def fit_logreg(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    max_train_rows: int | None = None,
    seed: int = 42,
    **params: Any,
) -> Pipeline:
    if max_train_rows and len(X) > max_train_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_train_rows, replace=False)
        X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
    cats = [c for c in X.columns if c in set(cat_cols)]
    nums = [c for c in X.columns if c not in set(cats)]
    model = make_logreg(nums, cats, **params)
    model.fit(X, y)
    return model


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cat_cols: list[str],
    **params: Any,
) -> lgb.Booster:
    """Train with early stopping on validation PR-AUC (never on the test month)."""
    p = {**LGBM_DEFAULTS, **params}
    rounds = p.pop("n_estimators")
    patience = p.pop("early_stopping_rounds")
    cats = [c for c in X_train.columns if c in set(cat_cols)]
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cats, free_raw_data=True)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)
    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(200)],
    )
    log.info("lightgbm best iteration %s", booster.best_iteration)
    return booster


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, lgb.Booster):
        # num_iteration=None uses the early-stopping best iteration when there is one.
        return model.predict(X[model.feature_name()], num_iteration=None)
    return model.predict_proba(X)[:, 1]
