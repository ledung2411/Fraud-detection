"""Static features and the train-fitted encoders shared by training and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .data import AMOUNT_COL, DAY_COL, ID_COL, TARGET, TIME_COL

# Identifiers, label and absolute time never go into the model: the absolute day (and
# D1n, the client start day) only tells the model *when* a row is, which does not
# generalise to future months.
NON_FEATURES = {ID_COL, TARGET, TIME_COL, DAY_COL, "D1n"}
DERIVED = ["amt_log", "amt_cents", "hour", "dow", "email_match"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features that need no history and no fitting."""
    amt = pd.to_numeric(df[AMOUNT_COL], errors="coerce").astype(float)
    t = pd.to_numeric(df[TIME_COL], errors="coerce")
    p = df["P_emaildomain"] if "P_emaildomain" in df else pd.Series(np.nan, index=df.index)
    r = df["R_emaildomain"] if "R_emaildomain" in df else pd.Series(np.nan, index=df.index)
    both = p.notna() & r.notna()
    return pd.DataFrame(
        {
            "amt_log": np.log1p(amt),
            # Non-round cents usually mean a currency conversion (foreign card/merchant).
            "amt_cents": (amt - np.floor(amt)).round(3),
            # TransactionDT has an unknown offset, so "hour" is shifted but still cyclic.
            "hour": (t // 3600) % 24,
            "dow": (t // 86400) % 7,
            "email_match": np.where(both, (p.astype(str) == r.astype(str)).astype(float), np.nan),
        },
        index=df.index,
    )


def _num(v: Any) -> float:
    """Scalar version of ``pd.to_numeric(errors="coerce")``."""
    if v is None or isinstance(v, bool):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _present(v: Any) -> bool:
    return not (v is None or (isinstance(v, float) and math.isnan(v)))


def derived_record(rec: Mapping[str, Any]) -> list[float]:
    """:func:`add_derived` for one transaction, without pandas overhead."""
    amt, t = _num(rec.get(AMOUNT_COL)), _num(rec.get(TIME_COL))
    p, r = rec.get("P_emaildomain"), rec.get("R_emaildomain")
    return [
        math.log1p(amt),
        float(np.round(amt - math.floor(amt), 3)),
        (t // 3600) % 24,
        (t // 86400) % 7,
        float(str(p) == str(r)) if _present(p) and _present(r) else math.nan,
    ]


class FeatureBuilder:
    """Turns raw transactions (+ velocity features) into the model matrix.

    Everything learned from data - which columns are categorical, category codes,
    frequency encodings - is fitted on the training period only and then frozen, so
    validation/test rows and live requests are encoded exactly the same way.
    """

    def __init__(
        self,
        velocity_features: Sequence[str],
        freq_encode: Sequence[str] = (),
        min_category_count: int = 20,
        drop: Sequence[str] = (),
    ):
        self.velocity_features = list(velocity_features)
        self.freq_encode = list(freq_encode)
        self.min_category_count = min_category_count
        self.drop = set(drop)
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.cat_maps: dict[str, dict[str, int]] = {}
        self.freq_maps: dict[str, dict] = {}
        self.feature_names: list[str] = []

    @property
    def base_features(self) -> list[str]:
        v = set(self.velocity_features)
        return [f for f in self.feature_names if f not in v]

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        skip = NON_FEATURES | self.drop | set(self.velocity_features) | set(DERIVED)
        cols = [c for c in df.columns if c not in skip]
        self.num_cols = [c for c in cols if is_numeric_dtype(df[c])]
        self.cat_cols = [c for c in cols if not is_numeric_dtype(df[c])]

        for c in self.cat_cols:
            vc = df[c].dropna().astype(str).value_counts()
            kept = vc[vc >= self.min_category_count].index
            # Codes 0..k-1 for frequent categories; k = "rare or unseen"; NaN = missing.
            self.cat_maps[c] = {v: i for i, v in enumerate(kept)}

        for c in self.freq_encode:
            if c not in df:
                continue
            s = df[c].dropna()
            s = s.astype(str) if c in self.cat_cols else pd.to_numeric(s, errors="coerce")
            self.freq_maps[c] = (s.value_counts() / len(df)).to_dict()

        self.feature_names = (
            self.num_cols
            + DERIVED
            + self.velocity_features
            + [f"{c}_freq" for c in self.freq_maps]
            + self.cat_cols
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model matrix (float32) with columns ``feature_names``; works for one row too."""
        out: dict[str, pd.Series] = {}
        for c in self.num_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
        derived = add_derived(df)
        for c in DERIVED:
            out[c] = derived[c]
        for c in self.velocity_features:
            out[c] = df.get(c, np.nan)
        for c, freq in self.freq_maps.items():
            if c not in df:
                out[f"{c}_freq"] = np.nan
                continue
            s = df[c].astype(str) if c in self.cat_cols else pd.to_numeric(df[c], errors="coerce")
            out[f"{c}_freq"] = s.map(freq).fillna(0.0).where(df[c].notna())
        for c, mapping in self.cat_maps.items():
            if c not in df:
                out[c] = np.nan
                continue
            s = df[c]
            codes = s.astype(str).map(mapping).astype(float)
            codes = codes.where(s.isna() | codes.notna(), float(len(mapping)))
            out[c] = codes.where(s.notna())
        X = pd.DataFrame(out, index=df.index)
        return X[self.feature_names].astype(np.float32)

    def transform_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Same matrix as :meth:`transform`, built row by row in plain Python.

        A one-row DataFrame pays pandas overhead on every column (~60 ms for ~100
        columns); this path takes well under a millisecond and is what the API uses.
        Equality with :meth:`transform` is covered by the tests.
        """
        rows = []
        for rec in records:
            row = [_num(rec.get(c)) for c in self.num_cols]
            row += derived_record(rec)
            row += [_num(rec.get(c)) for c in self.velocity_features]
            for c, freq in self.freq_maps.items():
                v = rec.get(c)
                if not _present(v):
                    row.append(math.nan)
                    continue
                key = str(v) if c in self.cat_maps else _num(v)
                row.append(freq.get(key, 0.0))
            for c, mapping in self.cat_maps.items():
                v = rec.get(c)
                row.append(float(mapping.get(str(v), len(mapping))) if _present(v) else math.nan)
            rows.append(row)
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(self.feature_names))
