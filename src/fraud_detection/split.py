"""Out-of-time split: train on the past, validate on the next month, test on the last."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import SplitConfig
from .data import DAY_COL


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray  # boolean masks over the full frame
    valid: np.ndarray
    test: np.ndarray
    train_days: tuple[int, int]
    valid_days: tuple[int, int]
    test_days: tuple[int, int]

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("train", "valid", "test"):
            d[k] = int(d[k].sum())
        return d


def time_split(df: pd.DataFrame, cfg: SplitConfig) -> TimeSplit:
    """Split by calendar day so that every validation/test day is after every train day.

    A random split would put transactions from the same fraud burst (same card, same
    hour) on both sides and leak the answer; it would also hide concept drift, which is
    exactly what the model faces once deployed.
    """
    day = df[DAY_COL].to_numpy()
    last = int(np.nanmax(day))
    test_start = last - cfg.test_days + 1
    valid_start = test_start - cfg.valid_days
    train_end = valid_start - 1 - cfg.gap_days
    first = int(np.nanmin(day))
    if train_end <= first:
        raise ValueError("not enough history for the requested valid/test/gap days")
    return TimeSplit(
        train=day <= train_end,
        valid=(day >= valid_start) & (day < test_start),
        test=day >= test_start,
        train_days=(first, train_end),
        valid_days=(valid_start, test_start - 1),
        test_days=(test_start, last),
    )
