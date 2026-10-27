"""Loading IEEE-CIS (or the synthetic stand-in) into one chronologically sorted frame."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

log = logging.getLogger(__name__)

ID_COL = "TransactionID"
TARGET = "isFraud"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
DAY_COL = "TransactionDay"
SECONDS_PER_DAY = 86400

# V/C/D columns are numeric and make up most of the 434 columns; reading them as
# float32 roughly halves peak memory on the 590k-row Kaggle file.
_FLOAT32_RE = re.compile(r"^(V|C|D)\d+$")


def read_ieee(transaction_path: str | Path, identity_path: str | Path | None) -> pd.DataFrame:
    """Read the Kaggle ``train_transaction.csv`` (+ optional ``train_identity.csv``)."""
    header = pd.read_csv(transaction_path, nrows=0).columns
    dtypes = {c: np.float32 for c in header if _FLOAT32_RE.match(c)}
    tx = pd.read_csv(transaction_path, dtype=dtypes)
    if identity_path and Path(identity_path).exists():
        identity = pd.read_csv(identity_path)
        # Kaggle's test identity file uses "id-01"; normalise to the train spelling.
        identity.columns = [c.replace("id-", "id_") for c in identity.columns]
        tx = tx.merge(identity, on=ID_COL, how="left")
    else:
        log.warning(
            "identity file not found (%s); continuing with transactions only", identity_path
        )
    return sort_chronologically(tx)


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Order by event time, ties broken by TransactionID - the order a stream would see."""
    return df.sort_values([TIME_COL, ID_COL], kind="mergesort").reset_index(drop=True)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the day index and ``D1n``, the day the client relationship started.

    D1 is "days since the client started"; ``day - D1`` is therefore constant for a
    client and, combined with card1/addr1, gives the ``uid`` used for velocity features.
    Works on a single-row frame too, so the API derives it the same way.
    """
    df = df.copy()
    dt = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[DAY_COL] = np.floor(dt / SECONDS_PER_DAY)
    d1 = pd.to_numeric(df["D1"], errors="coerce") if "D1" in df else np.nan
    df["D1n"] = df[DAY_COL] - d1
    return df


def add_time_fields(rec: dict) -> dict:
    """:func:`add_time_columns` for a single transaction dict (API fast path)."""
    out = dict(rec)
    out[DAY_COL] = float(math.floor(float(rec[TIME_COL]) / SECONDS_PER_DAY))
    d1 = rec.get("D1")
    out["D1n"] = math.nan if d1 is None else out[DAY_COL] - float(d1)
    return out


def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    """Return the labelled dataset, generating synthetic data on first use if configured."""
    if cfg.cache_path and Path(cfg.cache_path).exists():
        log.info("loading cached dataset %s", cfg.cache_path)
        return pd.read_parquet(cfg.cache_path)

    tx_path = Path(cfg.transaction_path)
    if not tx_path.exists():
        if cfg.source != "synthetic":
            raise FileNotFoundError(
                f"{tx_path} not found. Download the competition data with\n"
                "  kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-fraud-detection\n"
                "and unzip it, or run with configs/synthetic.yaml. See data/README.md."
            )
        from .synthetic import write

        log.info("generating synthetic dataset into %s", tx_path.parent)
        write(tx_path.parent, **cfg.synthetic)

    df = read_ieee(tx_path, cfg.identity_path)
    if cfg.cache_path:
        Path(cfg.cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cfg.cache_path, index=False)
    return df
