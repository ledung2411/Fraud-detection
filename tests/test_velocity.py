import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import add_time_columns, sort_chronologically
from fraud_detection.synthetic import generate
from fraud_detection.velocity import OnlineVelocityStore, compute_velocity

ENT = {"card1": ["card1"], "uid": ["card1", "addr1", "D1n"]}
WIN = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}


def _frame(rows):
    df = pd.DataFrame(
        rows, columns=["TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1"]
    )
    return add_time_columns(df)


@pytest.fixture(scope="module")
def stream():
    tx, _ = generate(
        n_clients=1500, n_days=40, seed=11, n_bin_attacks=10, n_ato=40, n_new_account_fraud=50
    )
    return add_time_columns(sort_chronologically(tx))


def test_hand_computed_windows():
    t0 = 86400 * 10
    df = _frame(
        [
            (1, t0, 10.0, 7, 100.0, 5.0),
            (2, t0 + 1000, 20.0, 7, 100.0, 5.0),
            (3, t0 + 4000, 60.0, 7, 100.0, 5.0),  # 1h window starts at t0+400: only txn 2 is in it
            (4, t0 + 4000, 5.0, 8, 100.0, 5.0),  # other card: sees nothing
        ]
    )
    f = compute_velocity(df, {"card1": ["card1"]}, {"1h": 3600, "24h": 86400})
    assert f["card1_cnt_1h"].tolist() == [0, 1, 1, 0]
    assert f["card1_cnt_24h"].tolist() == [0, 1, 2, 0]
    assert f["card1_amt_sum_24h"].tolist() == [0, 10, 30, 0]
    assert f.loc[2, "card1_amt_ratio_24h"] == pytest.approx(60 / 15)
    assert math.isnan(f.loc[0, "card1_amt_ratio_1h"])
    assert f["card1_secs_since_prev"].fillna(-1).tolist() == [-1, 1000, 3000, -1]
    assert f["card1_n_prior"].tolist() == [0, 1, 2, 0]


def test_same_second_ties_follow_transaction_id():
    t = 86400 * 3
    df = _frame([(5, t, 1.0, 7, 1.0, 0.0), (4, t, 2.0, 7, 1.0, 0.0)])
    f = compute_velocity(df, {"card1": ["card1"]}, {"1h": 3600})
    # ID 4 comes first in the stream, so ID 5 (row 0) is the one that sees a prior txn.
    assert f["card1_cnt_1h"].tolist() == [1, 0]


def test_missing_key_gives_missing_features():
    df = _frame([(1, 86400, 10.0, 7, np.nan, 1.0), (2, 86401, 10.0, 7, np.nan, 1.0)])
    f = compute_velocity(df, ENT, WIN)
    assert f["uid_cnt_1h"].isna().all()
    assert f["card1_cnt_1h"].tolist() == [0, 1]


def test_no_lookahead(stream):
    """Changing the future must not change features of the past."""
    cut = stream["TransactionDT"].quantile(0.6)
    before = compute_velocity(stream, ENT, WIN)
    future = stream.copy()
    late = future["TransactionDT"] > cut
    future.loc[late, "TransactionAmt"] *= 7
    future = pd.concat(
        [future, future[late].assign(TransactionID=lambda d: d.TransactionID + 10**7)]
    )
    after = compute_velocity(future, ENT, WIN).iloc[: len(stream)]
    early = ~late.to_numpy()
    pd.testing.assert_frame_equal(before[early], after[early])


def _assert_close(a: pd.DataFrame, b: pd.DataFrame):
    for c in a.columns:
        np.testing.assert_allclose(
            a[c].to_numpy(float),
            b[c].to_numpy(float),
            rtol=1e-5,
            atol=1e-4,
            equal_nan=True,
            err_msg=c,
        )


def test_online_store_matches_offline(stream):
    offline = compute_velocity(stream, ENT, WIN).reset_index(drop=True)
    store = OnlineVelocityStore(ENT, WIN)
    online = pd.DataFrame([store.compute_and_update(r) for r in stream.to_dict("records")])
    _assert_close(offline, online)


def test_store_built_from_history_matches_offline(stream):
    offline = compute_velocity(stream, ENT, WIN).reset_index(drop=True)
    half = len(stream) // 2
    store = OnlineVelocityStore.from_frame(stream.iloc[:half], ENT, WIN)
    online = pd.DataFrame(
        [store.compute_and_update(r) for r in stream.iloc[half:].to_dict("records")]
    )
    _assert_close(offline.iloc[half:].reset_index(drop=True), online)


def test_compute_does_not_mutate(stream):
    store = OnlineVelocityStore.from_frame(stream, ENT, WIN)
    rec = stream.iloc[-1].to_dict()
    assert store.compute(rec) == store.compute(rec)
