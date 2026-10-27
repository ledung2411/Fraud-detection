import math

import numpy as np
import pandas as pd

from fraud_detection.data import add_time_columns, sort_chronologically
from fraud_detection.features import FeatureBuilder
from fraud_detection.synthetic import generate
from fraud_detection.velocity import compute_velocity

ENT = {"card1": ["card1"], "uid": ["card1", "addr1", "D1n"]}
WIN = {"1h": 3600, "24h": 86400}


def _data():
    tx, ident = generate(
        n_clients=1500, n_days=40, seed=5, n_bin_attacks=10, n_ato=30, n_new_account_fraud=40
    )
    df = add_time_columns(sort_chronologically(tx.merge(ident, on="TransactionID", how="left")))
    vel = compute_velocity(df, ENT, WIN)
    return pd.concat([df, vel], axis=1), list(vel.columns)


def test_row_path_equals_frame_path():
    df, vel = _data()
    b = FeatureBuilder(vel, ["card1", "addr1", "P_emaildomain", "DeviceInfo"], 5).fit(
        df.iloc[:5000]
    )
    a = b.transform(df).to_numpy()
    r = b.transform_records(df.to_dict("records"))
    assert a.shape == r.shape
    assert ((a == r) | (np.isnan(a) & np.isnan(r))).all()


def test_encoders_fitted_on_train_only_and_handle_unseen():
    df, vel = _data()
    train = df.iloc[:5000]
    b = FeatureBuilder(vel, ["P_emaildomain"], 5).fit(train)
    assert "isFraud" not in b.feature_names and "TransactionDT" not in b.feature_names
    assert "D1n" not in b.feature_names
    probe = df.iloc[[0, 1]].copy()
    probe["P_emaildomain"] = ["never-seen.example", None]
    X = b.transform(probe)
    rare_code = len(b.cat_maps["P_emaildomain"])
    assert X["P_emaildomain"].iloc[0] == rare_code
    assert math.isnan(X["P_emaildomain"].iloc[1])
    assert X["P_emaildomain_freq"].iloc[0] == 0.0
    assert math.isnan(X["P_emaildomain_freq"].iloc[1])
    rec = b.transform_records(probe.to_dict("records"))
    np.testing.assert_array_equal(np.nan_to_num(rec, nan=-9), np.nan_to_num(X.to_numpy(), nan=-9))


def test_missing_columns_are_missing_values():
    df, vel = _data()
    b = FeatureBuilder(vel, ["card1"], 5).fit(df.iloc[:3000])
    minimal = [{"TransactionID": 1, "TransactionDT": 86400 * 50, "TransactionAmt": 12.5}]
    X = b.transform_records(minimal)
    assert X.shape == (1, len(b.feature_names))
    assert np.isnan(X[0, b.feature_names.index("C1")])
    assert X[0, b.feature_names.index("amt_cents")] == np.float32(0.5)
