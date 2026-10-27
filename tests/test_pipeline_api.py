import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.replay import replay

MODELS = {"logreg_base", "logreg_velocity", "lgbm_base", "lgbm_velocity"}


def test_training_writes_report_and_artifacts(trained):
    cfg, summary = trained
    assert set(summary["results"]) == MODELS
    rep = Path(cfg.output.report_dir)
    assert (rep / "report.md").exists() and json.loads((rep / "metrics.json").read_text())
    for fig in summary["figures"]:
        assert (rep / "figures" / fig).exists(), fig
    for name in (
        "model.txt",
        "feature_builder.joblib",
        "velocity_store.joblib",
        "drift_reference.json",
        "metadata.json",
    ):
        assert (Path(cfg.output.artifact_dir) / name).exists()
    test = summary["results"]["lgbm_velocity"]["test"]
    assert test["pr_auc"] > 3 * test["fraud_rate"]  # far better than random ranking
    split = summary["split"]
    assert (
        split["train_days"][1]
        < split["valid_days"][0]
        <= split["valid_days"][1]
        < split["test_days"][0]
    )


@pytest.fixture(scope="module")
def client(trained):
    cfg, _ = trained
    with TestClient(create_app(cfg.output.artifact_dir)) as c:
        yield c


TXN = {
    "TransactionID": 99_000_001,
    "TransactionDT": 86400 * 80,
    "TransactionAmt": 75.5,
    "ProductCD": "C",
    "card1": 5000,
    "addr1": 299.0,
    "D1": 0.0,
    "P_emaildomain": "gmail.com",
    "card4": "visa",
}


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["review_budget"] == 0.01 and info["n_velocity_features"] > 0


def test_score_contract(client):
    r = client.post("/score", params={"explain": "all", "update_state": False}, json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"review", "approve"}
    assert body["reasons"] and all(x["contribution"] > 0 for x in body["reasons"])
    cat_reasons = [
        x for x in body["reasons"] if x["feature"] in {"ProductCD", "P_emaildomain", "card4"}
    ]
    assert all(isinstance(x["value"], str) for x in cat_reasons)


def test_velocity_state_updates_only_when_asked(client):
    txn = {**TXN, "TransactionID": 99_000_002, "card1": 4242}
    first = client.post("/score", params={"update_state": False}, json=txn).json()["fraud_score"]
    again = client.post("/score", params={"update_state": False}, json=txn).json()["fraud_score"]
    assert first == again
    burst = [
        {
            **txn,
            "TransactionID": 99_000_010 + i,
            "TransactionDT": txn["TransactionDT"] + 60 * i,
            "TransactionAmt": 1.0,
        }
        for i in range(8)
    ]
    client.post("/score/batch", json={"transactions": burst})
    after = client.post(
        "/score",
        params={"update_state": False, "explain": "all"},
        json={**txn, "TransactionDT": txn["TransactionDT"] + 600},
    ).json()
    assert after["fraud_score"] != first
    assert any(x["feature"].startswith(("card1_", "uid_")) for x in after["reasons"])


def test_validation_errors(client):
    assert client.post("/score", json={"TransactionID": 1, "TransactionDT": 1}).status_code == 422
    assert client.post("/score", json={**TXN, "TransactionAmt": -5}).status_code == 422
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_drift_endpoint_needs_traffic_then_reports(trained):
    cfg, _ = trained
    with TestClient(create_app(cfg.output.artifact_dir)) as c:
        assert c.get("/monitoring/drift").json()["status"] == "insufficient_data"
        batch = [
            {**TXN, "TransactionID": 1 + i, "TransactionDT": TXN["TransactionDT"] + i}
            for i in range(250)
        ]
        c.post("/score/batch", json={"transactions": batch})
        d = c.get("/monitoring/drift").json()
        assert d["n_rows"] == 250 and d["status"] in {"stable", "moderate", "significant"}
        assert "score_psi" in d and "alert_rate" in d


def test_replay_has_no_training_serving_skew(trained):
    cfg, _ = trained
    out = replay(cfg, limit=400)
    assert out["transactions"] == 400
    assert out["max_abs_score_diff_online_vs_offline"] < 1e-9
