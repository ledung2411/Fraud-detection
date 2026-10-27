import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.app import create_app
from fraud_detection.api.auth import ApiKeyAuth
from fraud_detection.api.service import ScoringService
from fraud_detection.artifacts import load_bundle
from fraud_detection.replay import replay

MODELS = {"logreg_base", "logreg_velocity", "lgbm_base", "lgbm_velocity"}
SCORER_KEY = "scorer-key-" + "s" * 16
ANALYST_KEY = "analyst-key-" + "a" * 16
SCORER = {"X-API-Key": SCORER_KEY}
ANALYST = {"X-API-Key": ANALYST_KEY}


def make_app(cfg, **kwargs):
    return create_app(cfg.output.artifact_dir, ApiKeyAuth([SCORER_KEY], [ANALYST_KEY]), **kwargs)


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
    """Analyst client; the clock check is off so tests can use arbitrary timestamps."""
    cfg, _ = trained
    with TestClient(make_app(cfg, max_future_skew_s=None), headers=ANALYST) as c:
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


def test_health_is_open_and_model_needs_analyst(client):
    assert client.get("/health", headers={"X-API-Key": ""}).json() == {"status": "ok"}
    info = client.get("/model").json()
    assert info["review_budget"] == 0.01 and info["n_velocity_features"] > 0


def test_requests_without_a_valid_key_are_rejected(client):
    for headers in ({"X-API-Key": ""}, {"X-API-Key": "wrong-key-" + "x" * 16}):
        assert client.post("/score", json=TXN, headers=headers).status_code == 401
        assert (
            client.post("/score/batch", json={"transactions": [TXN]}, headers=headers).status_code
            == 401
        )
        assert client.get("/model", headers=headers).status_code == 401
        assert client.get("/monitoring/drift", headers=headers).status_code == 401


def test_scorer_gets_the_decision_only(client):
    r = client.post("/score", json={**TXN, "TransactionID": 99_000_100}, headers=SCORER)
    assert r.status_code == 200
    assert set(r.json()) == {"TransactionID", "decision", "state_updated", "model_version"}
    batch = client.post("/score/batch", json={"transactions": [TXN]}, headers=SCORER).json()
    assert "fraud_score" not in batch["results"][0]
    # No probing: what-if calls, explanations, model info and drift are analyst-only.
    assert (
        client.post("/score", json=TXN, params={"explain": "all"}, headers=SCORER).status_code
        == 403
    )
    assert (
        client.post("/score", json=TXN, params={"update_state": False}, headers=SCORER).status_code
        == 403
    )
    assert client.get("/model", headers=SCORER).status_code == 403
    assert client.get("/monitoring/drift", headers=SCORER).status_code == 403


def test_service_refuses_to_start_without_keys(trained):
    cfg, _ = trained
    for auth in (ApiKeyAuth(), ApiKeyAuth(scorer_keys=["short"])):
        with pytest.raises(RuntimeError), TestClient(create_app(cfg.output.artifact_dir, auth)):
            pass
    with TestClient(create_app(cfg.output.artifact_dir, ApiKeyAuth(disabled=True))) as c:
        assert c.post("/score", json=TXN, params={"update_state": False}).status_code == 200


def test_score_contract(client):
    r = client.post("/score", params={"explain": "all", "update_state": False}, json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"review", "approve"}
    assert body["state_updated"] is False
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
    res = client.post("/score/batch", json={"transactions": burst}).json()["results"]
    assert all(r["state_updated"] for r in res)
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


def _victim_features(svc, t, d1):
    probe = {
        "TransactionDT": t,
        "TransactionAmt": 900.0,
        "card1": 7777,
        "addr1": 123.0,
        "TransactionDay": t // 86400,
        "D1n": t // 86400 - d1,
    }
    f = svc.bundle.store.compute(probe)
    return {
        k: f[k] for k in ("uid_cnt_1h", "uid_amt_sum_24h", "card1_cnt_1h", "card1_secs_since_prev")
    }


def test_forged_timestamps_cannot_rewrite_velocity_history(trained):
    """Regression test: one event dated a month ahead used to erase a card's windows."""
    cfg, _ = trained
    wall = [0.0]
    svc = ScoringService(load_bundle(cfg.output.artifact_dir), wall_clock=lambda: wall[0])
    t0 = int(svc.clock.start_t)
    victim = {"card1": 7777, "addr1": 123.0, "D1": 10.0}
    history = [
        {"TransactionID": 1 + i, "TransactionDT": t0 + 40 * i, "TransactionAmt": 20.0, **victim}
        for i in range(6)
    ]
    assert all(r["state_updated"] for r in svc.score(history))
    before = _victim_features(svc, t0 + 240, 10.0)
    assert before["uid_cnt_1h"] == 6 and before["card1_cnt_1h"] == 6

    day = 86400
    forged_future = {
        **history[0],
        "TransactionID": 50,
        "TransactionDT": t0 + 30 * day,
        "TransactionAmt": 1.0,
        "D1": 40.0,
    }
    late = {**history[0], "TransactionID": 51, "TransactionDT": t0 - 1000}
    res = svc.score([forged_future, late])
    assert [r["state_updated"] for r in res] == [False, False]
    assert svc.not_recorded == {"future_timestamp": 1, "out_of_order": 1}
    assert _victim_features(svc, t0 + 240, 10.0) == before

    # The clock follows real time, not event time: slightly-future events cannot ratchet it.
    steps = [
        {**history[0], "TransactionID": 60 + i, "TransactionDT": t0 + 250 * (i + 1)}
        for i in range(3)
    ]
    assert [r["state_updated"] for r in svc.score(steps)] == [True, False, False]
    wall[0] += 3600  # an hour of real time later, events an hour later are fine
    assert svc.score([{**steps[-1], "TransactionID": 70, "TransactionDT": t0 + 3600}])[0][
        "state_updated"
    ]


def test_state_size_is_bounded(trained):
    cfg, _ = trained
    svc = ScoringService(
        load_bundle(cfg.output.artifact_dir), max_future_skew_s=None, max_entities=1000
    )
    start = len(svc.bundle.store.state["card1"])
    assert start <= 1000  # cap applied to the loaded state too
    t0 = int(svc.clock.start_t)
    batch = [
        {
            "TransactionID": i,
            "TransactionDT": t0 + i,
            "TransactionAmt": 1.0,
            "card1": 900_000 + i,
            "addr1": 1.0,
            "D1": 0.0,
        }
        for i in range(2500)
    ]
    svc.score(batch)
    assert len(svc.bundle.store.state["card1"]) == 1000
    assert len(svc.bundle.store.state["uid"]) == 1000
    assert svc.info()["entities_evicted"]["card1"] >= 2500 - 1000


def test_drift_endpoint_needs_traffic_then_reports(trained):
    cfg, _ = trained
    with TestClient(make_app(cfg, max_future_skew_s=None), headers=ANALYST) as c:
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
