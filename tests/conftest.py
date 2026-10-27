import pytest

from fraud_detection.config import Config


def small_config(root) -> Config:
    """A few thousand synthetic transactions: the full pipeline in seconds."""
    return Config.from_dict(
        {
            "data": {
                "source": "synthetic",
                "transaction_path": str(root / "data" / "train_transaction.csv"),
                "identity_path": str(root / "data" / "train_identity.csv"),
                "cache_path": None,
                "synthetic": {
                    "n_clients": 2500,
                    "n_days": 75,
                    "seed": 3,
                    "n_bin_attacks": 25,
                    "n_ato": 80,
                    "n_new_account_fraud": 120,
                },
            },
            "split": {"valid_days": 15, "test_days": 15},
            "evaluation": {"bootstrap": 20},
            "models": {
                "logreg": {"max_iter": 300},
                "lightgbm": {"n_estimators": 80, "num_leaves": 15, "early_stopping_rounds": 20},
            },
            "explain": {"sample_size": 500},
            "output": {
                "artifact_dir": str(root / "artifacts"),
                "report_dir": str(root / "reports"),
            },
        }
    )


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once for the whole test session; returns (config, summary)."""
    from fraud_detection.train import run

    cfg = small_config(tmp_path_factory.mktemp("run"))
    return cfg, run(cfg)
