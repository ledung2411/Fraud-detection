"""FastAPI app: real-time scoring, reason codes and drift monitoring.

FRAUD_ARTIFACTS=artifacts/synthetic uvicorn fraud_detection.api.app:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query

from .schemas import BatchRequest, BatchResponse, ScoreResponse, Transaction
from .service import ScoringService


def create_app(artifact_dir: str | None = None) -> FastAPI:
    artifact_dir = artifact_dir or os.environ.get("FRAUD_ARTIFACTS", "artifacts/synthetic")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = ScoringService.from_dir(artifact_dir)
        yield

    app = FastAPI(
        title="Fraud scoring API",
        version="0.1.0",
        description="LightGBM fraud score with velocity features, SHAP reason codes and PSI drift checks.",
        lifespan=lifespan,
    )

    def svc() -> ScoringService:
        return app.state.service

    update_q = Query(True, description="add the transaction(s) to the velocity state")
    explain_q = Query("review", description="which rows get SHAP reason codes")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": svc().meta["model_version"]}

    @app.get("/model")
    def model_info() -> dict:
        return svc().info()

    # Plain `def` endpoints run in FastAPI's thread pool; the service serialises
    # access to the velocity state with a lock.
    @app.post("/score", response_model=ScoreResponse)
    def score(
        txn: Transaction,
        update_state: bool = update_q,
        explain: Literal["review", "all", "none"] = explain_q,
    ) -> dict:
        return svc().score([txn.model_dump()], update_state, explain)[0]

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(
        req: BatchRequest,
        update_state: bool = update_q,
        explain: Literal["review", "all", "none"] = explain_q,
    ) -> dict:
        txns = sorted(
            (t.model_dump() for t in req.transactions),
            key=lambda t: (t["TransactionDT"], t["TransactionID"]),
        )
        return {"results": svc().score(txns, update_state, explain)}

    @app.get("/monitoring/drift")
    def drift() -> dict:
        return svc().drift()

    return app


app = create_app()
