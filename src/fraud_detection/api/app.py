"""FastAPI app: real-time scoring, reason codes and drift monitoring.

FRAUD_ARTIFACTS=artifacts/synthetic FRAUD_API_KEYS=... FRAUD_ANALYST_API_KEYS=... \\
    uvicorn fraud_detection.api.app:app --port 8000

Environment:
    FRAUD_ARTIFACTS            trained bundle directory (default artifacts/synthetic)
    FRAUD_API_KEYS             comma-separated keys for the payment switch (role scorer)
    FRAUD_ANALYST_API_KEYS     comma-separated keys for analysts (role analyst)
    FRAUD_AUTH_DISABLED=1      no authentication - local development only
    FRAUD_MAX_FUTURE_SKEW_S    how far ahead of the service clock an event may be dated
                               and still update the velocity state (default 300; "off"
                               only for offline replays faster than real time)
    FRAUD_MAX_ENTITIES         cap on cards/clients kept per entity type (default 1e6)
    FRAUD_ENTITY_TTL_DAYS      forget entities idle for longer than this (default 400)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from ..velocity import DEFAULT_MAX_ENTITIES, DEFAULT_TTL_S
from .auth import ANALYST, ApiKeyAuth
from .schemas import BatchRequest, BatchResponse, ScoreResponse, Transaction
from .service import DEFAULT_MAX_FUTURE_SKEW_S, ScoringService

_UNSET = object()


def _env_number(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return None if raw.strip().lower() in {"off", "none"} else float(raw)


def create_app(
    artifact_dir: str | None = None,
    auth: ApiKeyAuth | None = None,
    max_future_skew_s: float | None | object = _UNSET,
) -> FastAPI:
    artifact_dir = artifact_dir or os.environ.get("FRAUD_ARTIFACTS", "artifacts/synthetic")
    auth = auth or ApiKeyAuth.from_env()
    if max_future_skew_s is _UNSET:
        max_future_skew_s = _env_number("FRAUD_MAX_FUTURE_SKEW_S", DEFAULT_MAX_FUTURE_SKEW_S)
    ttl_days = _env_number("FRAUD_ENTITY_TTL_DAYS", DEFAULT_TTL_S / 86400)
    max_entities = _env_number("FRAUD_MAX_ENTITIES", DEFAULT_MAX_ENTITIES)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        auth.check_configured()
        app.state.service = ScoringService.from_dir(
            artifact_dir,
            max_future_skew_s=max_future_skew_s,
            max_entities=None if max_entities is None else int(max_entities),
            entity_ttl_s=None if ttl_days is None else int(ttl_days * 86400),
        )
        yield

    app = FastAPI(
        title="Fraud scoring API",
        version="0.1.0",
        description="LightGBM fraud score with velocity features, SHAP reason codes and PSI drift checks.",
        lifespan=lifespan,
    )

    def svc() -> ScoringService:
        return app.state.service

    def caller_role(x_api_key: str | None = Header(None)) -> str:
        role = auth.role(x_api_key)
        if role is None:
            raise HTTPException(401, "missing or invalid API key", {"WWW-Authenticate": "ApiKey"})
        return role

    def analyst_only(role: str = Depends(caller_role)) -> str:
        if role != ANALYST:
            raise HTTPException(403, "this endpoint needs an analyst API key")
        return role

    def present(results: list[dict], role: str) -> list[dict]:
        """The payment switch gets the decision only; analysts get score and reasons."""
        if role == ANALYST:
            return results
        keep = ("TransactionID", "decision", "state_updated", "model_version")
        return [{k: r[k] for k in keep} for r in results]

    def options(role: str, update_state: bool, explain: str) -> str:
        if role != ANALYST and (not update_state or explain == "all"):
            raise HTTPException(403, "update_state=false and explain=all need an analyst API key")
        return explain if role == ANALYST else "none"

    update_q = Query(True, description="add the transaction(s) to the velocity state")
    explain_q = Query("review", description="which rows get SHAP reason codes (analysts)")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model")
    def model_info(_: str = Depends(analyst_only)) -> dict:
        return svc().info()

    # Plain `def` endpoints run in FastAPI's thread pool; the service serialises
    # access to the velocity state with a lock.
    @app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(
        txn: Transaction,
        update_state: bool = update_q,
        explain: Literal["review", "all", "none"] = explain_q,
        role: str = Depends(caller_role),
    ) -> dict:
        explain = options(role, update_state, explain)
        return present(svc().score([txn.model_dump()], update_state, explain), role)[0]

    @app.post("/score/batch", response_model=BatchResponse, response_model_exclude_none=True)
    def score_batch(
        req: BatchRequest,
        update_state: bool = update_q,
        explain: Literal["review", "all", "none"] = explain_q,
        role: str = Depends(caller_role),
    ) -> dict:
        explain = options(role, update_state, explain)
        txns = sorted(
            (t.model_dump() for t in req.transactions),
            key=lambda t: (t["TransactionDT"], t["TransactionID"]),
        )
        return {"results": present(svc().score(txns, update_state, explain), role)}

    @app.get("/monitoring/drift")
    def drift(_: str = Depends(analyst_only)) -> dict:
        return svc().drift()

    return app


app = create_app()
