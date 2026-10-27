"""Request / response models of the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "TransactionID": 3577540,
    "TransactionDT": 15811047,
    "TransactionAmt": 49.0,
    "ProductCD": "W",
    "card1": 2803,
    "card2": 100.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 299.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "D1": 0.0,
    "C1": 1.0,
    "M4": "M0",
}


class Transaction(BaseModel):
    """One transaction in IEEE-CIS format.

    Only the three first fields are required. Any other IEEE-CIS column (C1-C14,
    D1-D15, M1-M9, V1-V339, id_01-id_38, DeviceType, DeviceInfo, ...) may be sent and
    is used if the model knows it; missing columns are treated as missing values.
    """

    model_config = ConfigDict(extra="allow", json_schema_extra={"examples": [EXAMPLE]})

    TransactionID: int
    TransactionDT: int = Field(description="seconds since the dataset's reference time")
    TransactionAmt: float = Field(gt=0)
    ProductCD: str | None = None
    card1: int | None = None
    addr1: float | None = None
    D1: float | None = Field(default=None, description="days since the client relationship started")
    P_emaildomain: str | None = None


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Reason(BaseModel):
    feature: str
    description: str
    value: float | str | None
    contribution: float = Field(description="SHAP value, log-odds")


class ScoreResponse(BaseModel):
    TransactionID: int
    fraud_score: float
    decision: str = Field(description="'review' if the score is above the budget threshold")
    threshold: float
    reasons: list[Reason]
    model_version: str


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
