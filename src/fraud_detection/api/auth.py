"""API-key authentication with two roles.

* ``scorer`` - the payment switch. It may score live transactions (which updates the
  velocity state) and receives only the decision: no score, threshold or reason codes,
  so a leaked key cannot be used to probe the model.
* ``analyst`` - fraud analysts and operators. Everything a scorer can do, plus scores,
  SHAP reason codes, what-if calls (``update_state=false``), model info and drift.

Keys come from the environment as comma-separated lists (``FRAUD_API_KEYS`` for
scorers, ``FRAUD_ANALYST_API_KEYS`` for analysts) and are sent in the ``X-API-Key``
header. The service refuses to start without keys unless ``FRAUD_AUTH_DISABLED=1``
is set explicitly (local development only).
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Iterable

SCORER = "scorer"
ANALYST = "analyst"
MIN_KEY_LENGTH = 16


def _split(value: str | None) -> list[str]:
    return [k.strip() for k in (value or "").split(",") if k.strip()]


class ApiKeyAuth:
    def __init__(
        self,
        scorer_keys: Iterable[str] = (),
        analyst_keys: Iterable[str] = (),
        disabled: bool = False,
    ):
        self.scorer_keys = [k.encode() for k in scorer_keys]
        self.analyst_keys = [k.encode() for k in analyst_keys]
        self.disabled = disabled

    @classmethod
    def from_env(cls) -> ApiKeyAuth:
        return cls(
            _split(os.environ.get("FRAUD_API_KEYS")),
            _split(os.environ.get("FRAUD_ANALYST_API_KEYS")),
            os.environ.get("FRAUD_AUTH_DISABLED") == "1",
        )

    def check_configured(self) -> None:
        """Fail fast at startup rather than serve an open API by accident."""
        if self.disabled:
            return
        keys = self.scorer_keys + self.analyst_keys
        if not keys:
            raise RuntimeError(
                "no API keys configured: set FRAUD_API_KEYS and/or FRAUD_ANALYST_API_KEYS "
                "(or FRAUD_AUTH_DISABLED=1 for local development only)"
            )
        if any(len(k) < MIN_KEY_LENGTH for k in keys):
            raise RuntimeError(f"API keys must be at least {MIN_KEY_LENGTH} characters long")

    def role(self, key: str | None) -> str | None:
        """Role of the presented key, or None if it is missing or unknown."""
        if self.disabled:
            return ANALYST
        if not key:
            return None
        presented = key.encode()
        # Compare against every key (constant time per comparison, no early exit).
        is_analyst = any([hmac.compare_digest(presented, k) for k in self.analyst_keys])
        is_scorer = any([hmac.compare_digest(presented, k) for k in self.scorer_keys])
        if is_analyst:
            return ANALYST
        return SCORER if is_scorer else None
