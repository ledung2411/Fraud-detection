"""Velocity features: how much an entity (card BIN, client) transacted *before* now.

Two implementations of exactly the same definition:

* :func:`compute_velocity` - vectorised, for the whole history at training time.
* :class:`OnlineVelocityStore` - incremental per-entity state, for the scoring API.

Keeping both is the usual batch/streaming split in production. Their equality is
covered by tests (``tests/test_velocity.py``) and checked again by ``replay.py``,
because a silent mismatch between them (training/serving skew) is one of the most
common ways a fraud model degrades after deployment.

Definitions, for transaction *i* of entity *e* at time *t_i* (stream order is
``(TransactionDT, TransactionID)``; only strictly earlier stream positions count, so
the current transaction never sees itself and no label is ever used):

* ``{e}_cnt_{w}`` / ``{e}_amt_sum_{w}``: count / amount of prior transactions with
  ``t_j > t_i - w``.
* ``{e}_amt_ratio_{w}``: current amount / mean prior amount in the window.
* ``{e}_n_prior``: number of prior transactions ever seen for the entity.
* ``{e}_secs_since_prev``: seconds since the previous transaction.
* ``{e}_amt_mean_hist`` / ``{e}_amt_z_hist``: mean of prior amounts and the deviation
  of the current amount from it, in standard deviations (std floored at 1 currency
  unit so a handful of identical amounts cannot blow it up).
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .data import AMOUNT_COL, ID_COL, TIME_COL

STD_FLOOR = 1.0


def feature_names(entities: Mapping[str, list[str]], windows: Mapping[str, int]) -> list[str]:
    names = []
    for e in entities:
        for w in windows:
            names += [f"{e}_cnt_{w}", f"{e}_amt_sum_{w}", f"{e}_amt_ratio_{w}"]
        # No "days since first seen": the first sighting is the start of the data, so
        # that feature is a disguised timestamp (the drift report flagged it, PSI > 1).
        names += [f"{e}_n_prior", f"{e}_secs_since_prev", f"{e}_amt_mean_hist", f"{e}_amt_z_hist"]
    return names


def _canonical(v: Any) -> Any:
    """Hashable, type-stable key component; None when missing."""
    if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    return str(v)


def entity_key(row: Mapping[str, Any], cols: list[str]) -> tuple | None:
    """Key of one transaction for one entity; None if any component is missing."""
    parts = tuple(_canonical(row.get(c)) for c in cols)
    return None if any(p is None for p in parts) else parts


def _group_codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Integer code per row for the entity (-1 when any key column is missing)."""
    missing = np.zeros(len(df), dtype=bool)
    for c in cols:
        missing |= df[c].isna().to_numpy()
    codes = df.groupby(cols, dropna=True, sort=False).ngroup().to_numpy()
    return np.where(missing, -1, codes).astype(np.int64)


def compute_velocity(
    df: pd.DataFrame, entities: Mapping[str, list[str]], windows: Mapping[str, int]
) -> pd.DataFrame:
    """Velocity features for every row of ``df`` (any order), using all rows as history."""
    n = len(df)
    order = np.lexsort((df[ID_COL].to_numpy(), df[TIME_COL].to_numpy()))
    t_all = df[TIME_COL].to_numpy(dtype=np.int64)[order]
    a_all = df[AMOUNT_COL].to_numpy(dtype=np.float64)[order]
    out = {name: np.full(n, np.nan) for name in feature_names(entities, windows)}
    max_w = max(windows.values())

    for e, cols in entities.items():
        codes = _group_codes(df, cols)[order]
        sel = np.flatnonzero(codes >= 0)
        if len(sel) == 0:
            continue
        # Stable sort by entity keeps chronological order inside each entity.
        sel = sel[np.argsort(codes[sel], kind="stable")]
        c, t, a = codes[sel], t_all[sel], a_all[sel]
        pos = np.arange(len(sel))
        is_start = np.r_[True, c[1:] != c[:-1]]
        start = np.maximum.accumulate(np.where(is_start, pos, 0))
        n_prior = pos - start

        # Per-entity running sums of strictly prior rows (no global cumsum, so no
        # precision loss on large data).
        grp = pd.Series(c)
        csum = pd.Series(a).groupby(grp).cumsum().to_numpy() - a
        csq = pd.Series(a * a).groupby(grp).cumsum().to_numpy() - a * a

        # The composite (entity, time) key is non-decreasing; a binary search finds, for
        # each row, the first prior row of the same entity still inside the window.
        big = int(t.max()) + max_w + 1
        comp = c * big + t
        res: dict[str, np.ndarray] = {}
        for w_name, w in windows.items():
            lo = np.searchsorted(comp, comp - w, side="right")
            cnt = pos - lo
            # csum[x] is the sum of the entity's rows before x, so this is rows lo..pos-1.
            s = np.where(cnt > 0, csum - csum[lo], 0.0)
            res[f"{e}_cnt_{w_name}"] = cnt.astype(float)
            res[f"{e}_amt_sum_{w_name}"] = s
            with np.errstate(divide="ignore", invalid="ignore"):
                res[f"{e}_amt_ratio_{w_name}"] = np.where(
                    (cnt > 0) & (s > 0), a / (s / cnt), np.nan
                )

        prev_t = np.r_[0, t[:-1]]
        res[f"{e}_n_prior"] = n_prior.astype(float)
        res[f"{e}_secs_since_prev"] = np.where(n_prior > 0, t - prev_t, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_prior > 0, csum / n_prior, np.nan)
            var = np.where(n_prior > 1, csq / n_prior - mean * mean, np.nan)
            std = np.maximum(np.sqrt(np.maximum(var, 0.0)), STD_FLOOR)
        res[f"{e}_amt_mean_hist"] = mean
        res[f"{e}_amt_z_hist"] = np.where(n_prior > 1, (a - mean) / std, np.nan)

        rows = order[sel]
        for k, v in res.items():
            out[k][rows] = v

    return pd.DataFrame(out, index=df.index).astype(np.float32)


class OutOfOrderEventError(ValueError):
    """An event older than the last recorded event of one of its entities."""


class _EntityState:
    __slots__ = ("events", "n", "s", "s2", "last_t")

    def __init__(self) -> None:
        self.events: deque[tuple[int, float]] = deque()
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.last_t = 0


# Bounds on the state, so traffic cannot grow it without limit: at most this many
# entities per entity type (least recently seen evicted first) and none idle for longer
# than the TTL. Both are far above what IEEE-CIS needs (~250k clients, 182 days), so
# they never change a feature on the dataset.
DEFAULT_MAX_ENTITIES = 1_000_000
DEFAULT_TTL_S = 400 * 86400


class OnlineVelocityStore:
    """Per-entity state that produces the same features one transaction at a time.

    In production this state would live in Redis or a feature store, keyed the same
    way; an in-process dict keeps the project self-contained. Events must be recorded
    in stream order: :meth:`update` refuses an event older than the last event of any of
    its entities, so a stray or forged timestamp cannot reorder or erase history.
    """

    def __init__(
        self,
        entities: Mapping[str, list[str]],
        windows: Mapping[str, int],
        max_entities: int | None = DEFAULT_MAX_ENTITIES,
        ttl_s: int | None = DEFAULT_TTL_S,
    ):
        self.entities = {k: list(v) for k, v in entities.items()}
        self.windows = dict(windows)
        self.max_w = max(self.windows.values())
        self.max_entities = max_entities
        self.ttl_s = ttl_s
        # Insertion order = least recently updated first, which makes LRU/TTL eviction O(1).
        self.state: dict[str, OrderedDict[tuple, _EntityState]] = {
            e: OrderedDict() for e in self.entities
        }
        self.evicted = dict.fromkeys(self.entities, 0)
        self.last_seen_t: int | None = None

    @property
    def feature_names(self) -> list[str]:
        return feature_names(self.entities, self.windows)

    def compute(self, txn: Mapping[str, Any]) -> dict[str, float]:
        """Features for ``txn`` from the current state (the state is not modified)."""
        t = int(txn[TIME_COL])
        a = float(txn[AMOUNT_COL])
        out = dict.fromkeys(self.feature_names, math.nan)
        for e, cols in self.entities.items():
            key = entity_key(txn, cols)
            if key is None:
                continue
            st = self.state[e].get(key)
            n = st.n if st else 0
            for w_name, w in self.windows.items():
                cnt, s = 0, 0.0
                if st:
                    for tj, aj in reversed(st.events):
                        if tj > t:  # recorded after a late event: not in its past
                            continue
                        if tj <= t - w:
                            break
                        cnt += 1
                        s += aj
                out[f"{e}_cnt_{w_name}"] = float(cnt)
                out[f"{e}_amt_sum_{w_name}"] = s
                out[f"{e}_amt_ratio_{w_name}"] = a / (s / cnt) if cnt > 0 and s > 0 else math.nan
            out[f"{e}_n_prior"] = float(n)
            if n > 0:
                out[f"{e}_secs_since_prev"] = float(max(t - st.last_t, 0))
                mean = st.s / n
                out[f"{e}_amt_mean_hist"] = mean
                if n > 1:
                    std = max(math.sqrt(max(st.s2 / n - mean * mean, 0.0)), STD_FLOOR)
                    out[f"{e}_amt_z_hist"] = (a - mean) / std
        return out

    def late_entities(self, txn: Mapping[str, Any]) -> list[str]:
        """Entity types whose last recorded event is newer than ``txn``."""
        t = int(txn[TIME_COL])
        late = []
        for e, cols in self.entities.items():
            key = entity_key(txn, cols)
            st = self.state[e].get(key) if key is not None else None
            if st is not None and st.last_t > t:
                late.append(e)
        return late

    def update(self, txn: Mapping[str, Any]) -> None:
        """Append ``txn`` to the state of each of its entities.

        Raises :class:`OutOfOrderEventError` (and changes nothing) if ``txn`` is older
        than the last event of one of its entities.
        """
        late = self.late_entities(txn)
        if late:
            raise OutOfOrderEventError(f"event is older than the last event of: {', '.join(late)}")
        t = int(txn[TIME_COL])
        a = float(txn[AMOUNT_COL])
        self.last_seen_t = t if self.last_seen_t is None else max(self.last_seen_t, t)
        for e, cols in self.entities.items():
            key = entity_key(txn, cols)
            if key is None:
                continue
            states = self.state[e]
            st = states.get(key)
            if st is None:
                st = states[key] = _EntityState()
            else:
                states.move_to_end(key)
            st.events.append((t, a))
            while st.events[0][0] <= t - self.max_w:
                st.events.popleft()
            st.n += 1
            st.s += a
            st.s2 += a * a
            st.last_t = t
            self._evict(e, t)

    def apply_limits(self, max_entities: int | None, ttl_s: int | None) -> None:
        """Change the bounds and enforce them on the current state."""
        self.max_entities, self.ttl_s = max_entities, ttl_s
        for e in self.state:
            self._evict(e, self.last_seen_t or 0)

    def _evict(self, e: str, now: int) -> None:
        states = self.state[e]
        while states:
            oldest = next(iter(states.values()))
            too_many = self.max_entities is not None and len(states) > self.max_entities
            expired = self.ttl_s is not None and oldest.last_t <= now - self.ttl_s
            if not (too_many or expired):
                break
            states.popitem(last=False)
            self.evicted[e] += 1

    def compute_and_update(self, txn: Mapping[str, Any]) -> dict[str, float]:
        feats = self.compute(txn)
        self.update(txn)
        return feats

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        entities: Mapping[str, list[str]],
        windows: Mapping[str, int],
        max_entities: int | None = DEFAULT_MAX_ENTITIES,
        ttl_s: int | None = DEFAULT_TTL_S,
    ) -> OnlineVelocityStore:
        """State as it would be after streaming every row of ``df`` (vectorised)."""
        store = cls(entities, windows, max_entities, ttl_s)
        if df.empty:
            return store
        t_end = int(df[TIME_COL].max())
        store.last_seen_t = t_end
        df = df.sort_values([TIME_COL, ID_COL], kind="mergesort")
        for e, cols in store.entities.items():
            sub = df.dropna(subset=cols)
            if sub.empty:
                continue
            amt = sub[AMOUNT_COL].astype(float)
            agg = (
                sub.assign(_a=amt, _a2=amt * amt, _row=np.arange(len(sub)))
                .groupby(cols, sort=False)
                .agg(
                    n=(TIME_COL, "size"),
                    s=("_a", "sum"),
                    s2=("_a2", "sum"),
                    last_t=(TIME_COL, "max"),
                    last_row=("_row", "max"),
                )
                .reset_index()
                # Same order a stream would leave behind: least recently updated first.
                .sort_values("last_row", kind="mergesort")
            )
            recent = sub[sub[TIME_COL] > t_end - store.max_w]
            recent_events: dict[tuple, list[tuple[int, float]]] = {}
            for row in recent[[*cols, TIME_COL, AMOUNT_COL]].itertuples(index=False):
                k = entity_key(dict(zip(cols, row[: len(cols)], strict=True)), cols)
                recent_events.setdefault(k, []).append((int(row[-2]), float(row[-1])))
            states = store.state[e]
            for row in agg.itertuples(index=False):
                k = entity_key(dict(zip(cols, row[: len(cols)], strict=True)), cols)
                st = _EntityState()
                st.n, st.s, st.s2 = int(row.n), float(row.s), float(row.s2)
                st.last_t = int(row.last_t)
                st.events = deque(recent_events.get(k, ()))
                states[k] = st
            store._evict(e, t_end)
        return store
