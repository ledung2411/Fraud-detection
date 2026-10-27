"""Synthetic transactions with the IEEE-CIS schema.

The real IEEE-CIS data cannot be redistributed (Kaggle competition rules), so this
generator exists to make the whole pipeline runnable in CI and in a fresh clone.
It writes ``train_transaction.csv`` and ``train_identity.csv`` with the same column
names and conventions as the Kaggle files (TransactionDT in seconds starting at 86400,
card1..card6, addr1, D1 = days since the client relationship started, C/D/M/V/id_
columns), so the loader, features, model and API are exercised exactly as on real data.

Fraud patterns that are injected:

* **BIN attack / card testing** - several stolen cards from one ``card1`` make small
  test payments within a few hours, then a few large ones.
* **Account takeover** - an existing client suddenly makes a burst of payments several
  times larger than usual, often at night, sometimes from a new device.
* **New-account fraud** - recently opened accounts (D1 close to 0, sometimes risky
  e-mail domains) cash out over one to three days, often in round amounts.
* **Opportunistic fraud** - a small share of ordinary-looking transactions.

Legitimate look-alikes keep any single signal from being decisive: new clients are
busy in their first days, people go on shopping sprees and buy big-ticket items, and
red flags (device, foreign-currency cents, e-mail mismatch...) appear on only part of
the fraud.

Labels are noisy on purpose, as in the real data (labels come from chargebacks):
``unreported_rate`` of the fraud episodes are never reported and keep ``isFraud = 0``.

Drift is injected on purpose (amount inflation, more card testing over time, and in
the last month a new e-mail provider and a browser update) so that the drift report
has something to find.
This data is NOT a substitute for the real dataset: numbers obtained on it only prove
that the pipeline works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMAIL_DOMAINS = np.array(
    [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "anonymous.com",
        "aol.com",
        "comcast.net",
        "icloud.com",
        "outlook.com",
        "msn.com",
        "att.net",
        "live.com",
        "sbcglobal.net",
        "verizon.net",
        "ymail.com",
        "bellsouth.net",
        "mail.com",
        "protonmail.com",
        "outlook.es",
    ]
)
EMAIL_P = np.array([40, 17, 8, 6, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0.6, 0.3, 0.5], dtype=float)
RISKY_DOMAINS = np.array(["protonmail.com", "mail.com", "outlook.es", "anonymous.com", "gmail.com"])
PRODUCTS = np.array(["W", "C", "R", "H", "S"])
PRODUCT_P = np.array([0.74, 0.11, 0.06, 0.06, 0.03])
DEVICES = np.array(
    [
        "Windows",
        "iOS Device",
        "MacOS",
        "Trident/7.0",
        "SM-G960U",
        "SM-G930V",
        "Moto G (4)",
        "rv:11.0",
        "SAMSUNG SM-G892A",
        "Linux",
    ]
)
DEVICE_P = np.array([30, 25, 15, 8, 6, 5, 4, 3, 2, 2], dtype=float)
FRAUD_DEVICES = np.array(
    ["Windows", "Linux", "rv:57.0", "Moto E (4) Build/NCQ26.69-56", "iOS Device"]
)
BROWSERS = np.array(
    [
        "chrome 63.0",
        "mobile safari 11.0",
        "ie 11.0 for desktop",
        "safari generic",
        "chrome 64.0",
        "firefox 57.0",
        "edge 16.0",
    ]
)


def _norm(p: np.ndarray) -> np.ndarray:
    return p / p.sum()


def _daytime_seconds(rng: np.random.Generator, n: int) -> np.ndarray:
    """Seconds within a day, concentrated in waking hours."""
    hours = rng.normal(14.5, 4.5, n) % 24
    night = rng.random(n) < 0.08
    hours[night] = rng.uniform(0, 24, night.sum())
    return (hours * 3600).astype(np.int64)


class _Builder:
    """Accumulates generated transactions as dicts of column arrays."""

    def __init__(self) -> None:
        self.parts: list[pd.DataFrame] = []

    def add(self, **cols) -> None:
        n = max(len(np.atleast_1d(v)) for v in cols.values())
        self.parts.append(
            pd.DataFrame({k: (np.repeat(v, n) if np.ndim(v) == 0 else v) for k, v in cols.items()})
        )

    def frame(self) -> pd.DataFrame:
        return pd.concat(self.parts, ignore_index=True)


def generate(
    n_clients: int = 20000,
    n_days: int = 182,
    seed: int = 7,
    n_bin_attacks: int = 90,
    n_ato: int = 450,
    n_new_account_fraud: int = 700,
    opportunistic_rate: float = 0.006,
    unreported_rate: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(transactions, identity)`` frames in IEEE-CIS format."""
    rng = np.random.default_rng(seed)
    b = _Builder()

    def reported() -> bool:
        """Labels come from chargebacks; some fraud is never reported and stays 0."""
        return bool(rng.random() >= unreported_rate)

    next_client = 0

    # ------------------------------------------------------------------ clients
    bins = np.unique(rng.integers(1000, 18400, 2500))
    bin_p = _norm(1.0 / np.arange(1, len(bins) + 1) ** 0.9)
    zips = np.arange(100, 541)
    zip_p = _norm(1.0 / np.arange(1, len(zips) + 1) ** 0.8)
    rng.shuffle(zip_p)

    def new_clients(n: int, start_day: np.ndarray, email: np.ndarray | None = None) -> pd.DataFrame:
        nonlocal next_client
        ids = np.arange(next_client, next_client + n)
        next_client += n
        card1 = rng.choice(bins, n, p=bin_p)
        if email is None:
            email = rng.choice(EMAIL_DOMAINS, n, p=_norm(EMAIL_P)).astype(object)
            email[rng.random(n) < 0.15] = None
        addr1 = rng.choice(zips, n, p=zip_p).astype(float)
        addr1[rng.random(n) < 0.1] = np.nan
        return pd.DataFrame(
            {
                "client": ids,
                "start_day": start_day,
                "card1": card1,
                "card2": (card1 * 7919 % 500 + 100).astype(float),
                "card3": np.where(
                    rng.random(n) < 0.88, 150.0, rng.choice([185.0, 144.0, 146.0], n)
                ),
                "card4": rng.choice(
                    ["visa", "mastercard", "american express", "discover"],
                    n,
                    p=[0.65, 0.32, 0.015, 0.015],
                ),
                "card5": rng.choice(
                    [226.0, 224.0, 166.0, 102.0, 117.0, 138.0],
                    n,
                    p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05],
                ),
                "card6": rng.choice(["debit", "credit"], n, p=[0.75, 0.25]),
                "addr1": addr1,
                "P_emaildomain": email,
                "product": rng.choice(PRODUCTS, n, p=PRODUCT_P),
                "rate": rng.lognormal(np.log(0.05), 0.9, n),
                "amt_mu": rng.normal(np.log(55), 0.6, n),
                "device": rng.choice(DEVICES, n, p=_norm(DEVICE_P)),
                "n_addr": rng.choice([1, 1, 1, 2, 2, 3], n),
                "dist_home": rng.lognormal(np.log(6), 1.0, n),
                "issue_day": start_day - rng.integers(0, 400, n),
            }
        )

    base = new_clients(n_clients, rng.integers(-700, n_days - 5, n_clients))

    def emit(
        clients: pd.DataFrame,
        day: np.ndarray,
        sec: np.ndarray,
        amt: np.ndarray,
        product: np.ndarray,
        fraud: np.ndarray,
        scenario: str,
        risky: bool = False,
    ) -> None:
        """Expand per-transaction client attributes and append to the builder."""
        n = len(day)
        c = clients.reset_index(drop=True)
        device = c["device"].to_numpy().copy()
        if risky:  # fraudsters often, not always, use a different device
            swap = rng.random(n) < 0.3
            device[swap] = rng.choice(FRAUD_DEVICES, swap.sum())
        b.add(
            client=c["client"].to_numpy(),
            day=day,
            sec=sec,
            TransactionAmt=amt,
            ProductCD=product,
            card1=c["card1"].to_numpy(),
            card2=c["card2"].to_numpy(),
            card3=c["card3"].to_numpy(),
            card4=c["card4"].to_numpy(),
            card5=c["card5"].to_numpy(),
            card6=c["card6"].to_numpy(),
            addr1=c["addr1"].to_numpy(),
            P_emaildomain=c["P_emaildomain"].to_numpy(),
            start_day=c["start_day"].to_numpy(),
            issue_day=c["issue_day"].to_numpy(),
            n_addr=c["n_addr"].to_numpy(),
            dist_home=c["dist_home"].to_numpy(),
            device=device,
            isFraud=fraud.astype(np.int8),
            scenario=scenario,
            risky=risky,
        )

    # ------------------------------------------------------- normal behaviour
    first_day = np.maximum(base["start_day"].to_numpy(), 0)
    active = n_days - first_day
    counts = rng.poisson(base["rate"].to_numpy() * active)
    idx = np.repeat(np.arange(n_clients), counts)
    day = first_day[idx] + (rng.random(len(idx)) * active[idx]).astype(np.int64)
    sec = _daytime_seconds(rng, len(idx))

    # Legitimate behaviour that *looks* like fraud, so no single signal is decisive:
    # new clients are busy in their first days, and people go on shopping sprees.
    onboard = np.flatnonzero(base["start_day"].to_numpy() >= 0)
    k = rng.poisson(1.5, len(onboard))
    o_idx = np.repeat(onboard, k)
    o_day = np.minimum(
        base["start_day"].to_numpy()[o_idx] + rng.integers(0, 3, len(o_idx)), n_days - 1
    )
    spree = rng.choice(n_clients, int(0.06 * n_clients))
    k = rng.integers(3, 7, len(spree))
    s_idx = np.repeat(spree, k)
    s_day = np.repeat(rng.integers(first_day[spree], n_days), k)
    s_sec = np.repeat(rng.integers(8 * 3600, 20 * 3600, len(spree)), k) + rng.integers(
        0, 3 * 3600, len(s_idx)
    )
    idx = np.concatenate([idx, o_idx, s_idx])
    day = np.concatenate([day, o_day, s_day])
    sec = np.concatenate([sec, _daytime_seconds(rng, len(o_idx)), s_sec])

    cl = base.iloc[idx]
    inflation = 1 + 0.15 * day / n_days  # prices drift up over time
    amt = np.exp(rng.normal(cl["amt_mu"].to_numpy(), 0.7)) * inflation
    big_ticket = rng.random(len(idx)) < 0.015
    amt = np.where(big_ticket, amt * rng.uniform(3, 8, len(idx)), amt)
    product = np.where(
        rng.random(len(idx)) < 0.8,
        cl["product"].to_numpy(),
        rng.choice(PRODUCTS, len(idx), p=PRODUCT_P),
    )
    amt = np.where(product == "C", amt * 0.6, amt)
    amt = np.where(rng.random(len(idx)) < 0.1, np.ceil(amt), amt)  # subscriptions, round prices
    fraud = rng.random(len(idx)) < opportunistic_rate  # opportunistic fraud, weak signal
    amt = np.where(fraud, amt * rng.uniform(1.5, 4, len(idx)), amt)
    emit(cl, day, sec, amt, product, fraud, "normal")

    # ------------------------------------------------- BIN attack / card testing
    # Attacks get more frequent over time (concept drift in the fraud mix).
    attack_days = rng.choice(n_days, n_bin_attacks, p=_norm(np.linspace(0.5, 2.0, n_days)))
    for d in attack_days:
        n_cards = rng.integers(2, 8)
        t0 = rng.integers(0, 86400 - 4 * 3600)
        bin_ = rng.choice(bins[:400])
        # Half the stolen cards are new to the merchant network, half have a history.
        start = np.where(rng.random(n_cards) < 0.5, d, d - rng.integers(30, 700, n_cards))
        stolen = new_clients(n_cards, start)
        stolen["card1"] = bin_
        stolen["card2"] = float(bin_ * 7919 % 500 + 100)
        for i in range(n_cards):
            n_test = rng.integers(2, 6)
            n_big = rng.integers(0, 3)
            offs = np.sort(rng.integers(0, 4 * 3600, n_test + n_big))
            amts = np.concatenate([rng.uniform(0.5, 15, n_test), rng.uniform(150, 1500, n_big)])
            k = n_test + n_big
            emit(
                stolen.iloc[[i] * k],
                np.full(k, d),
                t0 + offs,
                amts,
                rng.choice(["C", "W"], k, p=[0.35, 0.65]),
                np.full(k, reported()),
                "bin_attack",
                risky=True,
            )

    # ------------------------------------------------------- account takeover
    victims = base[(base["start_day"] < n_days - 40)].sample(n_ato, random_state=seed)
    for _, v in victims.iterrows():
        d = rng.integers(max(v["start_day"], 0) + 20, n_days)
        k = rng.integers(2, 8)
        # Takeovers happen disproportionately at night, while the owner sleeps.
        t0 = rng.integers(0, 5 * 3600) if rng.random() < 0.6 else rng.integers(0, 86400 - 6 * 3600)
        secs = t0 + np.sort(rng.integers(0, 6 * 3600, k))
        # Large for *this* client, but often within the range of big spenders overall.
        amts = np.exp(v["amt_mu"]) * rng.uniform(2.5, 8, k)
        product = np.where(rng.random(k) < 0.5, v["product"], rng.choice(["C", "W", "H"], k))
        emit(
            pd.DataFrame([v] * k),
            np.full(k, d),
            secs,
            amts,
            product,
            np.full(k, reported()),
            "ato",
            risky=True,
        )

    # --------------------------------------------------------- new-account fraud
    ring_zips = rng.choice(zips, 15)
    ring_days = rng.choice(n_days, n_new_account_fraud, p=_norm(np.linspace(1.0, 1.5, n_days)))
    email = rng.choice(EMAIL_DOMAINS, n_new_account_fraud, p=_norm(EMAIL_P)).astype(object)
    risky_email = rng.random(n_new_account_fraud) < 0.3
    email[risky_email] = rng.choice(RISKY_DOMAINS, risky_email.sum())
    email[rng.random(n_new_account_fraud) < 0.2] = None
    ring = new_clients(n_new_account_fraud, ring_days, email=email)
    ring["addr1"] = rng.choice(ring_zips, len(ring)).astype(float)
    ring["n_addr"] = rng.integers(2, 8, len(ring))
    ring["dist_home"] = rng.lognormal(np.log(40), 1.2, len(ring))
    k = rng.integers(2, 10, len(ring))
    idx = np.repeat(np.arange(len(ring)), k)
    day = np.minimum(ring["start_day"].to_numpy()[idx] + rng.integers(0, 3, len(idx)), n_days - 1)
    emit(
        ring.iloc[idx],
        day,
        _daytime_seconds(rng, len(idx)),
        # Cash-out amounts are often round numbers.
        np.where(
            rng.random(len(idx)) < 0.5,
            50 * rng.integers(1, 8, len(idx)),
            rng.lognormal(np.log(90), 0.9, len(idx)),
        ),
        rng.choice(PRODUCTS, len(idx), p=[0.5, 0.3, 0.05, 0.05, 0.1]),
        (rng.random(len(ring)) >= unreported_rate)[idx],
        "new_account",
        risky=True,
    )

    df = b.frame()
    return _finalise(df, rng, n_days)


def _finalise(
    df: pd.DataFrame, rng: np.random.Generator, n_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    df["TransactionDT"] = 86400 * (1 + df["day"].to_numpy()) + df["sec"].to_numpy()
    df = df.sort_values(["TransactionDT", "client"], kind="mergesort").reset_index(drop=True)
    df["TransactionID"] = 2987000 + np.arange(n)
    df["TransactionAmt"] = df["TransactionAmt"].round(2).clip(lower=0.25)
    fraud = df["isFraud"].to_numpy() == 1
    # Only part of the fraud carries the static red flags below; the rest has to be
    # caught from behaviour (velocity, deviation from the client's own history).
    risky = df["risky"].to_numpy() & (rng.random(n) < 0.5)
    df["risky"] = risky
    tday = df["TransactionDT"].to_numpy() // 86400

    # Foreign-currency purchases have non-round cents (a known IEEE-CIS signal).
    fx = rng.random(n) < np.where(risky, 0.25, 0.08)
    df.loc[fx, "TransactionAmt"] = (df.loc[fx, "TransactionAmt"] * 1.137).round(3)

    df["addr2"] = np.where(df["addr1"].isna(), np.nan, np.where(rng.random(n) < 0.98, 87.0, 60.0))
    df["dist1"] = np.where(
        rng.random(n) < 0.55, np.nan, (df["dist_home"] * rng.lognormal(0, 0.5, n)).round()
    )
    same = rng.random(n) < np.where(risky, 0.45, 0.75)
    r_email = rng.choice(EMAIL_DOMAINS, n, p=_norm(EMAIL_P)).astype(object)
    df["R_emaildomain"] = np.where(same, df["P_emaildomain"], r_email)
    df.loc[rng.random(n) < 0.55, "R_emaildomain"] = None
    # A new e-mail provider takes off during the last month (data drift).
    late = (tday > n_days - 25) & (rng.random(n) < 0.15) & ~fraud
    df.loc[late, "P_emaildomain"] = "newmail.com"

    # Client-level history (what the bank's own D/C counters would show).
    g = df.groupby("client", sort=False)
    prev_dt = g["TransactionDT"].shift(1)
    df["D1"] = (tday - (1 + df["start_day"].to_numpy())).astype(float)
    df["D2"] = np.where(rng.random(n) < 0.45, np.nan, df["D1"])
    df["D3"] = ((df["TransactionDT"] - prev_dt) // 86400).astype(float)
    df["D4"] = np.where(rng.random(n) < 0.3, np.nan, (df["D1"] * rng.uniform(0.5, 1.0, n)).round())
    df["D10"] = np.where(
        rng.random(n) < 0.15, np.nan, (df["D1"] * rng.uniform(0.0, 1.0, n)).round()
    )
    df["D11"] = np.where(rng.random(n) < 0.5, np.nan, df["D1"])
    df["D15"] = (tday - df["issue_day"].to_numpy()).astype(float)  # grows with time: drifts

    cum = g.cumcount().to_numpy()
    df["C1"] = df["n_addr"] + rng.poisson(0.3, n)
    df["C2"] = df["C1"] + rng.poisson(np.where(risky, 0.8, 0.3))
    df["C5"] = np.where(rng.random(n) < 0.7, 0, rng.poisson(1, n))
    df["C6"] = df["C1"] - rng.integers(0, 2, n).clip(0)
    df["C9"] = np.where(df["ProductCD"] == "W", 1 + rng.poisson(0.5, n), 0)
    df["C11"] = df["C1"]
    df["C13"] = (cum * rng.uniform(0.8, 1.2, n)).round() + 1
    df["C14"] = np.where(rng.random(n) < 0.85, 1, rng.integers(0, 4, n))
    for c in ["C3", "C4", "C7", "C8", "C10", "C12"]:
        df[c] = rng.poisson(np.where(df["ProductCD"] == "C", 1.0, 0.1))

    def flags(p_true: np.ndarray, p_nan: float) -> np.ndarray:
        out = np.where(rng.random(n) < p_true, "T", "F").astype(object)
        out[rng.random(n) < p_nan] = None
        return out

    is_w = df["ProductCD"].to_numpy() == "W"
    for m in ["M1", "M2", "M3"]:
        df[m] = np.where(is_w, flags(np.where(risky, 0.7, 0.95), 0.05), None)
    m4 = rng.choice(["M0", "M1", "M2"], n, p=[0.6, 0.15, 0.25]).astype(object)
    m4[risky & (rng.random(n) < 0.5)] = "M2"
    m4[rng.random(n) < 0.4] = None
    df["M4"] = m4
    for m, p in [("M5", 0.45), ("M6", 0.5), ("M7", 0.15), ("M8", 0.35), ("M9", 0.85)]:
        df[m] = flags(np.where(risky, p * 0.7, p), 0.4)

    v_nan = ~is_w & (rng.random(n) < 0.8)
    for i in range(1, 21):
        v = rng.normal(0, 1, n)
        if i in (3, 7, 12):
            v += np.where(fraud, 0.25, 0)  # weakly informative Vesta features
        if i in (5, 15):
            v = np.abs(v) * df["TransactionAmt"].to_numpy() / 100
        df[f"V{i}"] = np.where(v_nan, np.nan, v.round(4))

    # ---------------------------------------------------------- identity table
    has_id = (~is_w & (rng.random(n) < 0.9)) | (is_w & (rng.random(n) < 0.05))
    ids = df.loc[has_id, ["TransactionID", "device", "risky"]].reset_index(drop=True)
    id_day = tday[has_id]
    m = len(ids)
    r = ids["risky"].to_numpy()
    mobile = ids["device"].str.contains("iOS|SM-|Moto|SAMSUNG").to_numpy()
    identity = pd.DataFrame(
        {
            "TransactionID": ids["TransactionID"],
            "id_01": -5.0 * rng.integers(0, 10, m) * (rng.random(m) < 0.6),
            "id_02": rng.lognormal(np.log(np.where(r, 60000, 120000)), 0.8, m).round(),
            "id_05": rng.poisson(1, m).astype(float),
            "id_06": -rng.poisson(np.where(r, 3, 1), m).astype(float),
            "id_12": np.where(rng.random(m) < np.where(r, 0.4, 0.75), "NotFound", "Found"),
            "id_15": np.where(rng.random(m) < np.where(r, 0.6, 0.15), "New", "Found"),
            "id_19": rng.integers(100, 671, m).astype(float),
            "id_20": rng.integers(100, 661, m).astype(float),
            # Browsers auto-update: a new Chrome version replaces the old one late on.
            "id_31": np.where(
                (id_day > n_days - 20) & (rng.random(m) < 0.8),
                np.char.replace(rng.choice(BROWSERS, m).astype(str), "chrome 6", "chrome 7"),
                rng.choice(BROWSERS, m),
            ),
            "id_38": np.where(rng.random(m) < 0.55, "T", "F"),
            "DeviceType": np.where(mobile, "mobile", "desktop"),
            "DeviceInfo": ids["device"],
        }
    )

    keep = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "P_emaildomain",
        "R_emaildomain",
    ]
    keep += [f"C{i}" for i in range(1, 15)]
    keep += ["D1", "D2", "D3", "D4", "D10", "D11", "D15"]
    keep += [f"M{i}" for i in range(1, 10)] + [f"V{i}" for i in range(1, 21)]
    return df[keep], identity


def write(out_dir: str | Path, **kwargs) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tx, identity = generate(**kwargs)
    tx.to_csv(out / "train_transaction.csv", index=False)
    identity.to_csv(out / "train_identity.csv", index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-clients", type=int, default=20000)
    p.add_argument("--n-days", type=int, default=182)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = write(a.out, n_clients=a.n_clients, n_days=a.n_days, seed=a.seed)
    print(f"wrote synthetic IEEE-CIS-format data to {out}/")


if __name__ == "__main__":
    main()
