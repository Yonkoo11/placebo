"""Bitget public market data: 4H candles, fee rates and funding history for the trial window.

No API key. Snapshots are written to data/ so a trial reruns offline.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

from . import config as C

DATA = Path(__file__).resolve().parent.parent / "data"


def _get(path: str) -> dict:
    req = urllib.request.Request(C.BITGET + path, headers={"User-Agent": "placebo-trial"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = json.load(r)
            if body.get("code") != "00000":
                raise RuntimeError(f"Bitget error {body.get('code')}: {body.get('msg')}")
            return body
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def fetch_candles(symbol: str) -> list[dict]:
    """All 4H bars whose open time is inside the window, oldest first."""
    start = int(C.WINDOW_START.timestamp() * 1000)
    end = int(C.WINDOW_END.timestamp() * 1000)
    bars: dict[int, dict] = {}
    cursor = end + C.BAR_MS  # endTime is exclusive of later bars
    while cursor > start:
        rows = _get(
            f"/api/v2/mix/market/history-candles?symbol={symbol}&productType={C.PRODUCT_TYPE}"
            f"&granularity={C.GRANULARITY}&limit=200&endTime={cursor}"
        )["data"]
        if not rows:
            break
        for r in rows:
            t = int(r[0])
            if start <= t <= end:
                bars[t] = {"t": t, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                           "close": float(r[4]), "volume": float(r[5])}
        oldest = min(int(r[0]) for r in rows)
        if oldest >= cursor:
            break
        cursor = oldest
    return [bars[t] for t in sorted(bars)]


def fetch_funding(symbol: str) -> list[dict]:
    start = int(C.WINDOW_START.timestamp() * 1000)
    out, page = [], 1
    while page < 30:
        rows = _get(f"/api/v2/mix/market/history-fund-rate?symbol={symbol}&productType={C.PRODUCT_TYPE}"
                    f"&pageSize=100&pageNo={page}")["data"]
        if not rows:
            break
        out += [{"t": int(r["fundingTime"]), "rate": float(r["fundingRate"])} for r in rows]
        if min(int(r["fundingTime"]) for r in rows) < start:
            break
        page += 1
    end = int(C.WINDOW_END.timestamp() * 1000) + C.BAR_MS
    return sorted([f for f in out if start <= f["t"] < end], key=lambda f: f["t"])


def fetch_fees(symbol: str) -> dict:
    c = _get(f"/api/v2/mix/market/contracts?productType={C.PRODUCT_TYPE}&symbol={symbol}")["data"][0]
    return {"taker": float(c["takerFeeRate"]), "maker": float(c["makerFeeRate"])}


def snapshot() -> dict:
    DATA.mkdir(exist_ok=True)
    meta = {"fetched_at": int(time.time() * 1000), "source": C.BITGET, "symbols": {}}
    for s in C.SYMBOLS:
        bars = fetch_candles(s)
        with open(DATA / f"candles_{s}_4H.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "open", "high", "low", "close", "volume"])
            w.writeheader()
            w.writerows(bars)
        funding = fetch_funding(s)
        (DATA / f"funding_{s}.json").write_text(json.dumps(funding))
        meta["symbols"][s] = {"bars": len(bars), "fees": fetch_fees(s), "funding_events": len(funding),
                              "nonzero_funding": sum(1 for x in funding if x["rate"] != 0)}
    (DATA / "snapshot.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_candles(symbol: str) -> list[dict]:
    with open(DATA / f"candles_{symbol}_4H.csv") as f:
        return [{k: (int(v) if k == "t" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def load_funding(symbol: str) -> list[dict]:
    p = DATA / f"funding_{symbol}.json"
    return json.loads(p.read_text()) if p.exists() else []


def main(argv: list[str]) -> int:
    meta = snapshot() if "--refresh" in argv or not (DATA / "snapshot.json").exists() else json.loads(
        (DATA / "snapshot.json").read_text())
    for s, m in meta["symbols"].items():
        print(f"{s} {m['bars']} bars  taker {m['fees']['taker']}  funding events {m['funding_events']}"
              f" (non-zero {m['nonzero_funding']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
