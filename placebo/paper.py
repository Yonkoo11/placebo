"""Forward paper trading: the reference agent decides on the latest closed 4H bar at live Bitget
prices, and each fill is appended to data/paper.jsonl with source=local-paper.

No orders are sent anywhere. When a Bitget Demo API key exists this becomes a demo order; until
then every row says local-paper so nobody mistakes it for an exchange fill.

    python3 -m placebo.paper          # one step (run every 4 hours)
"""
import json
import sys
import time

import numpy as np

from . import agent as A
from . import config as C
from . import engine as E
from .data import DATA, _get

LOG = DATA / "paper.jsonl"


def live_market() -> tuple[E.Market, dict]:
    closes, times, prices = [], None, {}
    for s in C.SYMBOLS:
        rows = _get(f"/api/v2/mix/market/candles?symbol={s}&productType={C.PRODUCT_TYPE}"
                    f"&granularity={C.GRANULARITY}&limit=60")["data"]
        rows = sorted(rows, key=lambda r: int(r[0]))
        now = int(time.time() * 1000)
        rows = [r for r in rows if int(r[0]) + C.BAR_MS <= now]      # closed bars only
        t = [int(r[0]) for r in rows]
        times = t if times is None else [x for x in times if x in set(t)]
        closes.append({int(r[0]): float(r[4]) for r in rows})
        tick = _get(f"/api/v2/mix/market/ticker?symbol={s}&productType={C.PRODUCT_TYPE}")["data"][0]
        prices[s] = float(tick["lastPr"])
    c = np.array([[cl[t] for t in times] for cl in closes])
    r = np.zeros_like(c)
    r[:, 1:] = c[:, 1:] / c[:, :-1] - 1
    return E.Market(np.array(times, dtype=np.int64), c, r, np.zeros_like(c), list(C.SYMBOLS)), prices


def last_state() -> dict:
    if not LOG.exists():
        return {}
    lines = LOG.read_text().splitlines()
    return json.loads(lines[-1]) if lines else {}


def step() -> dict:
    m, prices = live_market()
    t = len(m.times) - 1
    prev = last_state()
    if prev.get("bar") == int(m.times[t]):
        return {"skipped": "bar already decided", "bar": int(m.times[t])}
    cur = {s: prev.get("positions", {}).get(s, 0) for s in C.SYMBOLS}
    inputs = [A.features(m, i, t, cur[s]) for i, s in enumerate(C.SYMBOLS)]
    assets = [C.ALIASES[s] for s in C.SYMBOLS]
    dec, raw = A.decide(json.dumps(inputs), assets, C.AGENT_MODEL)
    pnl = {}
    for s in C.SYMBOLS:  # mark the previous holding to the live price, costs on change
        if prev.get("prices"):
            ret = prices[s] / prev["prices"][s] - 1
            new = dec[C.ALIASES[s]][0]
            pnl[s] = round(cur[s] * ret - abs(new - cur[s]) * (C.TAKER_FEE + C.SLIPPAGE), 6)
    row = {
        "source": "local-paper", "logged_at": int(time.time() * 1000), "bar": int(m.times[t]),
        "model": C.AGENT_MODEL, "prices": prices,
        "positions": {s: dec[C.ALIASES[s]][0] for s in C.SYMBOLS},
        "reasons": {s: dec[C.ALIASES[s]][1] for s in C.SYMBOLS},
        "valid": {s: dec[C.ALIASES[s]][2] for s in C.SYMBOLS},
        "pnl_since_last": pnl, "inputs": inputs,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    r = step()
    print(json.dumps({k: r[k] for k in r if k != "inputs"}))
    sys.exit(0)
