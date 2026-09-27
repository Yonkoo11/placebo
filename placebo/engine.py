"""Backtest, placebo twin, metrics and verdict. Every number on screen comes from here.

Conventions (ai/PRODUCT-BRIEF.md, Engine):
- positions[s][t] is decided at the close of bar t and held over bar t+1.
- bar return r[s][t+1] = close[t+1] / close[t] - 1.
- cost when the position changes = |delta| * (taker fee + slippage).
- funding charged -position * rate for each funding time inside the held bar.
"""
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import data as D


@dataclass
class Market:
    times: np.ndarray              # (T,) bar open ms, common to all symbols
    closes: np.ndarray             # (S, T)
    rets: np.ndarray               # (S, T) r[:, t] = close[t]/close[t-1]-1, r[:, 0] = 0
    funding: np.ndarray            # (S, T) funding rate charged on the bar that starts at t
    symbols: list


def load_market(symbols=None) -> Market:
    symbols = symbols or C.SYMBOLS
    per = {s: {b["t"]: b["close"] for b in D.load_candles(s)} for s in symbols}
    times = np.array(sorted(set.intersection(*(set(v) for v in per.values()))), dtype=np.int64)
    closes = np.array([[per[s][t] for t in times] for s in symbols])
    rets = np.zeros_like(closes)
    rets[:, 1:] = closes[:, 1:] / closes[:, :-1] - 1
    funding = np.zeros_like(closes)
    for i, s in enumerate(symbols):
        for f in D.load_funding(s):
            j = np.searchsorted(times, f["t"], side="right") - 1  # bar containing the funding time
            if 0 <= j < len(times):
                funding[i, j] += f["rate"]
    return Market(times, closes, rets, funding, list(symbols))


def pnl(pos: np.ndarray, m: Market) -> np.ndarray:
    """Portfolio net return per bar. pos is (S, T); result is (T-1,) for bars 1..T-1."""
    held = pos[:, :-1]                         # position held over bar t+1
    gross = held * m.rets[:, 1:]
    prev = np.concatenate([np.zeros((pos.shape[0], 1)), pos[:, :-2]], axis=1)
    cost = np.abs(held - prev) * (C.TAKER_FEE + C.SLIPPAGE)
    fund = -held * m.funding[:, 1:]
    return (gross - cost + fund).mean(axis=0)


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(C.BARS_PER_YEAR)) if sd > 0 else 0.0


def metrics(pos: np.ndarray, m: Market) -> dict:
    r = pnl(pos, m)
    eq = np.cumprod(1 + r)
    dd = 1 - eq / np.maximum.accumulate(eq)
    down = r[r < 0]
    held = pos[:, :-1]
    active = r[np.abs(held).sum(axis=0) > 0]
    prev = np.concatenate([np.zeros((pos.shape[0], 1)), pos[:, :-2]], axis=1)
    changes = np.abs(held - prev)
    days = len(r) / 6
    return {
        "total_return": float(eq[-1] - 1),
        "sharpe": sharpe(r),
        "sortino": float(r.mean() / down.std(ddof=1) * np.sqrt(C.BARS_PER_YEAR)) if len(down) > 1 else 0.0,
        "max_drawdown": float(dd.max()),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "turnover_per_day": float(changes.sum() / pos.shape[0] / days),
        "costs_paid": float((changes * (C.TAKER_FEE + C.SLIPPAGE)).mean(axis=0).sum()),
        "exposure": float(np.abs(held).mean()),
    }


def rule_positions(m: Market) -> np.ndarray:
    k = C.RULE_LOOKBACK
    pos = np.zeros_like(m.rets)
    for t in range(k, m.rets.shape[1]):
        pos[:, t] = np.sign(m.rets[:, t - k + 1:t + 1].sum(axis=1))
    return pos


def hold_positions(m: Market) -> np.ndarray:
    pos = np.ones_like(m.rets)
    return pos


def episodes(row: np.ndarray) -> list[tuple[int, int]]:
    """Maximal stretches [start, end) of the same non-zero position."""
    out, t, n = [], 0, len(row)
    while t < n:
        if row[t] == 0:
            t += 1
            continue
        s = t
        while t < n and row[t] == row[s]:
            t += 1
        out.append((s, t))
    return out


def placebo_sharpes(pos: np.ndarray, m: Market, draws=C.PLACEBO_DRAWS, seed=C.PLACEBO_SEED) -> np.ndarray:
    """Sharpe of `draws` placebo twins: same episodes (timing, size, holding), random sign each."""
    rng = np.random.default_rng(seed)
    eps = [episodes(pos[i]) for i in range(pos.shape[0])]
    # Build an episode-index map so each draw is a vectorised sign flip.
    idx = np.full(pos.shape, -1, dtype=np.int64)
    offs, k = [], 0
    for i, es in enumerate(eps):
        offs.append(k)
        for j, (a, b) in enumerate(es):
            idx[i, a:b] = k + j
        k += len(es)
    base = np.abs(pos)
    out = np.empty(draws)
    for d in range(draws):
        signs = rng.choice([-1.0, 1.0], size=max(k, 1))
        twin = np.where(idx >= 0, base * signs[np.maximum(idx, 0)], 0.0)
        out[d] = sharpe(pnl(twin, m))
    return out


def placebo_test(pos: np.ndarray, m: Market, draws=C.PLACEBO_DRAWS) -> dict:
    agent = sharpe(pnl(pos, m))
    ps = placebo_sharpes(pos, m, draws)
    beat = int((ps >= agent).sum())
    return {
        "agent_sharpe": agent,
        "draws": draws,
        "p_value": (1 + beat) / (draws + 1),
        "percentile": float((ps < agent).mean() * 100),
        "placebo_mean": float(ps.mean()),
        "placebo_p05": float(np.percentile(ps, 5)),
        "placebo_p95": float(np.percentile(ps, 95)),
        "histogram": np.histogram(ps, bins=40)[0].tolist(),
        "bin_edges": np.histogram(ps, bins=40)[1].round(4).tolist(),
    }


def consistency(runs: list[np.ndarray]) -> float | None:
    if len(runs) < 2:
        return None
    stack = np.stack(runs)                    # (R, S, T)
    same = (stack == stack[0]).all(axis=0)
    return float(same.mean())


def verdict(p_worst: float, agent: dict, rule: dict, cons: float | None) -> str:
    if (p_worst <= 0.05 and agent["sharpe"] > rule["sharpe"] and cons is not None and cons >= 0.80
            and agent["max_drawdown"] <= 0.20):
        return C.VERDICT_DEPLOY
    if p_worst <= 0.20 and agent["sharpe"] > 0:
        return C.VERDICT_HALF
    return C.VERDICT_NO


def evaluate(runs: list[np.ndarray], m: Market, draws=C.PLACEBO_DRAWS, invalid: int = 0) -> dict:
    """Full trial for an agent given as one or more runs of (S, T) positions."""
    tests = [placebo_test(p, m, draws) for p in runs]
    worst = max(range(len(tests)), key=lambda i: tests[i]["p_value"])
    arms = {"agent": metrics(runs[worst], m), "rule": metrics(rule_positions(m), m),
            "hold": metrics(hold_positions(m), m)}
    arms["agent"]["invalid_outputs"] = invalid
    cons = consistency(runs)
    return {
        "arms": arms,
        "placebo": tests[worst],
        "p_values_per_run": [t["p_value"] for t in tests],
        "sharpe_per_run": [t["agent_sharpe"] for t in tests],
        "consistency": cons,
        "verdict": verdict(tests[worst]["p_value"], arms["agent"], arms["rule"], cons),
        "window": {"start": int(m.times[0]), "end": int(m.times[-1]), "bars": int(len(m.times))},
        "symbols": m.symbols,
        "costs": {"taker_fee": C.TAKER_FEE, "slippage_assumption": C.SLIPPAGE, "funding": "Bitget history"},
    }
