"""Run the trial: reference LLM agent vs its placebo twin, the momentum rule and buy-and-hold.

    python3 -m placebo.trial [--runs 3] [--draws 2000]

Writes out/report.json and out/report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import agent as A
from . import config as C
from . import engine as E
from .data import DATA

OUT = Path(__file__).resolve().parent.parent / "out"


def build_report(runs=C.AGENT_RUNS, draws=C.PLACEBO_DRAWS, progress=print) -> dict:
    m = E.load_market()
    progress(f"market: {len(m.symbols)} symbols x {len(m.times)} bars")
    positions, invalid = A.run_agent(m, runs=runs, progress=progress)
    rep = E.evaluate(positions, m, draws=draws, invalid=invalid)
    (DATA / "reference_positions.json").write_text(json.dumps({
        "model": C.AGENT_MODEL, "symbols": m.symbols, "times": [int(t) for t in m.times],
        "invalid_outputs": invalid, "runs": [p.astype(int).tolist() for p in positions],
        "sample_decisions": sample_decisions(m)}))
    rep["agent"] = {"model": C.AGENT_MODEL, "model_version": "claude-haiku-4-5-20251001 (Anthropic, released 2025-10)", "runner": "Claude Code headless: default system prompt replaced, tools off, empty working "
                              "directory, local settings only (no CLAUDE.md or memory); Claude Code's "
                              "environment block still states today's date",
                    "segments_per_run": C.AGENT_SEGMENTS,
                    "segment_note": "each run replays the window as consecutive segments in parallel; "
                                    "each segment starts flat",
                    "runs": runs, "decisions": runs * len(m.symbols) * len(m.times),
                    "inputs": "masked asset name, last 12 x 4H returns, 30-bar volatility, "
                              "US exchange closed flag, current position; no ticker, no date"}
    rep["data_snapshot"] = json.loads((DATA / "snapshot.json").read_text())
    rep["generated_at"] = int(time.time() * 1000)
    rep["sample_decisions"] = sample_decisions(m)
    return rep


def sample_decisions(m, n: int = 6) -> list:
    """A few real decisions with the agent's stated reason, spread across the window (run 0)."""
    recs = [json.loads(l) for l in A.CACHE.read_text().splitlines()]
    recs = sorted((r for r in recs if r["model"] == C.AGENT_MODEL and r["run"] == 0), key=lambda r: r["t"])
    pick = [recs[int(i)] for i in np.linspace(0, len(recs) - 1, n)] if recs else []
    out = []
    for r in pick:
        for sym, d in r["decisions"].items():
            if d["valid"] and d["position"] != 0:
                out.append({"t": r["t"], "symbol": sym, "position": d["position"], "reason": d["reason"]})
                break
    return out


def to_markdown(r: dict) -> str:
    rows = "\n".join(
        f"| {k} | {v['sharpe']:.2f} | {v['total_return']*100:.1f}% | {v['max_drawdown']*100:.1f}% | "
        f"{v['win_rate']*100:.0f}% | {v['turnover_per_day']:.2f} | {v['costs_paid']*100:.1f}% |"
        for k, v in r["arms"].items())
    p = r["placebo"]
    cons = "n/a" if r["consistency"] is None else f"{r['consistency']*100:.0f}%"
    return f"""# Placebo trial report

Verdict: **{r['verdict']}**

- Agent Sharpe (worst of {len(r['p_values_per_run'])} runs): {p['agent_sharpe']:.2f}
- Beat {p['percentile']:.0f}% of {p['draws']} placebo twins, p = {p['p_value']:.3f} (one-sided, worst run)
- p per run: {', '.join(f'{x:.3f}' for x in r['p_values_per_run'])}; Sharpe per run: {', '.join(f'{x:.2f}' for x in r['sharpe_per_run'])}
- Consistency across runs: {cons}
- Invalid outputs forced flat: {r['arms']['agent'].get('invalid_outputs', 0)}

| arm | Sharpe | return | max DD | win | turnover/day | costs |
|---|---|---|---|---|---|---|
{rows}

Costs: taker {r['costs']['taker_fee']}, slippage assumption {r['costs']['slippage_assumption']}, funding from Bitget history.
"""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=C.AGENT_RUNS)
    ap.add_argument("--draws", type=int, default=C.PLACEBO_DRAWS)
    a = ap.parse_args(argv)
    r = build_report(a.runs, a.draws)
    OUT.mkdir(exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(r, indent=1))
    (OUT / "report.md").write_text(to_markdown(r))
    (OUT.parent / "public" / "report.json").write_text(json.dumps(r))
    write_public_logs()
    print(to_markdown(r))
    return 0


def write_public_logs():
    """Static files the page reads: every decision, the paper log and the three hours questions."""
    from . import service as S
    pub = OUT.parent / "public"
    recs = [json.loads(l) for l in A.CACHE.read_text().splitlines()]
    dec = [{"run": r["run"], "t": r["t"], "symbol": s, "position": d["position"], "reason": d["reason"]}
           for r in sorted((r for r in recs if r["model"] == C.AGENT_MODEL), key=lambda r: (r["run"], r["t"]))
           for s, d in r["decisions"].items()]
    (pub / "decisions.json").write_text(json.dumps({"model": C.AGENT_MODEL, "count": len(dec), "decisions": dec}))
    paper_path = DATA / "paper.jsonl"
    paper = [json.loads(l) for l in paper_path.read_text().splitlines()] if paper_path.exists() else []
    keep = ("logged_at", "bar", "model", "prices", "positions", "reasons", "pnl_since_last", "source")
    (pub / "paper.json").write_text(json.dumps([{k: p.get(k) for k in keep} for p in paper]))
    all_runs, _ = S.reference_runs()
    m = E.load_market()
    forest = {}
    for hours in S.HOURS:
        res = E.evaluate([S.restrict_hours(r, m, hours) for r in all_runs], m)
        pl = res["placebo"]
        forest[hours] = {"verdict": res["verdict"], "percentile": pl["percentile"], "p_value": pl["p_value"],
                         "agent_sharpe": pl["agent_sharpe"], "p05": pl["placebo_p05"], "p95": pl["placebo_p95"],
                         "mean": pl["placebo_mean"]}
    (pub / "forest.json").write_text(json.dumps(forest))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
