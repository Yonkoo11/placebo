"""The reference LLM agent: one decision per bar per market, from masked inputs only.

Inputs never include a ticker or a date. Every call is cached in data/agent_runs.jsonl keyed by
(model, run, symbol, bar time), so a trial reruns offline and an interrupted run resumes.
"""
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .data import DATA

CACHE = DATA / "agent_runs.jsonl"

SYSTEM = (
    "You are an autonomous trading agent managing three positions, one in each of three perpetual "
    "contracts that track US-listed stocks or ETFs. The contracts trade 24/7, including when the US "
    "stock exchange is closed. Each decision sets every position for the next 4 hours: 1 = long, "
    "0 = flat, -1 = short. Every change of position costs 0.08% of notional. Decide from the data "
    "given. Reply with JSON only, one entry per asset: "
    "{\"Asset A\": {\"position\": -1|0|1, \"reason\": \"at most 20 words\"}, \"Asset B\": {...}, \"Asset C\": {...}}."
)


def us_closed(ms: int) -> bool:
    d = datetime.fromtimestamp(ms / 1000, timezone.utc)
    minutes = d.hour * 60 + d.minute
    return d.weekday() >= 5 or not (13 * 60 + 30 <= minutes < 20 * 60)


def features(m, s: int, t: int, current: int = 0) -> dict:
    """What the agent sees at the close of bar t. Uses bars <= t only."""
    lo = max(1, t - C.LOOKBACK_RETURNS + 1)
    recent = [round(float(x) * 100, 3) for x in m.rets[s, lo:t + 1]]
    vol_lo = max(1, t - C.VOL_WINDOW + 1)
    window = m.rets[s, vol_lo:t + 1]
    return {
        "asset": C.ALIASES.get(m.symbols[s], f"Asset {s}"),
        "last_4h_returns_pct_oldest_first": recent,
        "volatility_30_bars_pct": round(float(window.std(ddof=1)) * 100, 3) if len(window) > 1 else None,
        "us_stock_exchange_closed_now": us_closed(int(m.times[t]) + C.BAR_MS),
        "current_position": current,
    }


class _Pace:
    """Shared request pacing: the free Gemini tier allows 15 requests per minute per model."""

    def __init__(self, per_minute: float):
        self.gap, self.next, self.lock = 60.0 / per_minute, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.gap
        time.sleep(max(0.0, slot - now))


PACE = _Pace(float(os.environ.get("PLACEBO_RPM", C.LLM_RPM)))


def _claude_cli(prompt: str, model: str) -> str:
    """Claude through the local Claude Code login (used when no API credit is available).

    The default Claude Code system prompt is replaced by SYSTEM, tools are disabled and hooks are
    off, so the model sees exactly what an API call would send.
    """
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    out = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--output-format", "json",
         "--disable-slash-commands", "--no-session-persistence",
         "--disallowedTools", "*", "--settings", '{"disableAllHooks":true}',
         "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'],
        input=prompt, capture_output=True, text=True, timeout=120, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"claude exit {out.returncode}")
    d = json.loads(out.stdout)
    if d.get("is_error"):
        raise RuntimeError("claude error")
    text = d.get("result", "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text


def call_model(prompt: str, model: str) -> str:
    return _claude_cli(prompt, model) if model.startswith("claude") else _gemini(prompt, model)


def _gemini(prompt: str, model: str) -> str:
    PACE.wait()
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT_S) as r:
        return json.load(r)["candidates"][0]["content"]["parts"][0]["text"]


def parse(text: str, assets: list[str]) -> dict:
    """{asset: (position, reason, valid)}. Anything not exactly -1, 0 or 1 is invalid and forced flat."""
    try:
        obj = json.loads(text)
    except ValueError:
        obj = {}
    out = {}
    for a in assets:
        e = obj.get(a) if isinstance(obj, dict) else None
        p = e.get("position") if isinstance(e, dict) else None
        if isinstance(p, bool) or p not in (-1, 0, 1):
            out[a] = (0, "", False)
        else:
            out[a] = (int(p), str(e.get("reason", ""))[:200], True)
    return out


class InfraError(Exception):
    """The model could not be reached. Not the agent's fault: never cached, never counted."""


def decide(prompt: str, assets: list[str], model: str, attempts: int = 8) -> tuple[dict, str]:
    last = ""
    for attempt in range(attempts):
        try:
            text = call_model(prompt, model)
            return parse(text, assets), text[:600]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            time.sleep(min(60, 2 ** (attempt + 1)) if e.code in (429, 500, 503) else 1 + attempt)
        except Exception as e:  # timeouts, connection resets
            last = type(e).__name__
            time.sleep(1 + attempt)
    raise InfraError(last)


def _load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            out[(r["model"], r["run"], r["t"])] = r
    return out


def run_agent(m, runs: int = C.AGENT_RUNS, model: str = C.AGENT_MODEL, progress=print):
    """Positions for each run as (S, T) arrays, plus the count of invalid outputs.

    One call per bar per run decides all markets. Bars are sequential within a run because the
    agent sees its current positions; runs go in parallel.
    """
    cache = _load_cache()
    lock = threading.Lock()
    S, T = m.rets.shape
    assets = [C.ALIASES.get(sym, f"Asset {i}") for i, sym in enumerate(m.symbols)]
    positions = [np.zeros((S, T)) for _ in range(runs)]
    invalid = [0]
    done = [0]

    def lane(run: int):
        cur = [0] * S
        with open(CACHE, "a") as fh:
            for t in range(T):
                key = (model, run, int(m.times[t]))
                if key in cache:
                    rec = cache[key]
                else:
                    inputs = [features(m, s, t, cur[s]) for s in range(S)]
                    dec, raw = decide(json.dumps(inputs), assets, model)
                    rec = {"model": model, "run": run, "t": int(m.times[t]), "raw": raw,
                           "decisions": {m.symbols[s]: {"position": dec[assets[s]][0],
                                                        "reason": dec[assets[s]][1],
                                                        "valid": dec[assets[s]][2]} for s in range(S)},
                           "inputs": inputs}
                    with lock:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                for s in range(S):
                    d = rec["decisions"][m.symbols[s]]
                    positions[run][s, t] = d["position"]
                    cur[s] = d["position"]
                    with lock:
                        invalid[0] += 0 if d["valid"] else 1
                with lock:
                    done[0] += 1
                    if done[0] % 100 == 0:
                        progress(f"  {done[0]}/{runs * T} bar decisions")

    # Runs go in parallel under the shared pace; each run is sequential because the agent sees
    # its own current position.
    with ThreadPoolExecutor(max_workers=runs) as ex:
        for f in [ex.submit(lane, r) for r in range(runs)]:
            f.result()
    return positions, invalid[0]
