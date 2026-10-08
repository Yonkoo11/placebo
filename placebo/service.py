"""What the web page calls: question (+ optional CSV) in, trial result out.

The model only reads the question. Every number comes from engine.evaluate.
"""
import csv
import io
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

import numpy as np

from . import config as C
from . import engine as E
from . import share
from .data import DATA
from .share import UserError  # one error class for the page: {reason}

TICKERS = {"NVDA": "NVDAUSDT", "TSLA": "TSLAUSDT", "QQQ": "QQQUSDT"}


def parse_question(q: str) -> dict:
    """{"symbols": [...], "parser": "llm"|"keywords"}. Falls back to keywords when the model is unreachable."""
    found = [TICKERS[k] for k in TICKERS if re.search(rf"\b{k}\b", q.upper())]
    try:
        body = {
            "contents": [{"role": "user", "parts": [{"text": (
                "Read a trader's question about a trading agent. Which markets does it ask about "
                "(NVDA, TSLA, QQQ; none named or 'all' means all three)? Does it restrict trading "
                "hours: only while the US stock exchange is closed (\"us_closed\"), only while it "
                "is open (\"us_open\"), or no restriction (\"all\")? Reply JSON "
                "{\"markets\": [..], \"hours\": \"all\"|\"us_closed\"|\"us_open\"}.\n\nQuestion: " + q[:500])}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{C.PARSER_MODEL}:generateContent",
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
        obj = json.loads(text)
        mk = [TICKERS[x] for x in obj.get("markets", []) if x in TICKERS]
        hours = obj.get("hours") if obj.get("hours") in HOURS else "all"
        return {"symbols": [s for s in C.SYMBOLS if s in mk] or list(C.SYMBOLS), "hours": hours, "parser": "llm"}
    except Exception:
        return {"symbols": [s for s in C.SYMBOLS if s in found] or list(C.SYMBOLS),
                "hours": keyword_hours(q), "parser": "keywords"}


HOURS = ("all", "us_closed", "us_open")


def keyword_hours(q: str) -> str:
    ql = q.lower()
    if re.search(r"(market|exchange|us)\W+(is\W+)?closed|after[- ]hours|overnight|weekend", ql):
        return "us_closed"
    if re.search(r"(market|exchange|us)\W+(is\W+)?open|regular (hours|session)", ql):
        return "us_open"
    return "all"


def restrict_hours(pos: np.ndarray, m: E.Market, hours: str) -> np.ndarray:
    """Zero every position decided at a bar close outside the allowed hours."""
    if hours == "all":
        return pos
    from .agent import us_closed
    closed = np.array([us_closed(int(t) + C.BAR_MS) for t in m.times])
    keep = closed if hours == "us_closed" else ~closed
    return pos * keep[None, :]


def reference_runs() -> list[np.ndarray]:
    ref = json.loads((DATA / "reference_positions.json").read_text())
    return [np.array(run, dtype=float) for run in ref["runs"]], ref


def _parse_time(v: str) -> int:
    v = v.strip()
    if re.fullmatch(r"\d{12,13}", v):
        return int(v)
    d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def parse_csv(text: str, m: E.Market) -> list[np.ndarray]:
    """One run per distinct value of the optional `run` column; no column means one run."""
    if len(text.encode()) > C.MAX_CSV_BYTES:
        raise UserError("the CSV is larger than 1 MB")
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows or not {"time", "symbol", "position"} <= set(k.strip() for k in rows[0]):
        raise UserError("the CSV needs the header time,symbol,position")
    runs: dict[str, np.ndarray] = {}
    index = {int(t): i for i, t in enumerate(m.times)}
    matched = 0
    for n, r in enumerate(rows, start=2):
        r = {k.strip(): (v or "").strip() for k, v in r.items()}
        sym = r["symbol"].upper()
        sym = TICKERS.get(sym, sym)
        if sym not in C.SYMBOLS:
            raise UserError(f"line {n}: unknown symbol {r['symbol']!r} (use NVDA, TSLA or QQQ)")
        try:
            t, p = _parse_time(r["time"]), float(r["position"])
        except ValueError:
            raise UserError(f"line {n}: time or position is not readable")
        if p not in (-1.0, 0.0, 1.0):
            raise UserError(f"line {n}: position must be -1, 0 or 1")
        if sym in m.symbols and t in index:
            key = r.get("run", "") or "1"
            if key not in runs and len(runs) >= C.MAX_RUNS:
                raise UserError(f"line {n}: at most {C.MAX_RUNS} runs (the run column has more values)")
            pos = runs.setdefault(key, np.zeros(m.rets.shape))
            pos[m.symbols.index(sym), index[t]] = p
            matched += 1
    if matched == 0:
        raise UserError("no row matched a 4-hour bar open between 1 Jul and 26 Sep 2026 (UTC)")
    return list(runs.values())


_MOOD = {"at": 0.0, "value": None}


def market_mood() -> dict:
    """Bitget MCP Fear & Greed, checked at most every 5 minutes per server instance."""
    import time
    if _MOOD["value"] is not None and time.time() - _MOOD["at"] < 300:
        return _MOOD["value"]
    _MOOD["value"], _MOOD["at"] = _fetch_mood(), time.time()
    return _MOOD["value"]


def _fetch_mood() -> dict:
    """Returns {"ok": False, "code": ...} when the service is down."""
    url = "https://agent.bitget.com/mcp"
    hdr = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           "User-Agent": "placebo-trial"}
    try:
        def call(body, sid=None):
            h = dict(hdr, **({"Mcp-Session-Id": sid} if sid else {}))
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
            with urllib.request.urlopen(req, timeout=4) as r:
                return r.headers.get("mcp-session-id"), r.read().decode()
        sid, _ = call({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "placebo", "version": "1"}}})
        call({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
        _, text = call({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "do_query", "arguments": {"entry_id": "sentiment_market_fear_greed", "params": {"limit": 1}}}}, sid)
        data = json.loads(text.split("data: ", 1)[1].splitlines()[0])["result"]["structuredContent"]
        if not data.get("success"):
            return {"ok": False, "code": data.get("status_code")}
        return {"ok": True, "data": data.get("data")}
    except Exception as e:
        return {"ok": False, "code": getattr(e, "code", type(e).__name__)}


def _trial(question: str, hours: str, symbols: list[str], m: E.Market, uploaded: list[np.ndarray] | None,
           parser: str) -> dict:
    """The one trial path: question (reference agent) or uploaded runs, hours filter, evaluate, link token."""
    if uploaded is not None:
        runs, invalid, who, samples = uploaded, 0, "uploaded", []
    else:
        all_runs, ref = reference_runs()
        keep = [C.SYMBOLS.index(s) for s in symbols]
        runs, invalid, who = [r[keep] for r in all_runs], ref["invalid_outputs"], "reference"
        samples = [d for d in ref.get("sample_decisions", []) if d["symbol"] in symbols]
    token = share.encode(question, hours, symbols, uploaded)
    # bars on which at least one run holds a position in any market, before the hours filter
    bars_matched = int((np.abs(np.stack(runs)).sum(axis=(0, 1)) > 0).sum())
    runs = [restrict_hours(r, m, hours) for r in runs]
    res = E.evaluate(runs, m, invalid=invalid)
    res.update({"agent_source": who, "parser": parser, "hours": hours, "mood": market_mood(),
                "sample_decisions": samples, "token": token, "question": question[:500], "bars_matched": bars_matched})
    return res


def run(question: str, csv_text: str | None) -> dict:
    q = parse_question(question or "")
    m = E.load_market(q["symbols"])
    uploaded = parse_csv(csv_text, m) if csv_text else None
    return _trial(question or "", q["hours"], q["symbols"], m, uploaded, q["parser"])


def run_token(token: str) -> dict:
    """Recompute a trial from a shared link. Same path as run(); no model call (the link holds the parsed input)."""
    d = share.decode(token)
    m = E.load_market(d["symbols"])
    uploaded = share.runs_from(d, m.rets.shape) if d["packed"] is not None else None
    res = _trial(d["question"], d["hours"], d["symbols"], m, uploaded, "link")
    res["from_link"] = True
    return res
