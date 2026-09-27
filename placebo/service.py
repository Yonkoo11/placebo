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
from .data import DATA

TICKERS = {"NVDA": "NVDAUSDT", "TSLA": "TSLAUSDT", "QQQ": "QQQUSDT"}


class UserError(ValueError):
    """Shown to the user as {reason}."""


def parse_question(q: str) -> dict:
    """{"symbols": [...], "parser": "llm"|"keywords"}. Falls back to keywords when the model is unreachable."""
    found = [TICKERS[k] for k in TICKERS if re.search(rf"\b{k}\b", q.upper())]
    try:
        body = {
            "contents": [{"role": "user", "parts": [{"text": (
                "Which of these markets does the question ask about: NVDA, TSLA, QQQ? "
                "If it names none or says all, return all three. Reply JSON {\"markets\": [..]}.\n\nQuestion: " + q[:500])}]}],
            "generationConfig": {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{C.PARSER_MODEL}:generateContent",
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
        mk = [TICKERS[x] for x in json.loads(text).get("markets", []) if x in TICKERS]
        return {"symbols": [s for s in C.SYMBOLS if s in mk] or list(C.SYMBOLS), "parser": "llm"}
    except Exception:
        return {"symbols": [s for s in C.SYMBOLS if s in found] or list(C.SYMBOLS), "parser": "keywords"}


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


def parse_csv(text: str, m: E.Market) -> np.ndarray:
    if len(text.encode()) > C.MAX_CSV_BYTES:
        raise UserError("the CSV is larger than 1 MB")
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows or not {"time", "symbol", "position"} <= set(k.strip() for k in rows[0]):
        raise UserError("the CSV needs the header time,symbol,position")
    pos = np.zeros(m.rets.shape)
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
            pos[m.symbols.index(sym), index[t]] = p
            matched += 1
    if matched == 0:
        raise UserError("no row matched a 4-hour bar open between 1 Jul and 26 Sep 2026 (UTC)")
    return pos


def market_mood() -> dict:
    """Bitget MCP Fear & Greed. Returns {"ok": False, "code": ...} when the service is down."""
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


def run(question: str, csv_text: str | None) -> dict:
    q = parse_question(question or "")
    m = E.load_market(q["symbols"])
    if csv_text:
        runs, invalid, who = [parse_csv(csv_text, m)], 0, "uploaded"
    else:
        all_runs, ref = reference_runs()
        keep = [C.SYMBOLS.index(s) for s in q["symbols"]]
        runs, invalid, who = [r[keep] for r in all_runs], ref["invalid_outputs"], "reference"
        samples = [d for d in ref.get("sample_decisions", []) if d["symbol"] in q["symbols"]]
    res = E.evaluate(runs, m, invalid=invalid)
    res.update({"agent_source": who, "parser": q["parser"], "mood": market_mood(),
                "sample_decisions": [] if csv_text else samples})
    return res
