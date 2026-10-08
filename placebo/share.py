"""Shareable trial links. The link carries the whole trial input; the page recomputes the trial.

There is no database. A token is URL-safe base64 of zlib-compressed compact JSON:
  {"v": 1, "q": question, "h": hours, "s": [symbols]}                  a question trial
  {... , "t": bars, "b": bars with a CSV row, "p": [one base64 string per run]}   an uploaded trial
Positions are packed 2 bits per bar per symbol (0 flat, 1 long, 2 short), row-major over
(symbol, bar), so 528 bars x 3 symbols x 4 runs is about 1.6 KB before compression.
The engine seeds its placebo draws (config.PLACEBO_SEED), so a link always reopens to the same
numbers. Every decode failure is a UserError with a sentence a person can act on.
"""
import base64
import binascii
import json
import zlib

import numpy as np

from . import config as C

MAX_TOKEN_CHARS = 16_000
MAX_PAYLOAD_BYTES = 200_000
HOURS = ("all", "us_closed", "us_open")
CODES = {0.0: 0, 1.0: 1, -1.0: 2}
VALUES = np.array([0.0, 1.0, -1.0, np.nan])  # code 3 is invalid


class UserError(ValueError):
    """Shown to the user as {reason}."""


def pack_run(pos: np.ndarray) -> str:
    flat = pos.reshape(-1)
    codes = np.array([CODES[float(v)] for v in flat], dtype=np.uint8)
    pad = (-len(codes)) % 4
    codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    quads = codes.reshape(-1, 4)
    packed = (quads[:, 0] << 6) | (quads[:, 1] << 4) | (quads[:, 2] << 2) | quads[:, 3]
    return base64.urlsafe_b64encode(packed.astype(np.uint8).tobytes()).rstrip(b"=").decode()


def unpack_run(text: str, shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    try:
        raw = np.frombuffer(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)), dtype=np.uint8)
    except (binascii.Error, ValueError):
        raise UserError("the link's positions are not readable")
    if len(raw) != (n + 3) // 4:
        raise UserError("the link's positions do not match the market size")
    codes = np.stack([(raw >> 6) & 3, (raw >> 4) & 3, (raw >> 2) & 3, raw & 3], axis=1).reshape(-1)[:n]
    if (codes == 3).any():
        raise UserError("the link holds a position that is not -1, 0 or 1")
    return VALUES[codes].reshape(shape)


def encode(question: str, hours: str, symbols: list[str], runs: list[np.ndarray] | None = None,
           bars_matched: int | None = None) -> str:
    payload = {"v": 1, "q": (question or "")[:500], "h": hours, "s": list(symbols)}
    if runs is not None:
        payload["t"] = int(runs[0].shape[1])
        payload["b"] = int(bars_matched if bars_matched is not None else runs[0].shape[1])
        payload["p"] = [pack_run(r) for r in runs]
    z = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
    return base64.urlsafe_b64encode(z).rstrip(b"=").decode()


def decode(token: str) -> dict:
    """{"question", "hours", "symbols", "bars" | None, "packed" | None}; raises UserError."""
    if not isinstance(token, str) or not token:
        raise UserError("the link carries no trial")
    if len(token) > MAX_TOKEN_CHARS:
        raise UserError("the link is too long to be a Placebo trial")
    try:
        z = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        d = zlib.decompressobj()
        body = d.decompress(z, MAX_PAYLOAD_BYTES)
        if d.unconsumed_tail or not d.eof:
            raise UserError("the link is too large to be a Placebo trial")
        p = json.loads(body)
    except UserError:
        raise
    except (binascii.Error, zlib.error, ValueError, UnicodeDecodeError):
        raise UserError("the link is not a Placebo trial link (it did not decode)")
    if not isinstance(p, dict) or p.get("v") != 1:
        raise UserError("the link is not a Placebo trial link (unknown format)")
    q, h, s = p.get("q", ""), p.get("h"), p.get("s")
    if not isinstance(q, str) or len(q) > 500:
        raise UserError("the link's question is not readable")
    if h not in HOURS:
        raise UserError("the link's trading-hours rule is not one Placebo knows")
    if (not isinstance(s, list) or not s or len(set(s)) != len(s) or any(x not in C.SYMBOLS for x in s)):
        raise UserError("the link names a market Placebo does not have (NVDA, TSLA or QQQ)")
    symbols = [x for x in C.SYMBOLS if x in s]
    out = {"question": q, "hours": h, "symbols": symbols, "bars": None, "packed": None, "bars_matched": None}
    if "p" in p or "t" in p:
        t, packed, b = p.get("t"), p.get("p"), p.get("b", p.get("t"))
        if not isinstance(t, int) or isinstance(t, bool) or t <= 0:
            raise UserError("the link's bar count is not readable")
        if not isinstance(b, int) or isinstance(b, bool) or not 0 <= b <= t:
            raise UserError("the link's matched-bar count is not readable")
        out["bars_matched"] = b
        if not isinstance(packed, list) or not packed or not all(isinstance(x, str) for x in packed):
            raise UserError("the link's positions are not readable")
        if len(packed) > C.MAX_RUNS:
            raise UserError(f"the link holds more than {C.MAX_RUNS} runs")
        out.update({"bars": t, "packed": packed})
    return out


def runs_from(decoded: dict, shape: tuple[int, int]) -> list[np.ndarray]:
    if decoded["bars"] != shape[1]:
        raise UserError("the link's bar count does not match the market snapshot")
    return [unpack_run(x, shape) for x in decoded["packed"]]
