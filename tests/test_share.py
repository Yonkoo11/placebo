"""Shareable trial links: the token round-trips, the engine is deterministic, and hostile tokens fail readably."""
import base64
import json
import zlib

import numpy as np
import pytest

from placebo import config as C
from placebo import engine as E
from placebo import service as S
from placebo import share


def test_engine_is_deterministic_for_the_same_input():
    m = E.load_market(["NVDAUSDT"])
    runs, _ = S.reference_runs()
    pos = [runs[0][[0]]]
    a, b = E.evaluate(pos, m, draws=300), E.evaluate(pos, m, draws=300)
    assert a["placebo"]["p_value"] == b["placebo"]["p_value"]
    assert a["placebo"]["histogram"] == b["placebo"]["histogram"]
    assert C.PLACEBO_SEED == 7


def test_upload_token_round_trips_positions_and_runs():
    m = E.load_market()
    rng = np.random.default_rng(2)
    runs = [rng.choice([-1.0, 0.0, 1.0], size=m.rets.shape) for _ in range(3)]
    tok = share.encode("Should this agent trade NVDA?", "us_open", m.symbols, runs)
    assert len(tok) < 2500 and "=" not in tok and "+" not in tok and "/" not in tok
    d = share.decode(tok)
    assert d["hours"] == "us_open" and d["symbols"] == m.symbols and d["bars"] == 528
    back = share.runs_from(d, m.rets.shape)
    assert len(back) == 3 and all(np.array_equal(x, y) for x, y in zip(runs, back))


def test_question_token_replays_the_same_trial_without_the_parser():
    first = S.run("NVDA", None)
    again = S.run_token(first["token"])
    assert again["placebo"]["p_value"] == first["placebo"]["p_value"]
    assert again["verdict"] == first["verdict"]
    assert again["parser"] == "link" and again["from_link"] is True
    assert again["token"] == first["token"]


def _wrap(payload: dict) -> str:
    z = zlib.compress(json.dumps(payload).encode(), 9)
    return base64.urlsafe_b64encode(z).rstrip(b"=").decode()


@pytest.mark.parametrize("token,msg", [
    ("", "carries no trial"),
    ("not base64 at all!!", "did not decode"),
    (base64.urlsafe_b64encode(b"plain bytes").decode().rstrip("="), "did not decode"),
    ("A" * 16_001, "too long"),
    (_wrap([1, 2, 3]), "unknown format"),
    (_wrap({"v": 2, "q": "x", "h": "all", "s": ["NVDAUSDT"]}), "unknown format"),
    (_wrap({"v": 1, "q": "x", "h": "weekends", "s": ["NVDAUSDT"]}), "trading-hours rule"),
    (_wrap({"v": 1, "q": "x", "h": "all", "s": ["AAPLUSDT"]}), "market Placebo does not have"),
    (_wrap({"v": 1, "q": "x", "h": "all", "s": []}), "market Placebo does not have"),
    (_wrap({"v": 1, "q": "x" * 501, "h": "all", "s": ["NVDAUSDT"]}), "question is not readable"),
    (_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": 528, "p": ["a"] * 5}), "more than 4 runs"),
    (_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": "528", "p": ["a"]}), "bar count is not readable"),
    (_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": 528, "p": "a"}), "positions are not readable"),
])
def test_hostile_tokens_fail_with_a_sentence(token, msg):
    with pytest.raises(share.UserError, match=msg):
        S.run_token(token)


def test_token_with_wrong_bar_count_or_invalid_code_is_rejected():
    m = E.load_market(["NVDAUSDT"])
    good = share.pack_run(np.zeros((1, 528)))
    with pytest.raises(share.UserError, match="does not match the market snapshot"):
        S.run_token(_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": 100, "p": [good]}))
    with pytest.raises(share.UserError, match="do not match the market size"):
        S.run_token(_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": 528, "p": [good[:-10]]}))
    bad = base64.urlsafe_b64encode(bytes([0xFF]) * 132).rstrip(b"=").decode()  # every code is 3
    with pytest.raises(share.UserError, match="not -1, 0 or 1"):
        S.run_token(_wrap({"v": 1, "q": "x", "h": "all", "s": ["NVDAUSDT"], "t": 528, "p": [bad]}))


def test_zlib_bomb_is_refused_before_it_inflates():
    bomb = base64.urlsafe_b64encode(zlib.compress(b"0" * 5_000_000, 9)).rstrip(b"=").decode()
    with pytest.raises(share.UserError, match="too large"):
        share.decode(bomb)


def test_example_csv_parses_and_runs_as_two_runs():
    text = open("public/example-positions.csv").read()
    runs = S.parse_csv(text, E.load_market())
    assert len(runs) == 2 and runs[0].shape == (3, 528)
    res = S.run("Should this agent trade NVDA, TSLA, QQQ?", text)
    assert res["verdict"] in (C.VERDICT_NO, C.VERDICT_HALF, C.VERDICT_DEPLOY)
    assert res["agent_source"] == "uploaded" and len(res["p_values_per_run"]) == 2
    assert 0 < res["bars_matched"] <= 528
    assert S.run_token(res["token"])["placebo"]["p_value"] == res["placebo"]["p_value"]
