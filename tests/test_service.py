import numpy as np
import pytest

from placebo import engine as E
from placebo import service as S


def test_csv_accepts_iso_ms_and_short_tickers():
    m = E.load_market()
    t0 = int(m.times[10])
    text = f"time,symbol,position\n{t0},NVDA,1\n2026-07-01T00:00:00Z,TSLAUSDT,-1\n"
    runs = S.parse_csv(text, m)
    pos = runs[0]
    assert len(runs) == 1 and pos[0, 10] == 1 and pos[1, 0] == -1 and abs(pos).sum() == 2


def test_csv_run_column_gives_one_run_per_value():
    m = E.load_market()
    t0 = int(m.times[10])
    text = f"time,symbol,position,run\n{t0},NVDA,1,a\n{t0},NVDA,-1,b\n{t0},TSLA,1,a\n"
    runs = S.parse_csv(text, m)
    assert len(runs) == 2 and runs[0][0, 10] == 1 and runs[1][0, 10] == -1 and runs[0][1, 10] == 1
    assert E.consistency(runs) is not None


def test_csv_too_many_runs_is_readable():
    m = E.load_market()
    t0 = int(m.times[10])
    text = "time,symbol,position,run\n" + "".join(f"{t0},NVDA,1,{i}\n" for i in range(5))
    with pytest.raises(S.UserError, match="at most 4 runs"):
        S.parse_csv(text, m)


@pytest.mark.parametrize("text,msg", [
    ("a,b\n1,2", "header"),
    ("time,symbol,position\n2026-07-01T00:00:00Z,AAPL,1", "unknown symbol"),
    ("time,symbol,position\n2026-07-01T00:00:00Z,NVDA,2", "must be -1, 0 or 1"),
    ("time,symbol,position\nyesterday,NVDA,1", "not readable"),
    ("time,symbol,position\n2020-01-01T00:00:00Z,NVDA,1", "no row matched"),
])
def test_csv_rejections_are_readable(text, msg):
    with pytest.raises(S.UserError, match=msg):
        S.parse_csv(text, E.load_market())


def test_csv_size_limit():
    with pytest.raises(S.UserError, match="1 MB"):
        S.parse_csv("time,symbol,position\n" + "x" * 1_000_001, E.load_market())


def test_keyword_hours():
    assert S.keyword_hours("What if it only trades while the US market is closed?") == "us_closed"
    assert S.keyword_hours("only over the weekend") == "us_closed"
    assert S.keyword_hours("during regular hours only") == "us_open"
    assert S.keyword_hours("Should I let it trade NVDA?") == "all"


def test_restrict_hours_zeroes_outside_positions():
    m = E.load_market()
    pos = np.ones(m.rets.shape)
    closed = S.restrict_hours(pos, m, "us_closed")
    opened = S.restrict_hours(pos, m, "us_open")
    assert np.array_equal(closed + opened, pos)
    assert 0 < closed.sum() < pos.sum()
