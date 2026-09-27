import pytest

from placebo import engine as E
from placebo import service as S


def test_csv_accepts_iso_ms_and_short_tickers():
    m = E.load_market()
    t0 = int(m.times[10])
    text = f"time,symbol,position\n{t0},NVDA,1\n2026-07-01T00:00:00Z,TSLAUSDT,-1\n"
    pos = S.parse_csv(text, m)
    assert pos[0, 10] == 1 and pos[1, 0] == -1 and abs(pos).sum() == 2


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
