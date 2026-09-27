import numpy as np

from placebo import config as C
from placebo import engine as E
from placebo import agent as A


def toy_market(T=50, S=2, seed=1):
    rng = np.random.default_rng(seed)
    rets = np.zeros((S, T))
    rets[:, 1:] = rng.normal(0, 0.01, (S, T - 1))
    closes = 100 * np.cumprod(1 + rets, axis=1)
    return E.Market(np.arange(T, dtype=np.int64) * C.BAR_MS, closes, rets, np.zeros((S, T)), ["X", "Y"])


def test_pnl_uses_next_bar_return_and_charges_costs():
    m = toy_market()
    pos = np.zeros_like(m.rets)
    pos[:, 3] = 1  # decided at close of bar 3, held over bar 4
    r = E.pnl(pos, m)
    fee = C.TAKER_FEE + C.SLIPPAGE
    assert np.isclose(r[3], (m.rets[:, 4] - fee).mean())   # enter
    assert np.isclose(r[4], (-fee) * 1.0)                    # exit on the next bar (flat, pays exit)
    assert np.allclose(np.delete(r, [3, 4]), 0)


def test_no_lookahead_in_agent_inputs():
    m = E.load_market()
    t = 200
    before = A.features(m, 0, t)
    m.closes[:, t + 1:] *= 3.0
    m.rets[:, t + 1:] = 0.5
    assert A.features(m, 0, t) == before


def test_placebo_matches_exposure():
    m = toy_market(T=120)
    rng = np.random.default_rng(3)
    pos = rng.choice([-1.0, 0.0, 1.0], size=m.rets.shape)
    eps = [E.episodes(pos[i]) for i in range(pos.shape[0])]
    idx = np.full(pos.shape, -1)
    k = 0
    for i, es in enumerate(eps):
        for j, (a, b) in enumerate(es):
            idx[i, a:b] = k + j
        k += len(es)
    signs = rng.choice([-1.0, 1.0], size=k)
    twin = np.where(idx >= 0, np.abs(pos) * signs[np.maximum(idx, 0)], 0.0)
    assert np.array_equal(np.abs(twin), np.abs(pos))           # same timing and size
    for i in range(pos.shape[0]):                              # sign constant within each episode
        for a, b in eps[i]:
            assert len(set(twin[i, a:b])) == 1


def test_episodes_split_on_sign_change():
    assert E.episodes(np.array([0, 1, 1, -1, -1, 0, 1])) == [(1, 3), (3, 5), (6, 7)]


def test_perfect_foresight_beats_placebo():
    m = toy_market(T=300)
    pos = np.zeros_like(m.rets)
    pos[:, :-1] = np.sign(m.rets[:, 1:])
    res = E.evaluate([pos], m, draws=500)
    assert res["placebo"]["p_value"] < 0.01


def test_verdict_rule():
    good = {"sharpe": 2.0, "max_drawdown": 0.1}
    rule = {"sharpe": 1.0}
    assert E.verdict(0.01, good, rule, 0.9) == C.VERDICT_DEPLOY
    assert E.verdict(0.01, good, rule, 0.5) == C.VERDICT_HALF
    assert E.verdict(0.30, good, rule, 0.9) == C.VERDICT_NO
    assert E.verdict(0.10, {"sharpe": -1, "max_drawdown": 0.1}, rule, 0.9) == C.VERDICT_NO
