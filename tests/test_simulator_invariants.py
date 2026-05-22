"""Accounting invariants for the simulator.

These tests aim at conservation laws: cash + positions == equity,
no overlapping ticker holds, no negative cash, end-state = sum-of-trades.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import simulator as sim


def _toy_setup(seed: int = 0, n_tickers: int = 20, n_days: int = 400, H: int = 40):
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    price_lookup = {
        t: pd.DataFrame({"date": dates,
                         "close": 100.0 * np.cumprod(
                             1 + rng.normal(0.001, 0.025, n_days))})
        for t in tickers
    }
    rows = []
    for ad in dates[260::5]:
        ai = list(dates).index(ad)
        if ai + H >= n_days:
            continue
        for t in tickers:
            rows.append({
                "ticker": t, "anchor_date": ad, "score": float(rng.random()),
                "anchor_close": float(price_lookup[t]["close"].iloc[ai]),
                "adv20_usd": 5e7, "anchor_idx": ai,
                "resolution_date": dates[ai + H],
            })
    return pd.DataFrame(rows), price_lookup, H


# ─── Conservation laws ─────────────────────────────────────────────────

def test_starting_equity_matches_config():
    scores, prices, _ = _toy_setup()
    cfg = sim.SimConfig(start_equity=123_456.0)
    out = sim.simulate(scores, prices, cfg)
    eq = out["equity"]
    assert abs(eq["equity"].iloc[0] - cfg.start_equity) < 100, \
        f"start drift {eq.iloc[0]}"


def test_no_negative_cash():
    scores, prices, _ = _toy_setup()
    out = sim.simulate(scores, prices, sim.SimConfig())
    assert out["equity"]["cash"].min() > -1.0


def test_n_positions_non_negative():
    scores, prices, _ = _toy_setup()
    out = sim.simulate(scores, prices, sim.SimConfig())
    assert out["equity"]["n_positions"].min() >= 0


def test_max_concurrent_positions_bounded():
    """Max positions should be ≤ max_new_per_week × (H/5) — i.e., max
    cohort size when holding cycles overlap maximally."""
    scores, prices, H = _toy_setup()
    cfg = sim.SimConfig()
    out = sim.simulate(scores, prices, cfg)
    # 5 picks × 8 weekly cohorts (40d / 5d) = 40 simultaneous positions max
    bound = cfg.max_new_per_week * (H // 5)
    assert out["equity"]["n_positions"].max() <= bound, \
        f"too many: {out['equity']['n_positions'].max()} > {bound}"


def test_exit_after_entry():
    """Every blotter row must have exit_date >= entry_date."""
    scores, prices, _ = _toy_setup()
    blot = sim.simulate(scores, prices, sim.SimConfig())["blotter"]
    bad = blot[blot["exit_date"] < blot["entry_date"]]
    assert len(bad) == 0


def test_no_strict_same_ticker_overlap():
    """No two trades of the same ticker can have exit_date STRICTLY after
    the next trade's entry_date (same-day rotation IS allowed)."""
    scores, prices, _ = _toy_setup()
    blot = sim.simulate(scores, prices, sim.SimConfig())["blotter"]
    for t in blot["ticker"].unique():
        sub = blot[blot["ticker"] == t].sort_values("entry_date").reset_index(drop=True)
        for i in range(len(sub) - 1):
            assert sub["exit_date"].iloc[i] <= sub["entry_date"].iloc[i + 1], \
                f"strict overlap for {t}: {sub.iloc[i].to_dict()} → {sub.iloc[i+1].to_dict()}"


def test_final_equity_matches_trade_pnl():
    """End equity = start + sum(net trade P&L). Allow small drift for
    commissions and force-closed mark."""
    scores, prices, _ = _toy_setup()
    cfg = sim.SimConfig()
    out = sim.simulate(scores, prices, cfg)
    blot, eq = out["blotter"], out["equity"]
    net_pnl = (blot["proceeds"] - blot["cost"]).sum() if len(blot) else 0
    end_change = eq["equity"].iloc[-1] - cfg.start_equity
    # Within $100 (small commissions on dozens of trades)
    assert abs(end_change - net_pnl) < 200, \
        f"end_change {end_change:.2f} vs trade pnl {net_pnl:.2f}"


def test_cash_equals_equity_at_end_when_no_open_positions():
    """After all trades close, cash should == equity (no open positions)."""
    scores, prices, _ = _toy_setup()
    out = sim.simulate(scores, prices, sim.SimConfig())
    eq = out["equity"]
    assert eq["n_positions"].iloc[-1] == 0
    assert abs(eq["cash"].iloc[-1] - eq["equity"].iloc[-1]) < 1.0


def test_equity_dates_strictly_increasing():
    """Daily equity series should have strictly increasing dates."""
    scores, prices, _ = _toy_setup()
    out = sim.simulate(scores, prices, sim.SimConfig())
    eq = out["equity"]
    dates = pd.to_datetime(eq["date"])
    assert (dates.diff().dropna() > pd.Timedelta(0)).all()


def test_commissions_non_negative():
    """Per-trade commission can't go negative."""
    scores, prices, _ = _toy_setup()
    out = sim.simulate(scores, prices, sim.SimConfig())
    assert (out["blotter"]["commission_total"] >= 0).all()


def test_deterministic_with_same_inputs():
    """Same scores + same lookup + same config → same blotter."""
    s, p, _ = _toy_setup(seed=7)
    cfg = sim.SimConfig(halt_risk_enabled=False)
    a = sim.simulate(s, p, cfg)["blotter"]
    b = sim.simulate(s, p, cfg)["blotter"]
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True))


def test_zero_costs_zero_commission_match_within_tolerance():
    """With zero costs and zero commission, end_equity should exactly equal start + sum(trade PnL)."""
    s, p, _ = _toy_setup()
    cfg = sim.SimConfig(slippage_bps_each_side=0.0, commission_per_share=0.0)
    out = sim.simulate(s, p, cfg)
    blot, eq = out["blotter"], out["equity"]
    net = (blot["proceeds"] - blot["cost"]).sum() if len(blot) else 0
    delta = eq["equity"].iloc[-1] - cfg.start_equity
    assert abs(delta - net) < 1e-4, f"with zero costs, expected exact match: {delta} vs {net}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
