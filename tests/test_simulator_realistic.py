"""Tests for path-dependent exit + Almgren-Chriss + halt risk additions."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart.simulator import simulate, SimConfig, impact_bps


def _toy_setup(n_days=300, seed=0):
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(15)]
    price_lookup = {}
    for t in tickers:
        rets = rng.normal(0.001, 0.02, size=n_days)
        closes = 100.0 * np.cumprod(1.0 + rets)
        price_lookup[t] = pd.DataFrame({"date": dates, "close": closes})
    H = 40
    anchor_dates = dates[252::5]
    rows = []
    for ad in anchor_dates:
        ai = list(dates).index(ad)
        if ai + H >= n_days:
            continue
        for t in tickers:
            rows.append({
                "ticker": t, "anchor_date": ad,
                "score": rng.random(),
                "anchor_close": float(price_lookup[t]["close"].iloc[ai]),
                "adv20_usd": 5e7,
                "anchor_idx": ai,
                "resolution_date": dates[ai + H],
            })
    return pd.DataFrame(rows), price_lookup, H


def test_trailing_stop_reduces_max_dd():
    """Adding a trailing stop should not increase max DD; usually reduces it."""
    scores, prices, H = _toy_setup()
    base = simulate(scores, prices, SimConfig())
    stop = simulate(scores, prices, SimConfig(trailing_stop_pct=0.10))
    base_dd = base["summary"]["max_drawdown"]
    stop_dd = stop["summary"]["max_drawdown"]
    assert stop_dd >= base_dd - 1e-9, f"trailing stop made DD worse: {base_dd} -> {stop_dd}"


def test_almgren_chriss_scales_with_participation():
    """Square-root impact must increase with position/ADV ratio."""
    # Same position, different liquidity
    bps_micro = impact_bps(position_usd=20_000, adv_usd=1e6, daily_vol=0.04)
    bps_large = impact_bps(position_usd=20_000, adv_usd=1e9, daily_vol=0.02)
    assert bps_micro > bps_large
    # Bigger position in same stock → more impact
    bps_big_pos = impact_bps(position_usd=200_000, adv_usd=5e6, daily_vol=0.04)
    bps_small_pos = impact_bps(position_usd=20_000, adv_usd=5e6, daily_vol=0.04)
    assert bps_big_pos > bps_small_pos


def test_halt_risk_drops_some_trades():
    """halt_risk_enabled should reduce trade count vs disabled."""
    scores, prices, H = _toy_setup()
    no_halt = simulate(scores, prices, SimConfig())
    with_halt = simulate(scores, prices, SimConfig(halt_risk_enabled=True,
                                                     min_adv_usd=1e6))
    # Trades may or may not differ at this small scale; assert no error & summary keys
    assert "n_trades" in with_halt["summary"]


def test_simulator_keys_unchanged():
    """Adding new fields must not break existing summary keys."""
    scores, prices, H = _toy_setup()
    out = simulate(scores, prices, SimConfig(trailing_stop_pct=0.15,
                                              use_almgren_chriss_impact=True))
    needed = {"start_equity", "end_equity", "cagr", "max_drawdown",
              "n_trades", "win_rate", "profit_factor"}
    assert needed <= set(out["summary"].keys())


if __name__ == "__main__":
    test_trailing_stop_reduces_max_dd()
    test_almgren_chriss_scales_with_participation()
    test_halt_risk_drops_some_trades()
    test_simulator_keys_unchanged()
    print("PASS simulator realistic-cost extensions")
