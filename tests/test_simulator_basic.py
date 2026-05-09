"""Basic simulator accounting tests."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart.simulator import simulate, SimConfig


def _toy_setup(n_days: int = 300) -> tuple[pd.DataFrame, dict, int]:
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(20)]
    price_lookup = {}
    for t in tickers:
        rets = rng.normal(0.001, 0.02, size=n_days)
        closes = 100.0 * np.cumprod(1.0 + rets)
        price_lookup[t] = pd.DataFrame({"date": dates, "close": closes})

    H = 40
    anchor_dates = dates[252::5]  # weekly-ish anchors after warmup
    rows = []
    for ad in anchor_dates:
        anchor_idx = list(dates).index(ad)
        if anchor_idx + H >= n_days:
            continue
        res_date = dates[anchor_idx + H]
        for t in tickers:
            rows.append({
                "ticker": t, "anchor_date": ad,
                "score": rng.random(),
                "anchor_close": float(price_lookup[t]["close"].iloc[anchor_idx]),
                "adv20_usd": 5e7,
                "anchor_idx": anchor_idx,
                "resolution_date": res_date,
            })
    return pd.DataFrame(rows), price_lookup, H


def test_no_negative_cash():
    scores, prices, H = _toy_setup()
    cfg = SimConfig(start_equity=100_000.0)
    out = simulate(scores, prices, cfg)
    assert (out["equity"]["cash"] >= -1.0).all()


def test_no_duplicate_open():
    scores, prices, H = _toy_setup()
    cfg = SimConfig()
    out = simulate(scores, prices, cfg)
    assert (out["equity"]["n_positions"] <= 30).all()


def test_summary_keys():
    scores, prices, H = _toy_setup()
    out = simulate(scores, prices, SimConfig())
    s = out["summary"]
    for k in ["start_equity", "end_equity", "cagr", "max_drawdown", "n_trades",
              "win_rate", "profit_factor"]:
        assert k in s


if __name__ == "__main__":
    test_no_negative_cash()
    test_no_duplicate_open()
    test_summary_keys()
    print("PASS simulator basics")
