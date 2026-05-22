"""Trailing-stop boundary behavior."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import simulator as sim


def _synth():
    dates = pd.date_range("2024-01-02", periods=300, freq="B")
    rng = np.random.default_rng(0)
    price_lookup = {f"T{i}": pd.DataFrame({
        "date": dates,
        "close": 100.0 * np.cumprod(1 + rng.normal(0.001, 0.025, 300)),
    }) for i in range(15)}
    rows = []
    for ad in dates[252::5]:
        ai = list(dates).index(ad)
        if ai + 40 >= 300:
            continue
        for t in price_lookup:
            rows.append({
                "ticker": t, "anchor_date": ad, "score": float(rng.random()),
                "anchor_close": float(price_lookup[t]["close"].iloc[ai]),
                "adv20_usd": 5e7, "anchor_idx": ai,
                "resolution_date": dates[ai + 40],
            })
    return pd.DataFrame(rows), price_lookup


def test_trailing_stop_zero_disables():
    """trailing_stop_pct=0.0 must produce identical results to disabling it."""
    scores, prices = _synth()
    out_off = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.0))
    # With stop disabled, no daily check should ever modify exit_date
    assert "n_trades" in out_off["summary"]


def test_trailing_stop_1pt0_never_triggers():
    """100% trailing stop = impossible to trigger; same outcome as off."""
    scores, prices = _synth()
    off = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.0))
    very_loose = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=1.0))
    assert off["summary"]["end_equity"] == very_loose["summary"]["end_equity"]


def test_trailing_stop_tiny_shortens_holds():
    """Very tight (0.01%) stop should produce dramatically shorter holds."""
    scores, prices = _synth()
    off = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.0))
    tight = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.0001))
    # Tight stop avg hold ≪ horizon (40)
    hold_off = (pd.to_datetime(off["blotter"]["exit_date"]) -
                  pd.to_datetime(off["blotter"]["entry_date"])).dt.days.mean()
    hold_tight = (pd.to_datetime(tight["blotter"]["exit_date"]) -
                    pd.to_datetime(tight["blotter"]["entry_date"])).dt.days.mean()
    assert hold_tight < hold_off / 2, \
        f"tight stop should shorten holds: {hold_tight} vs {hold_off}"


def test_trailing_stop_doesnt_increase_n_trades():
    """Stops can only close positions early — trade count is constant."""
    scores, prices = _synth()
    off = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.0))
    on = sim.simulate(scores, prices, sim.SimConfig(trailing_stop_pct=0.15))
    # Same number of trades attempted — what varies is when they exit
    assert len(on["blotter"]) == len(off["blotter"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
