"""Tests for random_baseline.py determinism + simple-baseline correctness."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import simulator as sim
from stock_chart import random_baseline as rb


def _toy_setup(seed: int = 0, n_tickers: int = 10, n_days: int = 300):
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    price_lookup = {}
    for t in tickers:
        closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n_days))
        price_lookup[t] = pd.DataFrame({"date": dates, "close": closes})
    rows = []
    H = 40
    for ad in dates[252::5]:
        ai = list(dates).index(ad)
        if ai + H >= n_days:
            continue
        for t in tickers:
            rows.append({
                "ticker": t, "anchor_date": ad,
                "anchor_close": float(price_lookup[t]["close"].iloc[ai]),
                "adv20_usd": 10e6, "anchor_idx": ai,
                f"resolution_date_{H}d": dates[ai + H],
            })
    anchor_meta = pd.DataFrame(rows)
    return anchor_meta, price_lookup, H


def test_random_seeds_deterministic():
    anchor_meta, price_lookup, H = _toy_setup()
    cfg = sim.SimConfig()
    s_a = rb.run_random_seeds(anchor_meta, price_lookup, cfg, horizon=H,
                                 n_seeds=3, base_seed=42)
    s_b = rb.run_random_seeds(anchor_meta, price_lookup, cfg, horizon=H,
                                 n_seeds=3, base_seed=42)
    np.testing.assert_allclose(s_a["cagr"].to_numpy(), s_b["cagr"].to_numpy(),
                                  rtol=1e-9)


def test_random_seeds_different_base_seed_differs():
    anchor_meta, price_lookup, H = _toy_setup()
    cfg = sim.SimConfig()
    s_a = rb.run_random_seeds(anchor_meta, price_lookup, cfg, horizon=H,
                                 n_seeds=3, base_seed=42)
    s_c = rb.run_random_seeds(anchor_meta, price_lookup, cfg, horizon=H,
                                 n_seeds=3, base_seed=99)
    # At least one CAGR should differ
    assert not np.allclose(s_a["cagr"].to_numpy(), s_c["cagr"].to_numpy(),
                              atol=1e-6)


def test_simple_baselines_known_set():
    """SIMPLE_BASELINES should contain exactly the 5 documented baselines."""
    expected = {"rank_252d_return", "rank_60d_return", "rank_ma250_extension",
                "rank_52w_high_distance", "rank_inv_60d_vol"}
    assert set(rb.SIMPLE_BASELINES) == expected


def test_unknown_baseline_raises():
    """Asking for a baseline that doesn't exist should raise ValueError."""
    rng = np.random.default_rng(0)
    feats = pd.DataFrame({
        "ticker": ["A", "B"], "anchor_date": [pd.Timestamp("2024-01-05")] * 2,
        "ret_252d": [0.5, 0.3], "ret_63d": [0.1, 0.2],
        "ratio_ma200": [1.5, 1.4], "pct_from_252d_high": [-0.05, -0.10],
        "vol_63d": [0.3, 0.5],
    })
    with pytest.raises(ValueError, match="unknown baseline"):
        rb._score_from_features(feats, "nonsense_baseline")


def test_all_simple_baselines_produce_scores():
    """Each named baseline must produce a numeric score per row, no NaN."""
    feats = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "anchor_date": [pd.Timestamp("2024-01-05")] * 3,
        "ret_252d": [0.5, 0.3, 0.8],
        "ret_63d": [0.1, 0.2, -0.05],
        "ratio_ma200": [1.5, 1.4, 2.0],
        "pct_from_252d_high": [-0.05, -0.10, -0.02],
        "vol_63d": [0.3, 0.5, 0.2],
    })
    for bn in rb.SIMPLE_BASELINES:
        s = rb._score_from_features(feats, bn)
        assert len(s) == 3
        assert not s.isna().any(), f"baseline {bn} produced NaN"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
