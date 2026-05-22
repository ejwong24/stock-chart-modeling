"""Tests for features.py — engineered shape descriptors."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import features as feat


def _make_series(seed: int = 0, n: int = 260):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n))
    vols = rng.lognormal(14, 0.4, n)
    return closes, vols


# ─── _compute_window_features ──────────────────────────────────────────

def test_feature_count_matches_FEATURE_COLS():
    closes, vols = _make_series()
    spy = 400.0 * np.cumprod(1.0 + np.random.default_rng(1).normal(0.0005, 0.012, 300))
    f = feat._compute_window_features(closes, vols, spy, 259)
    assert set(f.keys()) == set(feat.FEATURE_COLS), \
        f"feature mismatch: missing={set(feat.FEATURE_COLS)-set(f.keys())} extra={set(f.keys())-set(feat.FEATURE_COLS)}"


def test_features_all_finite():
    closes, vols = _make_series()
    spy = 400.0 * np.cumprod(1.0 + np.random.default_rng(1).normal(0.0005, 0.012, 300))
    f = feat._compute_window_features(closes, vols, spy, 259)
    for k, v in f.items():
        assert np.isfinite(v), f"feature {k}={v} not finite"


def test_features_no_spy_works():
    """If SPY series isn't provided, beta should be 0 (not crash)."""
    closes, vols = _make_series()
    f = feat._compute_window_features(closes, vols, None, 259)
    assert f["beta_spy_63d"] == 0.0


def test_constant_price_finite_features():
    """A flat-line stock should produce finite features (zero vol etc.)."""
    closes = np.full(260, 50.0)
    vols = np.full(260, 1000.0)
    f = feat._compute_window_features(closes, vols, None, 259)
    for k, v in f.items():
        assert np.isfinite(v), f"flat-price feature {k}={v} not finite"
    assert abs(f["ret_252d"]) < 1e-9
    assert f["vol_252d"] >= 0  # could be 0 or near 0


def test_strong_uptrend_signs():
    """A monotonically rising price should produce positive returns + ratio_ma > 1."""
    closes = np.linspace(50.0, 200.0, 260)
    vols = np.full(260, 5000.0)
    f = feat._compute_window_features(closes, vols, None, 259)
    assert f["ret_252d"] > 0
    assert f["ratio_ma200"] > 1.0
    assert f["pct_from_252d_high"] >= -1e-9   # latest is near the high
    assert f["up_day_frac_63d"] > 0.95  # nearly all up days


def test_compute_for_anchors_with_missing_ticker(tmp_path):
    """compute_for_anchors() should skip tickers whose parquet doesn't exist."""
    anchor_df = pd.DataFrame({
        "ticker": ["NOPE"],
        "anchor_date": [pd.Timestamp("2024-01-15")],
        "anchor_idx": [259],
    })
    out = feat.compute_for_anchors(tmp_path, anchor_df, spy_path=None)
    # No tickers found → empty DataFrame but with the expected columns
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_compute_for_anchors_with_one_real(tmp_path):
    """Real ticker parquet should yield one row of features."""
    closes, vols = _make_series(seed=42)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=260, freq="B"),
        "close": closes, "volume": vols,
    })
    df.to_parquet(tmp_path / "ABC.parquet", index=False)
    anchor_df = pd.DataFrame({
        "ticker": ["ABC"],
        "anchor_date": [pd.Timestamp(df["date"].iloc[259])],
        "anchor_idx": [259],
    })
    out = feat.compute_for_anchors(tmp_path, anchor_df, spy_path=None)
    assert len(out) == 1
    assert set(feat.FEATURE_COLS).issubset(out.columns)
    for c in feat.FEATURE_COLS:
        assert np.isfinite(out[c].iloc[0]), f"{c} not finite"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
