"""Internal primitives of stats.py — block bootstrap, percentile, etc."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import stats as st


# ─── stationary_block_bootstrap ────────────────────────────────────

def test_block_bootstrap_returns_correct_length():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 500)
    out = st.stationary_block_bootstrap(r, n_resamples=200, block_size=20)
    assert out.shape == (200,)


def test_block_bootstrap_finite_outputs():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    out = st.stationary_block_bootstrap(r, n_resamples=300, block_size=10)
    assert np.all(np.isfinite(out))


def test_block_bootstrap_deterministic_with_seed():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    a = st.stationary_block_bootstrap(r, n_resamples=100, block_size=20, seed=42)
    b = st.stationary_block_bootstrap(r, n_resamples=100, block_size=20, seed=42)
    np.testing.assert_allclose(a, b)


def test_block_bootstrap_different_seeds_differ():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    a = st.stationary_block_bootstrap(r, n_resamples=200, block_size=20, seed=1)
    b = st.stationary_block_bootstrap(r, n_resamples=200, block_size=20, seed=2)
    assert not np.allclose(a, b)


def test_block_bootstrap_handles_block_size_one():
    """Block size 1 = iid bootstrap (still valid)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200)
    out = st.stationary_block_bootstrap(r, n_resamples=100, block_size=1)
    assert out.shape == (100,)
    assert np.all(np.isfinite(out))


def test_block_bootstrap_zero_volatility():
    """Constant series → zero Sharpe in every resample (divide-by-zero
    guarded in _annualized_sharpe)."""
    r = np.zeros(200)
    out = st.stationary_block_bootstrap(r, n_resamples=50, block_size=10)
    assert (out == 0.0).all()


# ─── _annualized_sharpe ────────────────────────────────────────────

def test_annualized_sharpe_empty():
    assert st._annualized_sharpe(np.array([])) == 0.0


def test_annualized_sharpe_constant():
    assert st._annualized_sharpe(np.zeros(100)) == 0.0


def test_annualized_sharpe_positive_drift():
    r = np.full(252, 0.01)  # 1% per day → huge Sharpe (zero vol)
    # vol = 0 → sharpe = 0 by our defensive guard
    assert st._annualized_sharpe(r) == 0.0


def test_annualized_sharpe_realistic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 252)
    s = st._annualized_sharpe(r)
    # roughly mean/std × √252 ≈ 1.58
    assert 0.5 < s < 3.0


# ─── block_bootstrap_p_value ───────────────────────────────────────

def test_block_bootstrap_p_value_high_diff_signal():
    """If model returns are noticeably higher than null, p-value < 0.05."""
    rng = np.random.default_rng(0)
    model = rng.normal(0.005, 0.01, 300)  # strong positive drift
    null = rng.normal(0.000, 0.01, 300)
    out = st.block_bootstrap_p_value(model, null, block_size=20, n_resamples=500)
    assert "p_value" in out
    assert 0.0 <= out["p_value"] <= 1.0
    # Realistically should be low
    assert out["obs_sharpe_diff"] > 0


def test_block_bootstrap_p_value_no_signal():
    """If model returns are SAME as null, p-value should be roughly 0.5."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 300)
    out = st.block_bootstrap_p_value(r, r, block_size=20, n_resamples=500)
    # obs_sharpe_diff is exactly 0 (model == null) ⇒ p is data-dependent on bootstrap
    assert abs(out["obs_sharpe_diff"]) < 1e-9


# ─── percentile_vs_distribution ────────────────────────────────────

def test_percentile_value_above_all():
    out = st.percentile_vs_distribution(1000.0, np.array([1, 2, 3, 4, 5]))
    assert out == 1.0


def test_percentile_value_below_all():
    out = st.percentile_vs_distribution(-1.0, np.array([1, 2, 3, 4, 5]))
    assert out == 0.0


def test_percentile_value_in_middle():
    out = st.percentile_vs_distribution(3.0, np.array([1, 2, 3, 4, 5]))
    # 3 of 5 are <= 3 → 0.6
    assert abs(out - 0.6) < 1e-9


# ─── daily_returns_from_equity ─────────────────────────────────────

def test_daily_returns_basic():
    eq = pd.DataFrame({"equity": [100.0, 110.0, 99.0, 105.0]})
    r = st.daily_returns_from_equity(eq)
    assert len(r) == 3
    np.testing.assert_allclose(r, [0.10, -0.10, 105/99 - 1], rtol=1e-9)


def test_daily_returns_too_short():
    """Equity with 0 or 1 row → empty array."""
    eq0 = pd.DataFrame({"equity": []})
    assert len(st.daily_returns_from_equity(eq0)) == 0
    eq1 = pd.DataFrame({"equity": [100.0]})
    assert len(st.daily_returns_from_equity(eq1)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
