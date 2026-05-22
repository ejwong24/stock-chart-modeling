"""Tests for stats module — bootstrap, deflated Sharpe, SPA, post-tax, eff N."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import stats as st


# ─── effective_sample_size ─────────────────────────────────────────────

def test_effective_sample_size_no_overlap():
    """Non-overlapping trades should give effective N = total trade count."""
    rng = np.random.default_rng(0)
    # 10 trades, each held 5 days, spaced 20 days apart → no overlap
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        entry = base + pd.Timedelta(days=i * 20)
        exit_ = entry + pd.Timedelta(days=5)
        rows.append({"entry_date": entry, "exit_date": exit_})
    blot = pd.DataFrame(rows)
    n_eff = st.effective_sample_size(blot, hold_days=5)
    # When trades don't overlap, each is uniquely 1.0 → N_eff = total count
    assert 9.5 <= n_eff <= 10.5, f"expected ~10, got {n_eff}"


def test_effective_sample_size_full_overlap():
    """All trades simultaneous → effective N collapses dramatically."""
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(20):
        rows.append({"entry_date": base, "exit_date": base + pd.Timedelta(days=40)})
    blot = pd.DataFrame(rows)
    n_eff = st.effective_sample_size(blot, hold_days=40)
    # 20 trades all open simultaneously → each gets uniqueness = 1/20
    # → N_eff = 20 × (1/20) = 1
    assert n_eff < 2.0, f"20 overlapping trades collapsed to {n_eff}, expected ~1"


def test_effective_sample_size_empty_blotter():
    """Empty blotter should not crash."""
    blot = pd.DataFrame(columns=["entry_date", "exit_date"])
    n_eff = st.effective_sample_size(blot, hold_days=40)
    assert n_eff == 0.0


# ─── bootstrap_cagr_ci ──────────────────────────────────────────────────

def test_bootstrap_cagr_ci_shape():
    eq = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=500, freq="B"),
        "equity": 100_000 * np.cumprod(1.0 + np.random.default_rng(0).normal(0.0005, 0.01, 500)),
    })
    ci = st.bootstrap_cagr_ci(eq, n_resamples=500, block_size=20)
    assert "cagr_ci" in ci and "sharpe_ci" in ci and "maxdd_ci" in ci
    # Each CI is (lo, median, hi) with lo <= median <= hi
    for k in ("cagr_ci", "sharpe_ci", "maxdd_ci"):
        lo, med, hi = ci[k]
        assert lo <= med <= hi, f"{k} CI ordering broken: {ci[k]}"


def test_bootstrap_cagr_ci_too_short_equity():
    """Tiny equity series should not crash."""
    eq = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "equity": [100_000.0]})
    ci = st.bootstrap_cagr_ci(eq, n_resamples=100, block_size=10)
    assert ci["cagr_ci"] == (0, 0, 0)


def test_bootstrap_cagr_ci_constant_equity():
    """Flat equity curve should give zero CAGR / Sharpe / DD."""
    eq = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=200, freq="B"),
        "equity": [100_000.0] * 200,
    })
    ci = st.bootstrap_cagr_ci(eq, n_resamples=100, block_size=10)
    assert abs(ci["cagr_ci"][1]) < 1e-9
    assert ci["sharpe_ci"][1] == 0.0
    assert abs(ci["maxdd_ci"][1]) < 1e-9


# ─── deflated_sharpe_ratio ──────────────────────────────────────────────

def test_dsr_basic_shape():
    out = st.deflated_sharpe_ratio(observed_sr=0.5, n_obs=1000, n_trials=72)
    needed = {"observed_sharpe_annualized", "n_trials", "n_observations",
              "deflated_sharpe_threshold_annualized",
              "deflated_sharpe_p_value_significant"}
    assert needed <= set(out.keys())
    # threshold must be > 0 for n_trials > 1
    assert out["deflated_sharpe_threshold_annualized"] > 0


def test_dsr_threshold_grows_with_trials():
    """More trials → higher significance threshold."""
    sr_1 = st.deflated_sharpe_ratio(0.5, 1000, n_trials=1)
    sr_10 = st.deflated_sharpe_ratio(0.5, 1000, n_trials=10)
    sr_100 = st.deflated_sharpe_ratio(0.5, 1000, n_trials=100)
    t1 = sr_1["deflated_sharpe_threshold_annualized"]
    t10 = sr_10["deflated_sharpe_threshold_annualized"]
    t100 = sr_100["deflated_sharpe_threshold_annualized"]
    assert t1 <= t10 <= t100, f"DSR thresholds should grow: {t1}, {t10}, {t100}"


def test_dsr_p_value_increases_with_sharpe():
    """At fixed n_trials, higher observed SR should give higher significance."""
    out_low = st.deflated_sharpe_ratio(0.1, 1000, n_trials=10)
    out_hi = st.deflated_sharpe_ratio(2.0, 1000, n_trials=10)
    assert out_hi["deflated_sharpe_p_value_significant"] > \
           out_low["deflated_sharpe_p_value_significant"]


# ─── post_tax_cagr ──────────────────────────────────────────────────────

def test_post_tax_cagr_with_pnl_columns():
    """Blotter with cost + proceeds columns should compute clean pre/post."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=300, freq="W-FRI")
    rows = []
    for d in dates:
        cost = 4000.0
        ret = rng.normal(0.02, 0.05)
        rows.append({
            "ticker": "FOO", "entry_date": d, "exit_date": d + pd.Timedelta(days=60),
            "trade_return_pct": ret, "cost": cost, "proceeds": cost * (1 + ret),
        })
    blot = pd.DataFrame(rows)
    out = st.post_tax_cagr(blot, federal_rate=0.32, state_rate=0.06)
    assert "pre_tax_cagr" in out and "post_tax_cagr" in out
    # Post-tax should be <= pre-tax (tax drag)
    assert out["post_tax_cagr"] <= out["pre_tax_cagr"]
    assert out["total_tax"] >= 0


def test_post_tax_cagr_empty():
    """Empty blotter must not crash."""
    out = st.post_tax_cagr(pd.DataFrame(columns=["exit_date"]))
    assert out["pre_tax_cagr"] == 0
    assert out["post_tax_cagr"] == 0


def test_post_tax_cagr_missing_pnl_cols():
    """Blotter without cost/proceeds returns error dict, not crash."""
    blot = pd.DataFrame({
        "ticker": ["A"], "entry_date": [pd.Timestamp("2024-01-01")],
        "exit_date": [pd.Timestamp("2024-02-01")],
    })
    out = st.post_tax_cagr(blot)
    assert "error" in out


# ─── reality_check_spa ──────────────────────────────────────────────────

def test_spa_shape():
    rng = np.random.default_rng(7)
    R = rng.normal(0.0001, 0.01, (300, 8))
    out = st.reality_check_spa(R, block_size=10, n_resamples=500)
    if "error" in out:
        return  # arch not installed in this env — OK
    assert 0.0 <= out["p_consistent"] <= 1.0
    assert out["n_strategies"] == 8
    assert out["n_obs"] == 300


def test_spa_too_short_returns_error():
    R = np.random.randn(10, 5)
    out = st.reality_check_spa(R, n_resamples=100)
    assert "error" in out


def test_spa_wrong_shape_returns_error():
    R = np.random.randn(100)  # 1D
    out = st.reality_check_spa(R, n_resamples=100)
    assert "error" in out


# ─── trial_count_from_registry ──────────────────────────────────────────

def test_trial_count_missing_registry(tmp_path):
    """Missing file returns 1 (conservative default)."""
    fake = tmp_path / "nope.jsonl"
    assert st.trial_count_from_registry(fake) == 1


def test_trial_count_deduplicates(tmp_path):
    p = tmp_path / "t.jsonl"
    import json as _j
    # Same (model, label, horizon, threshold, universe) twice + one different
    entries = [
        {"model": "lgbm", "label": "x", "horizon_d": 40, "threshold_q": 0.25, "universe": "us"},
        {"model": "lgbm", "label": "x", "horizon_d": 40, "threshold_q": 0.25, "universe": "us"},
        {"model": "lr",   "label": "x", "horizon_d": 40, "threshold_q": 0.25, "universe": "us"},
    ]
    p.write_text("\n".join(_j.dumps(e) for e in entries))
    assert st.trial_count_from_registry(p) == 2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
