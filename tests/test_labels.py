"""Tests for labels.py — weekly-anchor + multi-horizon labels."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import labels as lbl


def _make_df(seed=0, n=600, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, n))
    vols = rng.lognormal(14, 0.4, n)
    return pd.DataFrame({"date": dates, "close": closes, "volume": vols})


def test_label_one_basic():
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[20, 40], thresholds=[0.10, 0.25])
    assert len(out) > 0
    assert "anchor_date" in out.columns
    assert "anchor_close" in out.columns
    assert "fwd_ret_40d" in out.columns
    assert "ret_40d_ge_25pct" in out.columns
    assert "resolution_date_40d" in out.columns
    assert "prefilter_ma250_15" in out.columns


def test_label_one_resolution_date_is_correct_h_days_ahead():
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    # For each row, resolution_date_40d should be exactly 40 trading days after anchor_date
    dates_arr = df["date"].to_numpy()
    idx_map = {d: i for i, d in enumerate(dates_arr)}
    for _, r in out.iterrows():
        anchor_i = idx_map[np.datetime64(r["anchor_date"])]
        res_i = idx_map[np.datetime64(r["resolution_date_40d"])]
        assert res_i - anchor_i == 40, \
            f"anchor {r['anchor_date']} → res {r['resolution_date_40d']}: diff = {res_i - anchor_i}"


def test_label_one_binary_label_consistency():
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[20], thresholds=[0.10, 0.25])
    # ret_20d_ge_25pct should be 1 iff fwd_ret_20d >= 0.25
    for _, r in out.iterrows():
        expected = int(r["fwd_ret_20d"] >= 0.25)
        assert int(r["ret_20d_ge_25pct"]) == expected


def test_label_one_path_dependent_labels_exist():
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[20, 40], thresholds=[0.10, 0.25])
    # 5 new label families per horizon (from problem-5 implementation)
    expected_extras = [
        "tb_approx_20d_T5_S2", "mfe_mae_ratio_20d", "sortino_label_20d",
        "upside_dominance_20d", "clean_run_20d",
        "tb_approx_40d_T10_S5", "mfe_mae_ratio_40d",
    ]
    for c in expected_extras:
        assert c in out.columns, f"missing path-dep label: {c}"


def test_label_one_short_series_returns_empty():
    """A ticker with too few rows to have any valid anchors should return empty."""
    df = _make_df(n=200)  # < 252 warmup + 40 forward
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    assert len(out) == 0


def test_label_one_weekly_anchor_rule():
    """Anchors should be one-per-week (one per ISO week with ≥3 trading days)."""
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    # Same ISO week should never appear twice
    iso = pd.to_datetime(out["anchor_date"]).dt.isocalendar()
    keys = iso["year"] * 100 + iso["week"]
    assert keys.is_unique, "weekly anchor rule violated — duplicates"


def test_label_one_no_lookahead_in_anchor_close():
    """anchor_close should match the actual close on anchor_date in source df."""
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    df_indexed = df.set_index("date")
    for _, r in out.head(20).iterrows():
        actual_close = df_indexed.loc[r["anchor_date"], "close"]
        assert abs(r["anchor_close"] - actual_close) < 1e-9


def test_label_one_mfe_mae_consistency():
    """mfe >= 0 and mae <= 0 (by definition of max favorable / adverse excursion)."""
    df = _make_df()
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    # Quick sanity check on first 50 rows
    assert (out["fwd_mfe_40d"] >= -1e-12).all(), \
        f"some MFE is negative: min = {out['fwd_mfe_40d'].min()}"
    assert (out["fwd_mae_40d"] <= 1e-12).all(), \
        f"some MAE is positive: max = {out['fwd_mae_40d'].max()}"
    # mfe always >= log_ret (since path includes the endpoint)
    # mae always <= log_ret


def test_build_all_skips_missing_parquets(tmp_path):
    """build_all() should skip tickers without a parquet on disk."""
    df = _make_df()
    df.to_parquet(tmp_path / "REAL.parquet", index=False)
    out = lbl.build_all(tmp_path, ["REAL", "GHOST"],
                          horizons=[40], thresholds=[0.25],
                          prefilter_only=False)
    assert "REAL" in out["ticker"].unique()
    assert "GHOST" not in out["ticker"].unique()


def test_label_one_handles_zero_close():
    """Defensive: a stock that drops to 0 should not crash label generation."""
    df = _make_df()
    # Force a zero close near the middle (corporate action artifact)
    df.loc[300, "close"] = 0.0
    out = lbl.label_one(df, "TEST", horizons=[40], thresholds=[0.25])
    # Should not crash; some labels may be NaN/-Inf but not crash
    assert isinstance(out, pd.DataFrame)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
