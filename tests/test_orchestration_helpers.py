"""Tests for helper functions inside orchestration scripts."""
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_run_pipeline_imports():
    m = _load_script("run_pipeline")
    assert hasattr(m, "main")


def test_resimulate_imports_and_load_price_lookup(tmp_path):
    m = _load_script("resimulate")
    # Write a couple of parquets
    for t in ("ABC", "DEF"):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                            "close": [10, 11, 12, 13, 14]})
        df.to_parquet(tmp_path / f"{t}.parquet")
    out = m._load_price_lookup(tmp_path, ["ABC", "DEF", "MISSING"])
    assert set(out.keys()) == {"ABC", "DEF"}
    assert pd.api.types.is_datetime64_any_dtype(out["ABC"]["date"])


def test_sma250_config_is_zero_costs():
    m = _load_script("run_sma250_test")
    cfg = m.make_sim_config()
    assert cfg.slippage_bps_each_side == 0.0
    assert cfg.commission_per_share == 0.0
    assert cfg.min_adv_usd == 0.0
    assert cfg.trailing_stop_pct == 0.0
    assert cfg.use_almgren_chriss_impact is False
    assert cfg.halt_risk_enabled is False
    assert cfg.no_duplicate_ticker is True


def test_sma250_filter_anchors_date_range():
    m = _load_script("run_sma250_test")
    # Build a synthetic labels DF and verify date-filter behavior
    dates = pd.date_range("2018-01-01", "2027-01-01", freq="W-FRI")
    labels_df = pd.DataFrame({
        "ticker": "X", "anchor_date": dates,
        "prefilter_ma250_15": 1, "anchor_idx": range(len(dates)),
        "anchor_close": 100.0, "adv20_usd": 5e7,
        "resolution_date_20d": dates + pd.Timedelta(days=30),
        "resolution_date_40d": dates + pd.Timedelta(days=58),
    })
    out_20 = m.filter_anchors(labels_df, horizon=20)
    # First entry should be >= 2019-01-04
    assert out_20["anchor_date"].min() >= pd.Timestamp("2019-01-04")
    # Last entry should be <= 2026-03-20 (per spec)
    assert out_20["anchor_date"].max() <= pd.Timestamp("2026-03-20")

    out_40 = m.filter_anchors(labels_df, horizon=40)
    assert out_40["anchor_date"].max() <= pd.Timestamp("2026-02-20")


def test_sma250_filter_anchors_drops_unresolvable():
    """Anchors whose resolution_date is past the final-exit date should be dropped."""
    m = _load_script("run_sma250_test")
    dates = pd.date_range("2026-02-01", "2026-05-01", freq="W-FRI")
    labels_df = pd.DataFrame({
        "ticker": "X", "anchor_date": dates,
        "prefilter_ma250_15": 1, "anchor_idx": range(len(dates)),
        "anchor_close": 100.0, "adv20_usd": 5e7,
        "resolution_date_40d": dates + pd.Timedelta(days=58),  # late ones go past 2026-04-20
    })
    out = m.filter_anchors(labels_df, horizon=40)
    # All retained anchors should have resolution <= 2026-04-20
    assert (out["resolution_date_40d"] <= pd.Timestamp("2026-04-20")).all()


def test_forward_pick_imports():
    m = _load_script("forward_pick")
    assert hasattr(m, "main")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
