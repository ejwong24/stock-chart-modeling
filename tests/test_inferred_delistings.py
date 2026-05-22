"""Tests for scripts/detect_inferred_delistings.py classifier."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "detect_inferred_delistings",
        PROJECT / "scripts" / "detect_inferred_delistings.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_empty_df_returns_none():
    m = _load_module()
    df = pd.DataFrame(columns=["date", "close", "volume"])
    assert m.classify(df) is None


def test_single_row_returns_none():
    m = _load_module()
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")], "close": [10.0], "volume": [1000.0],
    })
    assert m.classify(df) is None


def test_active_clean_stock_returns_none():
    """Stock whose last date == TERMINAL with no distress signals."""
    m = _load_module()
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    # extend through TERMINAL
    df = pd.DataFrame({
        "date": list(dates) + [m.TERMINAL],
        "close": [50.0] * 31, "volume": [1e6] * 31,
    })
    assert m.classify(df) is None


def test_stale_terminal_date_flagged():
    """Last date well before TERMINAL → early_terminal flag."""
    m = _load_module()
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    df = pd.DataFrame({"date": dates, "close": [50.0]*30, "volume": [1e6]*30})
    out = m.classify(df)
    assert out is not None
    assert "early_terminal" in out["reasons"]


def test_zero_volume_tail_flagged():
    m = _load_module()
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "close": [50.0] * 30,
        "volume": [1e6]*15 + [0.0]*15,
    })
    out = m.classify(df)
    assert out is not None
    assert "zerovol15" in out["reasons"]


def test_bagel_drop_flagged():
    """20-day max drawdown >= 90% within last 20 days → bagel_drop."""
    m = _load_module()
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    # Last 20 days: peak at 50, then drops to 2 → 96% drawdown WITHIN the window
    closes = [50.0] * 10 + [50.0, 45.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0,
                              2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    df = pd.DataFrame({"date": dates, "close": closes, "volume": [1e6] * 30})
    out = m.classify(df)
    assert out is not None
    assert "bagel_drop" in out["reasons"], f"got reasons: {out['reasons']}"


def test_zero_close_flagged():
    m = _load_module()
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "date": dates, "close": [10.0]*20 + [0.0]*10, "volume": [1e6]*30,
    })
    out = m.classify(df)
    assert out is not None
    assert "zero_close" in out["reasons"]


def test_subdime_close_flagged():
    """Close <= 0.10 → subdime."""
    m = _load_module()
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "date": dates, "close": [5.0]*25 + [0.05]*5, "volume": [1e6]*30,
    })
    out = m.classify(df)
    assert out is not None
    assert "subdime_close" in out["reasons"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
