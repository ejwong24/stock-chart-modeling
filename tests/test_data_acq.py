"""Tests for data_acq._normalize_one — yfinance row shape handling."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import data_acq as da


def test_normalize_basic():
    df = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [102.0, 103.0],
        "Low": [99.0, 100.0], "Close": [101.0, 102.0],
        "Volume": [1000.0, 1100.0],
    }, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"],
                                 tz="America/New_York"))
    out = da._normalize_one(df)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    # Timezone-naive after normalization
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_normalize_drops_nan_close():
    df = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [101, 102], "Low": [99, 100],
        "Close": [101.0, np.nan], "Volume": [1000, 1100],
    }, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
    out = da._normalize_one(df)
    assert len(out) == 1


def test_normalize_lowercase_columns():
    """yfinance sometimes returns lowercase column names."""
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.5], "volume": [1000.0],
    }, index=pd.DatetimeIndex(["2024-01-02"]))
    out = da._normalize_one(df)
    assert len(out) == 1
    assert "close" in out.columns


def test_normalize_all_nan_rows():
    df = pd.DataFrame({"Close": [np.nan]*5, "Volume": [np.nan]*5},
                       index=pd.date_range("2024-01-02", periods=5))
    out = da._normalize_one(df)
    assert len(out) == 0


def test_normalize_empty_dataframe():
    """Bug #6 regression: empty df must not crash."""
    out = da._normalize_one(pd.DataFrame())
    assert len(out) == 0
    assert "close" in out.columns


def test_normalize_none_input():
    """Bug #6 regression: None must not crash."""
    out = da._normalize_one(None)
    assert len(out) == 0


def test_normalize_missing_close_column():
    """Bug #6 regression: missing Close col must not crash."""
    df = pd.DataFrame({"foo": [1, 2, 3]},
                       index=pd.date_range("2024-01-02", periods=3))
    out = da._normalize_one(df)
    assert len(out) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
