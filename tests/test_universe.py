"""Tests for universe.py — universe construction without HTTP."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import universe as u


def test_build_overlap_active_first_wins():
    """When same ticker is in active + delisted, active wins (drop_duplicates keep='first')."""
    active = pd.DataFrame({"ticker": ["A", "B"], "active": [True, True]})
    delisted = pd.DataFrame({"ticker": ["B", "C"], "active": [False, False]})
    out = u.build(active, delisted)
    rec = {r["ticker"]: r["active"] for r in out.to_dict("records")}
    assert rec == {"A": True, "B": True, "C": False}


def test_build_empty_active():
    active = pd.DataFrame(columns=["ticker", "active"])
    delisted = pd.DataFrame({"ticker": ["X"], "active": [False]})
    out = u.build(active, delisted)
    assert len(out) == 1
    assert out["ticker"].iloc[0] == "X"


def test_build_empty_delisted():
    active = pd.DataFrame({"ticker": ["A", "B"], "active": [True, True]})
    out = u.build(active, pd.DataFrame(columns=["ticker", "active"]))
    assert len(out) == 2


def test_build_both_empty():
    out = u.build(pd.DataFrame(columns=["ticker", "active"]),
                    pd.DataFrame(columns=["ticker", "active"]))
    assert len(out) == 0


def test_build_sorted_alphabetically():
    """Output should be sorted by ticker."""
    active = pd.DataFrame({"ticker": ["Z", "A", "M"], "active": [True]*3})
    out = u.build(active, pd.DataFrame(columns=["ticker", "active"]))
    assert out["ticker"].tolist() == ["A", "M", "Z"]


def test_load_delisted_seed_missing_file(tmp_path):
    out = u.load_delisted_seed(tmp_path / "nope.txt")
    assert len(out) == 0
    assert list(out.columns) == ["ticker", "active"]


def test_load_delisted_seed_with_comments_and_blanks(tmp_path):
    p = tmp_path / "seed.txt"
    p.write_text("# comment\nAAA\n\n# another\nBBB\n  CCC  \n")
    out = u.load_delisted_seed(p)
    assert out["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    # All seeded tickers should be active=False (delisted)
    assert (~out["active"]).all()


def test_load_delisted_seed_uppercase_normalization(tmp_path):
    """Mixed-case input should be uppercased."""
    p = tmp_path / "seed.txt"
    p.write_text("foo\nBar\nBAZ\n")
    out = u.load_delisted_seed(p)
    assert set(out["ticker"]) == {"FOO", "BAR", "BAZ"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
