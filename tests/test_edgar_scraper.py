"""Tests for scripts/scrape_edgar_form25.py helpers."""
import importlib.util
import sys
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scrape_edgar_form25",
        PROJECT / "scripts" / "scrape_edgar_form25.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extract_basic():
    m = _load_module()
    got = m.extract_tickers_from_display(
        "SIFY TECHNOLOGIES LTD  (SIFY)  (CIK 0001094324)")
    assert got == ["SIFY"]


def test_extract_dot_class():
    m = _load_module()
    assert m.extract_tickers_from_display(
        "Berkshire Hathaway  (BRK.A)  (CIK 0001067983)") == ["BRK.A"]


def test_extract_no_ticker():
    m = _load_module()
    assert m.extract_tickers_from_display(
        "M.D.C. HOLDINGS, INC.  (CIK 0000773141)") == []


def test_extract_empty_or_none():
    m = _load_module()
    assert m.extract_tickers_from_display("") == []
    assert m.extract_tickers_from_display(None) == []


def test_extract_lowercase_not_matched():
    """Tickers are uppercase; a parenthesised lowercase token shouldn't match."""
    m = _load_module()
    assert m.extract_tickers_from_display(
        "Lowercase corp  (foo)  (CIK 0000123)") == []


def test_extract_multi_paren_ignores_extras():
    m = _load_module()
    assert m.extract_tickers_from_display(
        "Multi-paren issuer  (XYZ)  (CIK 999)  (extra)") == ["XYZ"]


def test_extract_spac_multi_class():
    """Bug #4 regression: SPAC display names like (CNDA, CNDA-UN, CNDA-WT)
    where one token is 7 chars long must all be captured."""
    m = _load_module()
    got = m.extract_tickers_from_display(
        "Concord Acquisition Corp II  (CNDA, CNDA-UN, CNDA-WT)  (CIK 0001851959)")
    assert got == ["CNDA", "CNDA-UN", "CNDA-WT"], f"got {got}"


def test_extract_long_warrant_ticker():
    """Another 8-char ticker (e.g. ABCDE-WT) should be captured."""
    m = _load_module()
    got = m.extract_tickers_from_display(
        "Acme Spac  (ABCDE-WT)  (CIK 0000999)")
    assert got == ["ABCDE-WT"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
