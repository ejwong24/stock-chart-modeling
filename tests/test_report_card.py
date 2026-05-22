"""Tests for report_card.py — honest report card + linter."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import report_card as rc


# ─── lint() ──────────────────────────────────────────────────────────

def test_lint_passes_clean_report():
    md = """# Honest Report Card — foo

**BLUF — Would I trade this with my own money?** N

Pre-registration done. Headline metrics with CI. Baseline gap measured.
Post-cost CAGR computed. Hidden risks documented.
"""
    errors = rc.lint(md)
    assert errors == [], f"unexpected lint errors: {errors}"


def test_lint_rejects_best_of_K():
    """The forbidden phrase 'best of N' should be flagged."""
    md = """BLUF — N
Pre-registration: yes.
Headline metrics: best of 72 configs achieved Sharpe 0.5.
Baseline gap, Post-cost CAGR, Hidden risks: all there.
"""
    errors = rc.lint(md)
    assert any("best of" in e for e in errors)


def test_lint_rejects_100th_percentile():
    md = """BLUF, Pre-registration, Headline metrics, Baseline gap, Post-cost CAGR, Hidden risks all here.
The model ended in the 100th percentile vs random.
"""
    errors = rc.lint(md)
    assert any("100th percentile" in e for e in errors)


def test_lint_rejects_outperformed_random():
    md = """BLUF, Pre-registration, Headline metrics, Baseline gap, Post-cost CAGR, Hidden risks all here.
The model outperformed random baselines materially.
"""
    errors = rc.lint(md)
    assert any("outperformed random" in e for e in errors)


def test_lint_flags_missing_sections():
    md = "Just a BLUF and some text."
    errors = rc.lint(md)
    # Should flag at least 5 missing sections
    assert sum("missing required section" in e for e in errors) >= 5


# ─── render() ─────────────────────────────────────────────────────────

def test_render_with_no_values_has_TODOs():
    md = rc.render({})
    # Every TODO marker means a field that needs filling — that's the point
    assert "TODO" in md
    # Mandatory sections still present
    for section in rc.REQUIRED_SECTIONS:
        assert section in md, f"{section} missing from default render"


def test_render_substitutes_values():
    md = rc.render({"tag": "test_run", "bluf_y_n": "N", "cagr_point": "+5.12%"})
    assert "test_run" in md
    assert "+5.12%" in md
    # Filled fields no longer contain TODO at their slot (but other TODOs may remain)


# ─── auto_fill_from_run() ────────────────────────────────────────────

def test_auto_fill_returns_empty_for_empty_summaries(tmp_path):
    """A run whose headline.json has no summaries should return empty dict."""
    import json
    rd = tmp_path / "x"
    rd.mkdir()
    (rd / "headline.json").write_text(json.dumps({"summaries": []}))
    out = rc.auto_fill_from_run(rd, tmp_path)
    assert out == {}


def test_auto_fill_on_real_full_run():
    """Auto-fill should work on our real reports/full directory."""
    project = Path(__file__).resolve().parents[1]
    rd = project / "reports" / "full"
    if not (rd / "headline.json").exists():
        pytest.skip("reports/full not available")
    out = rc.auto_fill_from_run(rd, project)
    assert "tag" in out and out["tag"] == "full"
    assert "n_trials" in out
    assert "cagr_point" in out


def test_full_pipeline_render_lint_clean():
    """Auto-filling from a real run should produce a TODO-aware report."""
    project = Path(__file__).resolve().parents[1]
    rd = project / "reports" / "full"
    if not (rd / "headline.json").exists():
        pytest.skip("reports/full not available")
    values = rc.auto_fill_from_run(rd, project)
    md = rc.render(values)
    errors = rc.lint(md)
    # Forbidden phrases should not appear in the auto-filled doc
    forbidden = [e for e in errors if "forbidden phrase" in e]
    assert len(forbidden) == 0, f"auto-filled report has forbidden: {forbidden}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
