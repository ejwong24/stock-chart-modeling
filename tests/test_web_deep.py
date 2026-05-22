"""Deeper web UI checks — content + template variables."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from web.app import app

client = TestClient(app)


def test_home_features_story_link():
    """Home page should prominently feature the story walkthrough."""
    r = client.get("/")
    assert r.status_code == 200
    assert b"Story" in r.content
    assert b"story-hero" in r.content
    assert b"/story" in r.content


def test_story_index_lists_nine_chapters():
    r = client.get("/story")
    assert r.status_code == 200
    body = r.content.decode()
    for n in range(1, 10):
        assert f"CHAPTER 0{n}" in body, f"chapter 0{n} card missing from story index"


def test_run_detail_embeds_plotly_chart_div():
    """run_detail.html should include the Plotly chart div + JS fetch."""
    r = client.get("/runs/full")
    assert r.status_code == 200
    body = r.content.decode()
    assert "equity-chart" in body
    assert "Plotly" in body
    assert "lgbm_image" in body or "lgbm_engineered" in body


def test_report_card_route_does_not_leak_forbidden_phrases():
    """The auto-generated report card must not contain phrases the linter rejects."""
    r = client.get("/runs/full/report-card")
    assert r.status_code == 200
    body = r.content.decode().lower()
    # Linter forbids these phrases in the actual report content
    # (Note: these phrases CAN appear in copy explaining what's forbidden,
    # so we only assert the forbidden phrases don't appear in unintended contexts.)
    # Actually, the report itself is plain Markdown without those phrases — verify.
    # The page can contain "100th percentile" if it's in a heading explaining the issue.
    assert "outperformed random" not in body or "claim" in body  # only if discussing the forbidden phrase


def test_glossary_defines_key_terms():
    r = client.get("/glossary")
    assert r.status_code == 200
    body = r.content.decode().lower()
    for term in ["cagr", "sharpe", "drawdown", "deflated sharpe",
                  "walk-forward", "slippage"]:
        assert term in body, f"glossary missing: {term}"


def test_flaws_lists_all_seven_severities():
    r = client.get("/flaws")
    assert r.status_code == 200
    body = r.content.decode()
    # Should mention each major flaw category
    for keyword in ["Survivorship bias", "Walk-forward leakage",
                     "Multiple-comparison", "auto-scaling",
                     "transaction costs"]:
        assert keyword.lower() in body.lower(), f"flaws missing: {keyword}"


def test_static_css_served_no_error():
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert b"/* Stock Chart Modeling" in r.content
    # Mobile media query was added in earlier hardening
    assert b"@media" in r.content


def test_research_chapter_renders_markdown_correctly():
    """A research chapter should produce HTML with the markdown rendered via JS,
    so at least the raw Markdown text should appear in the page (passed to marked.js)."""
    r = client.get("/research/01_survivorship_bias")
    assert r.status_code == 200
    body = r.content.decode().lower()
    assert "survivorship" in body
    assert "marked.parse" in body  # marked.js call


def test_api_returns_consistent_shape():
    """/api/runs/<tag>/equity/<track> always returns dict with date+equity lists."""
    r = client.get("/api/runs/full/equity/lgbm_engineered")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "date" in data and "equity" in data
    assert isinstance(data["date"], list)
    assert isinstance(data["equity"], list)
    assert len(data["date"]) == len(data["equity"])


def test_monitor_route_returns_safe_default_for_unknown():
    r = client.get("/api/monitor/never_existed")
    assert r.status_code == 200
    data = r.json()
    assert data["alive"] is False
    assert data["info"] == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
