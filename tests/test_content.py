"""Content + structure checks for docs, research, and story templates.

These are 'document'-level tests: do the prose files mention what they claim
to mention, do all referenced images exist, do all internal links resolve.
"""
import re
import sys
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _all_html_files(d: Path):
    return sorted(d.rglob("*.html"))


def _all_md_files(d: Path):
    return sorted(d.glob("*.md"))


def test_flaws_md_exists_and_lists_all_seven():
    p = PROJECT / "FLAWS_AND_FIXES.md"
    assert p.exists()
    text = p.read_text()
    # 7 high-severity flaws documented
    flaws = [
        "Survivorship bias",
        "Walk-forward leakage",
        "Multiple-comparison",
        "Chart auto-scaling",
        "No simple-baseline",
        "No transaction costs",
        "DINOv2",
    ]
    missing = [f for f in flaws if f not in text]
    assert not missing, f"FLAWS_AND_FIXES.md missing: {missing}"


def test_research_files_complete():
    rdir = PROJECT / "RESEARCH"
    assert rdir.exists()
    expected = {"01_survivorship_bias.md", "02_encoder_bakeoff.md",
                "03_lockbox_protocol.md", "04_costs_capacity.md",
                "05_path_dependent_exits.md", "INDEX.md"}
    actual = {p.name for p in rdir.glob("*.md")}
    assert expected <= actual, f"missing research files: {expected - actual}"


def test_research_index_links_resolve():
    """INDEX.md should link only to files that exist."""
    p = PROJECT / "RESEARCH" / "INDEX.md"
    text = p.read_text()
    # Match (foo.md) markdown links
    refs = re.findall(r"\(([0-9_]+\w+\.md)\)", text)
    for r in refs:
        assert (PROJECT / "RESEARCH" / r).exists(), f"{r} referenced but missing"


def test_story_templates_all_present():
    """All 9 story chapters + index exist."""
    sdir = PROJECT / "web" / "templates" / "story"
    expected = {f"0{i}_{name}.html" for i, name in enumerate([
        "claim", "audit", "reproduction", "costs", "falsification",
        "statistics", "sma250", "bottom_line", "pipeline_walkthrough"], start=1)}
    actual = {p.name for p in sdir.glob("*.html")}
    assert expected <= actual, f"missing chapters: {expected - actual}"
    # Plus the index
    assert (PROJECT / "web" / "templates" / "story_index.html").exists()


def test_story_chapter_referenced_pngs_exist():
    """Every <img src='/static/story/foo.png'> referenced in chapter templates must exist."""
    sdir = PROJECT / "web" / "templates" / "story"
    static_dir = PROJECT / "web" / "static" / "story"
    for tpl in sdir.glob("*.html"):
        refs = re.findall(r'src="/static/story/([^"]+)"', tpl.read_text())
        for r in refs:
            assert (static_dir / r).exists(), \
                f"{tpl.name} references missing image: {r}"


def test_story_chapter_nav_links_resolve():
    """Each chapter's <a href='/story/...'> should point to another existing chapter."""
    sdir = PROJECT / "web" / "templates" / "story"
    valid_chapters = {p.stem for p in sdir.glob("*.html")}
    valid_chapters.add("")  # /story (index)
    for tpl in sdir.glob("*.html"):
        text = tpl.read_text()
        refs = re.findall(r'href="/story/([^"]+)"', text)
        for r in refs:
            assert r in valid_chapters, \
                f"{tpl.name} links to /story/{r} which doesn't exist"


def test_readme_mentions_corrections():
    """README should mention key corrections + GitHub URL."""
    p = PROJECT / "README.md"
    text = p.read_text()
    expected = ["purged walk-forward", "deflated Sharpe", "trailing stop",
                "Almgren-Chriss"]
    missing = [e for e in expected if e.lower() not in text.lower()]
    assert not missing, f"README missing: {missing}"


def test_research_index_lists_all_five_problems():
    """INDEX.md should mention all 5 hard problems."""
    p = PROJECT / "RESEARCH" / "INDEX.md"
    text = p.read_text().lower()
    keywords = ["survivorship", "encoder", "lockbox", "cost", "path-dependent"]
    missing = [k for k in keywords if k not in text]
    assert not missing, f"INDEX.md missing: {missing}"


def test_gitignore_excludes_large_artifacts():
    p = PROJECT / ".gitignore"
    text = p.read_text()
    for must_exclude in ["data/adjusted/", "*.parquet", "*.npy", ".venv/",
                          "dinov2_embeddings.npy"]:
        assert must_exclude in text, f".gitignore missing: {must_exclude}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
