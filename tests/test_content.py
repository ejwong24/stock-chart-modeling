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
    """Each chapter's <a href='/story/...'> should point to a real chapter or subsection."""
    sdir = PROJECT / "web" / "templates" / "story"
    valid_chapters = {p.stem for p in sdir.glob("*.html")}
    valid_chapters.add("")  # /story (index)
    # Chapter-9 deep-dive subsections (validated by file presence)
    deepdive_dir = PROJECT / "RESEARCH" / "pipeline_deepdives"
    valid_deepdives = {f"09/{p.stem}" for p in deepdive_dir.glob("*.md")}
    for tpl in sdir.glob("*.html"):
        text = tpl.read_text()
        refs = re.findall(r'href="/story/([^"]+)"', text)
        for r in refs:
            assert r in valid_chapters or r in valid_deepdives, \
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


ALL_DEEPDIVES = [
    "01_dinov2_architecture", "02_pca_math", "03_engineered_features",
    "04_logistic_regression", "05_volume_processing", "06_simulator_loop",
    "07_ma250_prefilter", "08_walkforward_embargo", "09_almgren_chriss",
    "10_deflated_sharpe", "11_trailing_stop_interactions",
    "12_reproducibility_seeds",
    "13_data_acquisition", "14_universe_construction", "15_deployment_tailscale",
    "16_trial_registry", "17_disaster_recovery", "18_weekly_anchors",
    "19_label_grid", "20_simple_baselines", "21_validation_modes",
    "22_block_bootstrap_params", "23_multiple_comparison_landscape",
    "24_effective_sample_size", "25_frictions_beyond_impact",
    "26_capacity_ceiling",
    "27_sharpe_ratio", "28_cagr_drawdown_calmar", "29_fixed_log_y_axis",
    "30_isotonic_calibration", "31_lightgbm_internals", "32_position_size_formula",
    "33_adv20_metric", "34_blotter_equity_summary", "35_log_vs_simple_returns",
    "36_allocation_budget", "37_trading_calendar", "38_spy_beta",
    "39_hardening_story", "40_lookahead_exit_bug", "41_row_alignment",
    "42_beta_zero_bug", "43_label_corruption", "44_numerical_stability",
    "45_testing_philosophy", "46_forward_pick_harness", "47_roadmap",
    "48_degenerate_folds", "49_phantom_positions", "50_atomic_writes_integrity",
    "51_audit_methodology",
    "52_paired_gap_test", "53_settle_loop", "54_data_hashes", "55_the_verdict",
    "56_cost_stack", "57_survivorship_accounting", "58_regime_dependence",
    "59_image_track_postmortem", "60_data_flow_contract", "61_portfolio_construction",
    "62_anatomy_of_a_trade", "63_config_surface", "64_glossary",
]


def test_pipeline_deepdives_all_present():
    """Every chapter-9 deep-dive markdown file exists with non-trivial content."""
    d = PROJECT / "RESEARCH" / "pipeline_deepdives"
    for slug in ALL_DEEPDIVES:
        p = d / f"{slug}.md"
        assert p.exists(), f"missing deep-dive markdown: {slug}.md"
        text = p.read_text()
        assert len(text) > 500, f"{slug}.md too short ({len(text)} chars)"
        assert text.startswith("# "), f"{slug}.md must start with H1"


def test_chapter_9_links_to_all_deepdives():
    """Chapter 9 main page should link to every deep-dive subsection."""
    p = PROJECT / "web" / "templates" / "story" / "09_pipeline_walkthrough.html"
    text = p.read_text()
    for slug in ALL_DEEPDIVES:
        assert f"/story/09/{slug}" in text, f"chapter 9 missing link to /story/09/{slug}"


def test_gitignore_excludes_large_artifacts():
    p = PROJECT / ".gitignore"
    text = p.read_text()
    for must_exclude in ["data/adjusted/", "*.parquet", "*.npy", ".venv/",
                          "dinov2_embeddings.npy"]:
        assert must_exclude in text, f".gitignore missing: {must_exclude}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
