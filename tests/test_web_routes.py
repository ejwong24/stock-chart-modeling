"""Smoke test every FastAPI route returns 200 (or expected non-200)."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from web.app import app

client = TestClient(app)


GET_200 = [
    "/",
    "/runs",
    "/runs/new",
    "/data",
    "/config",
    "/flaws",
    "/readme",
    "/research",
    "/glossary",
    "/story",
    "/health",
]
GET_200.extend(f"/story/{c}" for c in [
    "01_claim", "02_audit", "03_reproduction", "04_costs",
    "05_falsification", "06_statistics", "07_sma250", "08_bottom_line",
    "09_pipeline_walkthrough",
])
GET_200.extend(f"/research/{s}" for s in [
    "01_survivorship_bias", "02_encoder_bakeoff", "03_lockbox_protocol",
    "04_costs_capacity", "05_path_dependent_exits",
])

# Chapter 9 pipeline deep-dive subsections (rounds 1 + 2)
GET_200.extend(f"/story/09/{s}" for s in [
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
])

# Run-detail routes — only test if run exists
PROJECT = Path(__file__).resolve().parents[1]
EXISTING_RUNS = [p.name for p in (PROJECT / "reports").iterdir()
                  if p.is_dir() and (p / "headline.json").exists()]
for tag in EXISTING_RUNS:
    GET_200.append(f"/runs/{tag}")
    GET_200.append(f"/runs/{tag}/report-card")


@pytest.mark.parametrize("path", GET_200)
def test_get_returns_200(path):
    r = client.get(path)
    assert r.status_code == 200, f"GET {path} returned {r.status_code}"


def test_invalid_run_tag_returns_404():
    r = client.get("/runs/this_run_does_not_exist_at_all")
    assert r.status_code == 404


def test_invalid_research_slug_returns_404():
    r = client.get("/research/not_a_real_slug")
    assert r.status_code == 404


def test_invalid_story_chapter_returns_404():
    r = client.get("/story/not_a_chapter")
    assert r.status_code == 404


def test_api_equity_returns_json():
    """If we have any run, the equity API should return JSON with date+equity."""
    if not EXISTING_RUNS:
        pytest.skip("no runs to test")
    tag = EXISTING_RUNS[0]
    # find a track from the run
    import json
    head = json.loads((PROJECT / "reports" / tag / "headline.json").read_text())
    if not head.get("summaries"):
        pytest.skip("run has no summaries")
    track = head["summaries"][0]["track"]
    r = client.get(f"/api/runs/{tag}/equity/{track}")
    assert r.status_code == 200
    data = r.json()
    assert "date" in data and "equity" in data
    assert len(data["date"]) == len(data["equity"])


def test_api_blotter_returns_json_or_empty():
    if not EXISTING_RUNS:
        pytest.skip("no runs to test")
    tag = EXISTING_RUNS[0]
    r = client.get(f"/api/runs/{tag}/blotter/lgbm_engineered")
    assert r.status_code == 200
    data = r.json()
    # Either a list of trade dicts or empty
    assert isinstance(data, list)


def test_static_chart_files_served():
    """All story PNGs that templates reference should be reachable."""
    static_dir = PROJECT / "web" / "static" / "story"
    pngs = list(static_dir.glob("*.png"))
    assert len(pngs) > 0, "no story PNGs found"
    # spot check the first
    sample = pngs[0]
    r = client.get(f"/static/story/{sample.name}")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("image/")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
