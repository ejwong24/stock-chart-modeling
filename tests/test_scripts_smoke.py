"""Smoke tests that every CLI script accepts --help and emits sane output."""
import sys
import subprocess
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Scripts that take CLI args via argparse — these should respond to --help
HELP_SCRIPTS = [
    "scripts/run_pipeline.py",
    "scripts/resimulate.py",
    "scripts/forward_pick.py",
    "scripts/run_sma250_test.py",
    "scripts/scrape_edgar_form25.py",
    "scripts/settle_picks.py",
    "scripts/check_data_hashes.py",
]

# Scripts that just run their main() — we just want them to import cleanly
IMPORT_ONLY_SCRIPTS = [
    "scripts/01_build_universe.py",
    "scripts/02_acquire_data.py",
    "scripts/build_pipeline_walkthrough_visuals.py",
    "scripts/build_story_visuals.py",
    "scripts/capture_ui_screenshots.py",
    "scripts/detect_inferred_delistings.py",
    "scripts/freeze_manifest.py",
    "scripts/view_results.py",
]


@pytest.mark.parametrize("script", HELP_SCRIPTS)
def test_script_help_exits_clean(script):
    """python script.py --help should exit 0 and print usage."""
    p = PROJECT / script
    assert p.exists(), f"missing {p}"
    # 60s timeout: cold imports of torch + lightgbm + sklearn on ARM64 4-core CPU
    # routinely take 8-12s wall-clock for the heaviest scripts (run_pipeline,
    # forward_pick). 15s wasn't enough headroom under contention.
    r = subprocess.run([PYTHON, str(p), "--help"], capture_output=True,
                         timeout=60)
    assert r.returncode == 0, f"{script} --help exited {r.returncode}: " \
                                 f"{r.stderr.decode()[:500]}"
    assert b"usage" in r.stdout.lower() or b"help" in r.stdout.lower()


@pytest.mark.parametrize("script", IMPORT_ONLY_SCRIPTS)
def test_script_compiles(script):
    """py_compile every script — no syntax errors."""
    import py_compile
    p = PROJECT / script
    assert p.exists(), f"missing {p}"
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"{script} failed to compile: {e}")


def test_view_results_handles_missing_run():
    """view_results.py on a missing run should exit cleanly (not crash)."""
    r = subprocess.run([PYTHON, str(PROJECT / "scripts" / "view_results.py"),
                          "no_such_run_xyzzy"], capture_output=True, timeout=10)
    assert r.returncode == 0
    assert b"No such run" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
