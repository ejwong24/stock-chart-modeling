"""Lint test — keeps the codebase ruff-clean."""
import subprocess
import sys
from pathlib import Path
import pytest


PROJECT = Path(__file__).resolve().parents[1]


def test_ruff_clean():
    """Codebase must pass `ruff check` with the configured rules in pyproject.toml."""
    try:
        r = subprocess.run(["ruff", "check", "src/", "scripts/", "web/", "tests/"],
                            cwd=str(PROJECT), capture_output=True, timeout=30)
    except FileNotFoundError:
        pytest.skip("ruff not installed")
    assert r.returncode == 0, \
        f"ruff failed:\n{r.stdout.decode()}\n{r.stderr.decode()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
