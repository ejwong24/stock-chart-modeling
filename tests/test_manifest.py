"""Tests for manifest.py — reproducibility manifest writer."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import manifest as mf


def test_file_sha256_stable(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello world")
    h1 = mf._file_sha256(f)
    h2 = mf._file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_file_sha256_changes_on_content_change(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("aaa")
    h1 = mf._file_sha256(f)
    f.write_text("bbb")
    h2 = mf._file_sha256(f)
    assert h1 != h2


def test_dir_sha256_handles_empty(tmp_path):
    out = mf._dir_sha256(tmp_path, "*.txt")
    assert out["file_count"] == 0
    assert "combined_sha256" in out


def test_dir_sha256_picks_up_files(tmp_path):
    (tmp_path / "a.txt").write_text("foo")
    (tmp_path / "b.txt").write_text("bar")
    (tmp_path / "c.csv").write_text("not matched")
    out = mf._dir_sha256(tmp_path, "*.txt")
    assert out["file_count"] == 2


def test_env_packages_handles_missing():
    """Unknown package name should map to '(not installed)' string."""
    out = mf.env_packages(["pandas", "this_pkg_does_not_exist_xyzzy"])
    assert "pandas" in out
    assert out["this_pkg_does_not_exist_xyzzy"] == "(not installed)"


def test_write_manifest_creates_file(tmp_path):
    project = tmp_path
    (project / "config").mkdir()
    (project / "config" / "default.yaml").write_text("seeds:\n  numpy: 42\n")
    (project / "config" / "delisted_seed.txt").write_text("# none\n")
    (project / "data").mkdir()
    (project / "data" / "adjusted").mkdir()

    out_path = project / "manifest.json"
    cfg = {"seeds": {"numpy": 42}}
    m = mf.write_manifest(out_path, project, cfg)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert "timestamp_utc" in loaded
    assert loaded["seeds"]["numpy"] == 42
    assert "package_versions" in loaded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
