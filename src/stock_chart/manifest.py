"""Reproducibility manifest writer."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, subprocess, time
import importlib.metadata as imd


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_sha256(path: Path, glob: str) -> dict:
    files = sorted(path.glob(glob))
    digests = {p.name: _file_sha256(p) for p in files[:100]}  # first 100 for the manifest preview
    combined = hashlib.sha256()
    for p in files:
        combined.update(p.name.encode("utf-8"))
        combined.update(_file_sha256(p).encode("utf-8"))
    return {"file_count": len(files),
            "combined_sha256": combined.hexdigest(),
            "first_100_files_sha256": digests}


def env_packages(names: list[str]) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = imd.version(n)
        except imd.PackageNotFoundError:
            out[n] = "(not installed)"
    return out


def dinov2_commit() -> str:
    try:
        from .embed_dinov2 import commit_hash
        return commit_hash()
    except Exception:
        return "unknown"


def write_manifest(out_path: Path, project_root: Path, config: dict,
                   extra: dict | None = None) -> dict:
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": sys.platform,
        "config_sha256": _file_sha256(project_root / "config" / "default.yaml"),
        "delisted_seed_sha256": _file_sha256(project_root / "config" / "delisted_seed.txt"),
        "dataset_adjusted_dir": _dir_sha256(project_root / "data" / "adjusted", "*.parquet"),
        "dinov2_commit": dinov2_commit(),
        "package_versions": env_packages([
            "numpy", "pandas", "scikit-learn", "scipy", "lightgbm",
            "torch", "torchvision", "yfinance", "pillow", "pyarrow",
            "joblib", "matplotlib"
        ]),
        "seeds": config.get("seeds", {}),
    }
    if extra:
        manifest.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
