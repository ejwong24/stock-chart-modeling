"""Reproducibility manifest writer."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import time
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


def write_data_hashes(adjusted_dir: Path, out_path: Path, glob: str = "*.parquet") -> dict:
    """Write a SHA-256 manifest of every parquet under adjusted_dir.

    Reproducibility insurance: the pipeline's bit-identical guarantee only holds
    if data/adjusted/*.parquet is byte-identical, but yfinance silently restates
    history (split/dividend corrections, late adjustments). Committing this
    manifest lets a later restore detect drift instead of silently producing
    different results. Written atomically (temp + os.replace)."""
    files = sorted(adjusted_dir.glob(glob))
    hashes = {p.name: _file_sha256(p) for p in files}
    payload = {"file_count": len(files), "glob": glob, "hashes": hashes}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _os.replace(tmp, out_path)
    return payload


def verify_data_hashes(adjusted_dir: Path, hashes_path: Path,
                       glob: str = "*.parquet") -> dict:
    """Diff current parquet hashes against a committed manifest.

    Returns {changed, missing, added, ok} lists of filenames. `changed` = file
    present in both but hash differs (yfinance drift); `missing` = in manifest
    but not on disk; `added` = on disk but not in manifest. An empty changed+
    missing means the data tree reproduces the manifest exactly."""
    if not hashes_path.exists():
        return {"error": f"no manifest at {hashes_path}",
                "changed": [], "missing": [], "added": [], "ok": []}
    manifest = json.loads(hashes_path.read_text()).get("hashes", {})
    current = {p.name: _file_sha256(p) for p in sorted(adjusted_dir.glob(glob))}
    changed, missing, added, ok = [], [], [], []
    for name, h in manifest.items():
        if name not in current:
            missing.append(name)
        elif current[name] != h:
            changed.append(name)
        else:
            ok.append(name)
    for name in current:
        if name not in manifest:
            added.append(name)
    return {"changed": sorted(changed), "missing": sorted(missing),
            "added": sorted(added), "ok": sorted(ok)}


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
