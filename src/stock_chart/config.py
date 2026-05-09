from __future__ import annotations
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
