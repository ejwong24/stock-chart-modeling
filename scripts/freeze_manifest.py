"""Freeze a reproducibility manifest for the project state."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart.manifest import write_manifest


def main():
    c = cfg.load()
    out = cfg.project_path("manifest.json")
    m = write_manifest(out, cfg.PROJECT_ROOT, c)
    print("Wrote", out)
    print({k: v for k, v in m.items() if k != "dataset_adjusted_dir"})
    print("dataset adjusted file count:", m["dataset_adjusted_dir"]["file_count"])


if __name__ == "__main__":
    main()
