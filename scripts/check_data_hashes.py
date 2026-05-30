"""Write or verify the SHA-256 manifest of data/adjusted/*.parquet.

Reproducibility insurance against silent yfinance restatements (see
RESEARCH/pipeline_deepdives/47_roadmap.md item 3).

    python scripts/check_data_hashes.py --write     # snapshot current hashes
    python scripts/check_data_hashes.py --verify     # diff disk vs manifest (exit 1 on drift)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import manifest as mf


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="snapshot current hashes")
    g.add_argument("--verify", action="store_true", help="diff disk vs manifest")
    ap.add_argument("--out", default="data/data_hashes.json")
    args = ap.parse_args()

    adjusted = cfg.project_path("data", "adjusted")
    out = cfg.project_path(args.out)

    if args.write:
        p = mf.write_data_hashes(adjusted, out)
        print(f"wrote {p['file_count']} parquet hashes -> {out}")
        return 0

    r = mf.verify_data_hashes(adjusted, out)
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return 2
    print(f"ok={len(r['ok'])} changed={len(r['changed'])} "
          f"missing={len(r['missing'])} added={len(r['added'])}")
    if r["changed"]:
        print("  CHANGED (yfinance drift):", ", ".join(r["changed"][:20]))
    if r["missing"]:
        print("  MISSING:", ", ".join(r["missing"][:20]))
    # Drift = changed or missing. `added` (new tickers) is informational, not a failure.
    return 1 if (r["changed"] or r["missing"]) else 0


if __name__ == "__main__":
    sys.exit(main())
