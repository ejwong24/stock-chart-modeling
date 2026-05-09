"""View results from a pipeline run as a rendered table + key stats.

Usage:
    python scripts/view_results.py [out-tag]
    # e.g.
    python scripts/view_results.py smoke_100
    python scripts/view_results.py full
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "smoke_100"
    run = PROJECT / "reports" / tag
    if not run.exists():
        print(f"No such run: {run}")
        print("Available runs:", [p.name for p in (PROJECT / "reports").iterdir() if p.is_dir()])
        return

    print(f"=== Results for: {tag} ===\n")
    head = json.loads((run / "headline.json").read_text())

    print(f"Label:           {head['headline_label']}")
    print(f"Tickers loaded:  {head['n_tickers_loaded']}")
    print(f"Anchor rows:     {head['n_anchor_rows_after_prefilter']}")
    print(f"Folds run:       {head['folds_run']}")
    print(f"Wall-clock:      {head['wall_seconds']:.1f}s\n")

    rows = head["summaries"]
    df = pd.DataFrame(rows).sort_values("end_equity", ascending=False)
    df["end_equity"] = df["end_equity"].map(lambda x: f"${x:,.0f}")
    df["cagr"] = df["cagr"].map(lambda x: f"{x*100:+.2f}%")
    df["max_dd"] = df["max_dd"].map(lambda x: f"{x*100:.2f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:+.2f}")
    df["vs_random_pct"] = df["vs_random_pct"].map(lambda x: f"{x*100:.0f}th")
    print("All strategies, ranked by ending equity:")
    print(df.to_string(index=False))

    rb = head["random_baseline"]
    print(f"\nRandom baseline (n={rb['n_seeds']}):")
    print(f"  median CAGR:         {rb['median_cagr']*100:+.2f}%")
    print(f"  median end equity:   ${rb['median_end_equity']:,.0f}")
    print(f"  95th-pct end equity: ${rb['p95_end_equity']:,.0f}")
    print(f"  99th-pct end equity: ${rb['p99_end_equity']:,.0f}")

    dsr = head["deflated_sharpe"]
    print(f"\nDeflated Sharpe (best-of-tracks vs N_trials={dsr['n_trials']}):")
    print(f"  observed Sharpe:                 {dsr['observed_sharpe_annualized']:+.3f}")
    print(f"  deflated significance threshold: {dsr['deflated_sharpe_threshold_annualized']:+.3f}")
    print(f"  significance p-value:            {dsr['deflated_sharpe_p_value_significant']:.4f}")
    if dsr['deflated_sharpe_p_value_significant'] >= 0.95:
        print("  -> Sharpe is DEFLATED-SIGNIFICANT (>0.95).")
    else:
        print("  -> Sharpe is NOT deflated-significant after multiple-testing correction.")

    print("\nKey artifacts in this run directory:")
    for p in sorted(run.glob("*.parquet")):
        sz = p.stat().st_size / 1024
        print(f"  {p.name:<50} {sz:>8.1f} KB")
    for p in sorted(run.glob("*.npy")):
        sz = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<50} {sz:>8.1f} MB")
    for p in sorted(run.glob("*.csv")):
        sz = p.stat().st_size / 1024
        print(f"  {p.name:<50} {sz:>8.1f} KB")
    for p in sorted(run.glob("*.json")):
        sz = p.stat().st_size / 1024
        print(f"  {p.name:<50} {sz:>8.1f} KB")


if __name__ == "__main__":
    main()
