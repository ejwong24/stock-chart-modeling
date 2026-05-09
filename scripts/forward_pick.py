"""Weekly forward paper-trade picker (per RESEARCH P3A11).

Run every Friday after close (system cron or GitHub Actions weekly cron).
Freezes a model trained on data through today; emits next-Monday's top-5
picks; persists to data/forward_picks/<date>.csv.

A companion script `settle_picks.py` runs daily and looks up realized
returns 40 trading days later, writing data/forward_realized/.

After 26 weeks you have 130 fully out-of-sample trades evaluated under
real execution. This is the only result that survives all methodological
objections.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import labels as lbl
from stock_chart import features as feat
from stock_chart import models as mdl


def load_universe_with_data(c: dict) -> list[str]:
    universe = pd.read_csv(cfg.project_path("data", "universe", "working_universe.csv"))
    adj = cfg.project_path("data", "adjusted")
    on_disk = [t for t in universe["ticker"].tolist()
                if (adj / f"{t}.parquet").exists()]
    return sorted(on_disk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="ret_40d_ge_25pct",
                     help="Label column to train against")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--top-k", type=int, default=5,
                     help="Number of picks per anchor (default 5)")
    ap.add_argument("--track", default="lgbm_engineered",
                     choices=["lgbm_engineered", "lgbm_image", "lr_baseline"])
    ap.add_argument("--out-dir", default="data/forward_picks")
    args = ap.parse_args()

    c = cfg.load()
    today = pd.Timestamp.today().normalize()
    out_path = cfg.project_path(args.out_dir) / f"{today.date().isoformat()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"freezing model on data through {today.date()}...")

    tickers = load_universe_with_data(c)
    print(f"  universe with parquet: {len(tickers)}")

    # Generate labels for all anchors up through today (training)
    horizons = c["labels"]["horizons"]
    thresholds = c["labels"]["thresholds"]
    labels_df = lbl.build_all(
        cfg.project_path("data", "adjusted"), tickers, horizons, thresholds,
        warmup_days=c["labels"]["warmup_days"],
        ma_window=c["prefilter"]["ma_window"],
        ma_extension=c["prefilter"]["ma_extension"],
        prefilter_only=c["prefilter"]["enabled"])
    labels_df = labels_df[labels_df["anchor_date"] <= today].reset_index(drop=True)
    print(f"  labels rows (training): {len(labels_df)}")

    # Engineered features
    spy_path = cfg.project_path("data", "adjusted", "SPY.parquet")
    feats = feat.compute_for_anchors(
        cfg.project_path("data", "adjusted"),
        labels_df[["ticker", "anchor_date", "anchor_idx"]],
        spy_path=spy_path)
    merged = feats.merge(labels_df, on=["ticker", "anchor_date"], how="inner")
    print(f"  merged: {len(merged)}")

    if args.label not in merged.columns:
        print(f"label {args.label} not in columns; pick one of "
              f"{[c for c in merged.columns if c.startswith('ret_')][:5]}")
        return 2

    # Train on all rows whose forward window resolves before today (no leakage)
    H = args.horizon
    res_col = f"resolution_date_{H}d"
    train = merged[merged[res_col] < today].reset_index(drop=True)
    if len(train) < 1000:
        print(f"  too few training rows ({len(train)})")
        return 2

    feat_cols = feat.FEATURE_COLS
    X_tr = train[feat_cols].to_numpy(dtype=np.float32)
    y_tr = train[args.label].to_numpy(dtype=np.int8)
    print(f"  training {args.track} on {len(train)} examples ...")
    seed = c["seeds"]["sklearn_random_state"]
    art = mdl.fit_lgbm(X_tr, y_tr, seed=seed, lgbm_kwargs=c["models"]["lgbm"])

    # Score the most recent anchor for each ticker (not yet resolved)
    most_recent = (merged.sort_values("anchor_date")
                          .groupby("ticker", as_index=False).tail(1))
    most_recent = most_recent[most_recent[res_col] >= today].reset_index(drop=True)
    print(f"  candidates for next pick: {len(most_recent)}")
    if len(most_recent) == 0:
        print("  no eligible candidates today")
        return 0

    X_pred = most_recent[feat_cols].to_numpy(dtype=np.float32)
    scores = mdl.predict_lgbm(art, X_pred)
    most_recent = most_recent.assign(score=scores).sort_values(
        "score", ascending=False)

    picks = most_recent.head(args.top_k)[
        ["ticker", "anchor_date", "anchor_close", "adv20_usd", "score"]
    ].copy()
    picks["pick_generated"] = today
    picks["horizon_days"] = H
    picks["track"] = args.track
    picks.to_csv(out_path, index=False)
    print(f"\nwrote {len(picks)} picks -> {out_path}")
    print(picks.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
