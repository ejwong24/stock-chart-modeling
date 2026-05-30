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
import argparse
import sys
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


def latest_candidates(adjusted_dir: Path, tickers: list[str], c: dict,
                      today: pd.Timestamp) -> pd.DataFrame:
    """Most-recent TRADABLE weekly anchor per ticker (the bar we'd buy now).

    This is deliberately NOT derived from labels.build_all: label_one trims the
    trailing ~H bars because a label needs a resolved forward window. The whole
    point of a forward pick is the opposite — the freshest anchor whose forward
    window has NOT resolved. So we generate candidates directly: the latest bar
    at/before `today` with enough history for features (>= warmup) that passes
    the MA250 prefilter. anchor_idx is the position in the FULL sorted parquet,
    matching how features.compute_for_anchors indexes the price series.
    """
    warmup = c["labels"]["warmup_days"]
    ma_window = c["prefilter"]["ma_window"]
    ma_ext = c["prefilter"]["ma_extension"]
    prefilter = c["prefilter"]["enabled"]
    rows = []
    for t in tickers:
        p = adjusted_dir / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        elig = np.flatnonzero(df["date"].to_numpy() <= np.datetime64(today))
        if len(elig) == 0:
            continue
        ti = int(elig[-1])                      # latest bar at/before today
        if ti < warmup - 1 or ti < ma_window - 1:
            continue                             # not enough history for features/MA
        closes = df["close"].to_numpy(dtype=np.float64)
        if not np.isfinite(closes[ti]) or closes[ti] <= 0:
            continue
        ma = closes[ti - ma_window + 1: ti + 1].mean()
        if not np.isfinite(ma) or ma <= 0:
            continue
        if prefilter and (closes[ti] / ma) <= ma_ext:
            continue
        dollar = closes * df["volume"].to_numpy(dtype=np.float64)
        adv = float(np.nanmean(dollar[max(0, ti - 19): ti + 1]))
        rows.append({"ticker": t, "anchor_date": df["date"].iloc[ti],
                     "anchor_idx": ti, "anchor_close": float(closes[ti]),
                     "adv20_usd": adv, "last_bar_date": df["date"].iloc[ti]})
    return pd.DataFrame(rows)


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

    # Score the latest tradable anchor per ticker (forward window NOT yet
    # resolved). Generated independently of label trimming (see latest_candidates).
    cand = latest_candidates(cfg.project_path("data", "adjusted"), tickers, c, today)
    if len(cand) == 0:
        print("  no eligible candidates today")
        return 0
    # Freshness guard: if the newest available bar is far behind `today`, the
    # data is stale — warn rather than silently picking on old prices.
    newest_bar = pd.to_datetime(cand["last_bar_date"]).max()
    stale_days = (today - newest_bar).days
    if stale_days > 7:
        print(f"  WARNING: newest data bar is {newest_bar.date()} "
              f"({stale_days} days stale vs {today.date()})")

    cand_feats = feat.compute_for_anchors(
        cfg.project_path("data", "adjusted"),
        cand[["ticker", "anchor_date", "anchor_idx"]], spy_path=spy_path)
    cand = cand_feats.merge(cand, on=["ticker", "anchor_date"], how="inner")
    # ADV liquidity gate (mirror the simulator's min_adv filter).
    min_adv = c["simulator"]["min_adv_usd"]
    cand = cand[cand["adv20_usd"] >= min_adv].reset_index(drop=True)
    print(f"  candidates for next pick: {len(cand)}")
    if len(cand) == 0:
        print("  no eligible candidates after ADV filter")
        return 0

    X_pred = cand[feat_cols].to_numpy(dtype=np.float32)
    scores = mdl.predict_lgbm(art, X_pred)
    cand = cand.assign(score=scores).sort_values("score", ascending=False)

    picks = cand.head(args.top_k)[
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
