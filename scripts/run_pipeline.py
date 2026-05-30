"""End-to-end pipeline orchestrator.

Stages: labels -> features -> render+embed -> splits -> 3 model tracks ->
simulate model + 5 simple baselines + N random seeds -> stats report.

Usage:
    python scripts/run_pipeline.py --horizon 40 --threshold 0.25 \
        --n-tickers 100 --random-seeds 100

For full reproduction, omit --n-tickers (uses entire universe).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import labels as lbl
from stock_chart import features as feat
from stock_chart import render as rnd
from stock_chart import splits as sp
from stock_chart import models as mdl
from stock_chart import simulator as sim
from stock_chart import random_baseline as rb
from stock_chart import stats as st


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _load_price_lookup(adjusted_dir: Path, tickers) -> dict:
    out = {}
    for t in tickers:
        p = adjusted_dir / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            out[t] = df.sort_values("date").reset_index(drop=True)
    return out


def _render_and_embed(anchor_meta: pd.DataFrame, price_lookup: dict,
                      lookback: int, image_size: int,
                      log_y_min: float, log_y_max: float,
                      batch_size: int, t0: float,
                      flush_every: int = 1024):
    """Streaming render+embed: keep <= flush_every images in memory at any time.

    Avoids the OOM that hit the previous accumulate-then-embed implementation.
    """
    from stock_chart.embed_dinov2 import load_model, embed_arrays
    model = load_model(num_threads=4)
    _log(f"DINOv2 loaded; rendering+embedding {len(anchor_meta)} charts (streaming)", t0)

    buf, keep_idx = [], []
    out_chunks = []
    last_log = 0

    def _flush():
        nonlocal buf
        if not buf:
            return
        arrs = np.stack(buf).astype(np.uint8)
        e = embed_arrays(model, arrs, batch_size=batch_size)
        out_chunks.append(e.astype(np.float32))
        buf = []

    for i, r in enumerate(anchor_meta.itertuples(index=False)):
        t = r.ticker
        ti = int(r.anchor_idx)
        df = price_lookup.get(t)
        if df is None or ti < lookback - 1 or ti >= len(df):
            continue
        window = df["close"].to_numpy(dtype=np.float64)[ti - lookback + 1: ti + 1]
        if len(window) < lookback:
            continue
        arr = rnd.render_to_array(window, image_size=image_size,
                                   log_y_min=log_y_min, log_y_max=log_y_max)
        buf.append(arr)
        keep_idx.append(i)
        if len(buf) >= flush_every:
            _flush()
            if len(keep_idx) - last_log >= 1024:
                _log(f"  rendered+embedded {len(keep_idx)} / {len(anchor_meta)}", t0)
                last_log = len(keep_idx)

    _flush()

    if not out_chunks:
        return None, None
    embs = np.concatenate(out_chunks, axis=0)
    _log(f"embedded -> shape {embs.shape}", t0)
    return embs, anchor_meta.iloc[keep_idx].reset_index(drop=True)


def _volume_features_for_anchors(anchor_meta: pd.DataFrame, price_lookup: dict,
                                   lookback: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor: zscore(log1p(volume)) over the 252-day window. Returns (X_vol, keep_mask)."""
    out = np.empty((len(anchor_meta), lookback), dtype=np.float32)
    keep = np.zeros(len(anchor_meta), dtype=bool)
    for i, r in enumerate(anchor_meta.itertuples(index=False)):
        df = price_lookup.get(r.ticker)
        if df is None:
            continue
        ti = int(r.anchor_idx)
        if ti < lookback - 1 or ti >= len(df):
            continue
        v = df["volume"].to_numpy(dtype=np.float64)[ti - lookback + 1: ti + 1]
        if len(v) < lookback:
            continue
        z = np.log1p(v.astype(np.float64))
        z = (z - z.mean()) / (z.std() + 1e-9)
        out[i] = z.astype(np.float32)
        keep[i] = True
    return out[keep], keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--n-tickers", type=int, default=0,
                    help="0 = full universe, otherwise subset for smoke")
    ap.add_argument("--ticker-seed", type=int, default=42)
    ap.add_argument("--random-seeds", type=int, default=100)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out-tag", type=str, default="run")
    args = ap.parse_args()

    t0 = time.time()
    c = cfg.load(args.config)
    H = args.horizon
    T = args.threshold
    label_key = f"ret_{H}d_ge_{int(round(T * 100))}pct"

    out_dir = cfg.project_path("reports", args.out_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    adjusted_dir = cfg.project_path("data", "adjusted")
    universe_df = pd.read_csv(cfg.project_path("data", "universe", "working_universe.csv"))
    all_tickers = universe_df["ticker"].tolist()
    if args.n_tickers > 0:
        rng = np.random.default_rng(args.ticker_seed)
        idx = rng.choice(len(all_tickers), size=min(args.n_tickers, len(all_tickers)), replace=False)
        tickers = sorted(all_tickers[i] for i in idx)
    else:
        tickers = sorted(all_tickers)
    _log(f"universe size: {len(tickers)}", t0)

    on_disk = [t for t in tickers if (adjusted_dir / f"{t}.parquet").exists()]
    _log(f"with parquet on disk: {len(on_disk)}", t0)

    # 1. Build labels (prefilter applied internally)
    horizons = c["labels"]["horizons"]
    thresholds = c["labels"]["thresholds"]
    _log(f"building labels for {len(on_disk)} tickers...", t0)
    labels_df = lbl.build_all(adjusted_dir, on_disk, horizons, thresholds,
                               warmup_days=c["labels"]["warmup_days"],
                               ma_window=c["prefilter"]["ma_window"],
                               ma_extension=c["prefilter"]["ma_extension"],
                               prefilter_only=c["prefilter"]["enabled"])
    _log(f"labels rows: {len(labels_df)}", t0)
    if len(labels_df) == 0:
        print("No labels generated. Exiting.")
        return
    labels_df.to_parquet(out_dir / "labels.parquet", index=False)

    # 2. Anchor meta and price lookup
    anchor_meta = labels_df[["ticker", "anchor_date", "anchor_idx", "anchor_close",
                              "adv20_usd", f"resolution_date_{H}d"]].copy()
    price_lookup = _load_price_lookup(adjusted_dir, anchor_meta["ticker"].unique())
    _log(f"price lookup tickers: {len(price_lookup)}", t0)

    # 3. Engineered features (FALSIFICATION baseline)
    _log("computing engineered features...", t0)
    spy_path = adjusted_dir / "SPY.parquet"
    feats = feat.compute_for_anchors(adjusted_dir, anchor_meta, spy_path=spy_path)
    _log(f"features rows: {len(feats)}", t0)
    feats.to_parquet(out_dir / "engineered_features.parquet", index=False)

    # Align labels to features
    merged = feats.merge(labels_df, on=["ticker", "anchor_date"], how="inner")
    _log(f"feature-label merged rows: {len(merged)}", t0)

    # 4. Render + DINOv2 embed
    _log("rendering charts + embedding via DINOv2 (CPU)...", t0)
    embs, anchor_kept = _render_and_embed(
        anchor_meta=merged,
        price_lookup=price_lookup,
        lookback=c["render"]["lookback_days"],
        image_size=c["render"]["image_size"],
        log_y_min=c["render"]["log_y_min"],
        log_y_max=c["render"]["log_y_max"],
        batch_size=c["embed"]["batch_size"],
        t0=t0,
    )
    if embs is None:
        print("No embeddings produced. Exiting.")
        return
    np.save(out_dir / "dinov2_embeddings.npy", embs)
    anchor_kept.to_parquet(out_dir / "anchor_kept.parquet", index=False)

    # Volume features (as in original doc): zscore(log1p(volume)) over 252-day window.
    # CRITICAL alignment: `embs` is built from the render keep_idx and is 1:1 with
    # `anchor_kept`. Compute volume features on `anchor_kept` (NOT `merged`) and
    # apply the resulting mask to embs and the frame together, so embs, vol_feats,
    # and merged_kept are the SAME rows in the SAME order by construction. The
    # previous code used `merged.iloc[:len(embs)]` (a positional PREFIX) and a
    # mask computed over `merged` — which silently misaligned image features with
    # their labels/metadata the moment any anchor was dropped.
    vol_feats, vmask = _volume_features_for_anchors(anchor_kept, price_lookup,
                                                     lookback=c["render"]["lookback_days"])
    embs = embs[vmask]
    merged_kept = anchor_kept.iloc[vmask].reset_index(drop=True)
    _log(f"final aligned rows: emb={embs.shape[0]} vol={vol_feats.shape[0]} merged={len(merged_kept)}", t0)

    # 5. Splits: purged walk-forward
    folds = sp.yearly_walk_forward(
        merged_kept, horizon=H,
        test_years=c["splits"]["test_years"],
        embargo_horizon_multiplier=c["splits"]["embargo_horizon_multiplier"],
    )
    _log(f"folds: {sorted(folds.keys())}", t0)

    # 6. Train 3 tracks per fold; concatenate test scores across folds
    label_col = label_key
    if label_col not in merged_kept.columns:
        print(f"label column {label_col} not present; available labels include "
              f"{[c for c in merged_kept.columns if c.startswith('ret_')][:10]}")
        return
    y_all = merged_kept[label_col].to_numpy(dtype=np.int8)

    feat_cols = feat.FEATURE_COLS
    X_eng = merged_kept[feat_cols].to_numpy(dtype=np.float32)

    test_scores = {"lr_baseline": [], "lgbm_image": [], "lgbm_engineered": []}
    fold_artifacts = {}

    for year, (tr_df, te_df) in folds.items():
        if len(tr_df) < c["splits"]["min_train_examples"]:
            _log(f"  fold {year}: {len(tr_df)} train rows -- skipping (below min)", t0)
            continue
        # Need to align tr/te indices back into merged_kept positions
        tr_idx = tr_df.index.values
        te_idx = te_df.index.values
        y_tr = y_all[tr_idx]
        # te_idx labels are not used (we predict on test, don't train on it)
        if y_tr.sum() < 10 or (len(y_tr) - y_tr.sum()) < 10:
            _log(f"  fold {year}: degenerate class balance, skipping", t0)
            continue

        Xi_tr = embs[tr_idx]
        Xi_te = embs[te_idx]
        Xv_tr = vol_feats[tr_idx]
        Xv_te = vol_feats[te_idx]
        Xe_tr = X_eng[tr_idx]
        Xe_te = X_eng[te_idx]

        seed = c["seeds"]["sklearn_random_state"]
        _log(f"  fold {year}: training (n_tr={len(tr_idx)}, n_te={len(te_idx)})", t0)

        # Track A: LR baseline (matches doc)
        artA = mdl.fit_lr_baseline(Xi_tr, Xv_tr, y_tr,
                                     pca_dim=c["models"]["pca_dim"], seed=seed,
                                     lr_kwargs=c["models"]["logreg"])
        scA = mdl.predict_lr_baseline(artA, Xi_te, Xv_te)

        # Track B: LightGBM on image+volume PCA64
        artB = mdl.fit_lgbm_image(Xi_tr, Xv_tr, y_tr,
                                    pca_dim=c["models"]["pca_dim"], seed=seed,
                                    lgbm_kwargs=c["models"]["lgbm"])
        scB = mdl.predict_lgbm_image(artB, Xi_te, Xv_te)

        # Track C: LightGBM on engineered features (FALSIFICATION TEST)
        artC = mdl.fit_lgbm(Xe_tr, y_tr, seed=seed, lgbm_kwargs=c["models"]["lgbm"])
        scC = mdl.predict_lgbm(artC, Xe_te)

        for name, sc in zip(["lr_baseline", "lgbm_image", "lgbm_engineered"],
                            [scA, scB, scC]):
            df = te_df[["ticker", "anchor_date", "anchor_close", "adv20_usd",
                        "anchor_idx", f"resolution_date_{H}d"]].copy()
            df["score"] = sc
            df["fold_year"] = year
            df["model"] = name
            df = df.rename(columns={f"resolution_date_{H}d": "resolution_date"})
            test_scores[name].append(df)

        fold_artifacts[year] = {"lr": "ok", "lgbm_image": "ok", "lgbm_engineered": "ok"}

    if not any(test_scores.values()):
        print("No fold produced scores. Exiting.")
        return

    score_dfs = {k: pd.concat(v, ignore_index=True) for k, v in test_scores.items() if v}
    for k, df in score_dfs.items():
        df.to_parquet(out_dir / f"scores_{k}.parquet", index=False)

    # 7. Simulate each track + 5 simple baselines + N random seeds
    sim_section = c["simulator"]
    sim_cfg = sim.SimConfig(
        start_equity=sim_section["start_equity"],
        max_position_pct=sim_section["max_position_pct"],
        max_new_per_week=sim_section["max_new_per_week"],
        max_adv_pct=sim_section["max_adv_pct"],
        min_adv_usd=sim_section["min_adv_usd"],
        slippage_bps_each_side=sim_section["slippage_bps_each_side"],
        commission_per_share=sim_section["commission_per_share"],
        no_duplicate_ticker=sim_section["no_duplicate_ticker"],
        trailing_stop_pct=sim_section.get("trailing_stop_pct", 0.0),
        use_almgren_chriss_impact=sim_section.get("use_almgren_chriss_impact", False),
        impact_eta=sim_section.get("impact_eta", 0.142),
        impact_beta=sim_section.get("impact_beta", 0.5),
        permanent_frac=sim_section.get("permanent_frac", 0.5),
        daily_vol_default=sim_section.get("daily_vol_default", 0.03),
        halt_risk_enabled=sim_section.get("halt_risk_enabled", False),
        halt_risk_seed=sim_section.get("halt_risk_seed", 42),
    )
    _log(f"simulator config: trail_stop={sim_cfg.trailing_stop_pct}, "
         f"AC_impact={sim_cfg.use_almgren_chriss_impact}, halt={sim_cfg.halt_risk_enabled}", t0)
    summaries = []
    for k, df in score_dfs.items():
        _log(f"simulating model: {k} ({len(df)} candidate rows)", t0)
        out = sim.simulate(df, price_lookup, sim_cfg)
        out["equity"].to_parquet(out_dir / f"equity_{k}.parquet", index=False)
        out["blotter"].to_parquet(out_dir / f"blotter_{k}.parquet", index=False)
        s = out["summary"]; s["track"] = k
        summaries.append(s)

    # Simple baselines
    for bn in rb.SIMPLE_BASELINES:
        _log(f"simulating simple baseline: {bn}", t0)
        out = rb.run_simple_baseline(feats, anchor_meta, price_lookup, bn, sim_cfg, horizon=H)
        out["equity"].to_parquet(out_dir / f"equity_{bn}.parquet", index=False)
        out["blotter"].to_parquet(out_dir / f"blotter_{bn}.parquet", index=False)
        s = out["summary"]; s["track"] = bn
        summaries.append(s)

    # Random seeds
    _log(f"running {args.random_seeds} random-seed simulations...", t0)
    rnd_summary = rb.run_random_seeds(anchor_meta, price_lookup, sim_cfg, horizon=H,
                                        n_seeds=args.random_seeds,
                                        base_seed=c["seeds"]["random_baseline_base_seed"])
    rnd_summary.to_parquet(out_dir / "random_seeds_summary.parquet", index=False)

    # 8. Stats
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "all_track_summary.csv", index=False)
    _log(f"summary:\n{summary_df.to_string()}", t0)

    rnd_median_cagr = float(rnd_summary["cagr"].median())
    rnd_p95_eq = float(rnd_summary["end_equity"].quantile(0.95))
    rnd_p99 = float(rnd_summary["end_equity"].quantile(0.99))

    headline = []
    for r in summaries:
        headline.append({
            "track": r["track"],
            "end_equity": r["end_equity"],
            "cagr": r["cagr"],
            "max_dd": r["max_drawdown"],
            "sharpe": r["sharpe_annualized"],
            "vs_random_pct": float(percentile_in(r["end_equity"], rnd_summary["end_equity"])),
        })

    # Deflated Sharpe vs N_trials = 36 labels * 3 model tracks = 108 (we ran 1 label
    # but the configuration *space* enumerated by the original doc + this rebuild is 108).
    n_trials_assumed = 108
    best_track = max(summaries, key=lambda x: x["cagr"])
    eq_best = pd.read_parquet(out_dir / f"equity_{best_track['track']}.parquet")
    rets = st.daily_returns_from_equity(eq_best)
    n_obs = len(rets)
    sr = best_track["sharpe_annualized"]
    dsr = st.deflated_sharpe_ratio(observed_sr=sr, n_obs=n_obs, n_trials=n_trials_assumed)

    final = {
        "headline_label": label_key,
        "n_tickers_loaded": len(price_lookup),
        "n_anchor_rows_after_prefilter": len(merged_kept),
        "folds_run": list(fold_artifacts.keys()),
        "summaries": headline,
        "random_baseline": {
            "n_seeds": int(len(rnd_summary)),
            "median_cagr": rnd_median_cagr,
            "p95_end_equity": rnd_p95_eq,
            "p99_end_equity": rnd_p99,
            "median_end_equity": float(rnd_summary["end_equity"].median()),
        },
        "deflated_sharpe": dsr,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "headline.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    _log(f"DONE. Wrote {out_dir}/headline.json", t0)
    print(json.dumps(final, indent=2, default=str))


def percentile_in(value, distribution) -> float:
    arr = np.asarray(distribution, dtype=float)
    return float((arr <= value).mean())


if __name__ == "__main__":
    main()
