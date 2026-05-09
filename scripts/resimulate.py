"""Re-run the simulator stage on an existing run's cached scores.

Skips labels, features, render, embed, train (~4 hours of compute on full
universe). Just re-runs simulator + baselines + random seeds + stats with
a new SimConfig (e.g., trailing stop + Almgren-Chriss + halt risk).

Usage:
    python scripts/resimulate.py --source-tag full --out-tag full_realistic \\
        --config config/realistic.yaml --random-seeds 200
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import simulator as sim
from stock_chart import random_baseline as rb
from stock_chart import stats as st


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _load_price_lookup(adjusted_dir: Path, tickers) -> dict:
    out = {}
    for t in set(tickers):
        p = adjusted_dir / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            out[t] = df.sort_values("date").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", required=True,
                     help="Existing run with scores_*.parquet to reuse")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--config", default="config/realistic.yaml")
    ap.add_argument("--random-seeds", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    c = cfg.load(args.config)
    H = args.horizon

    src_dir = cfg.project_path("reports", args.source_tag)
    out_dir = cfg.project_path("reports", args.out_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (src_dir / "headline.json").exists():
        print(f"source run {args.source_tag} missing headline.json")
        return 2

    src_head = json.loads((src_dir / "headline.json").read_text())
    label_key = src_head.get("headline_label", f"ret_{H}d_ge_25pct")

    # Build SimConfig from the new YAML
    s = c["simulator"]
    sim_cfg = sim.SimConfig(
        start_equity=s["start_equity"],
        max_position_pct=s["max_position_pct"],
        max_new_per_week=s["max_new_per_week"],
        max_adv_pct=s["max_adv_pct"],
        min_adv_usd=s["min_adv_usd"],
        slippage_bps_each_side=s["slippage_bps_each_side"],
        commission_per_share=s["commission_per_share"],
        no_duplicate_ticker=s["no_duplicate_ticker"],
        trailing_stop_pct=s.get("trailing_stop_pct", 0.0),
        use_almgren_chriss_impact=s.get("use_almgren_chriss_impact", False),
        impact_eta=s.get("impact_eta", 0.142),
        impact_beta=s.get("impact_beta", 0.5),
        permanent_frac=s.get("permanent_frac", 0.5),
        daily_vol_default=s.get("daily_vol_default", 0.03),
        halt_risk_enabled=s.get("halt_risk_enabled", False),
        halt_risk_seed=s.get("halt_risk_seed", 42),
    )
    _log(f"simulator config: trail_stop={sim_cfg.trailing_stop_pct} | "
         f"AC_impact={sim_cfg.use_almgren_chriss_impact} | "
         f"halt={sim_cfg.halt_risk_enabled} | "
         f"min_ADV=${sim_cfg.min_adv_usd:,.0f}", t0)

    # Build price lookup from union of tickers across all source scores
    tickers = set()
    score_paths = list(src_dir.glob("scores_*.parquet"))
    for p in score_paths:
        df = pd.read_parquet(p, columns=["ticker"])
        tickers.update(df["ticker"].unique())
    # Plus simple-baseline candidate tickers from anchor_kept
    if (src_dir / "anchor_kept.parquet").exists():
        ak = pd.read_parquet(src_dir / "anchor_kept.parquet", columns=["ticker"])
        tickers.update(ak["ticker"].unique())

    _log(f"loading price lookup for {len(tickers)} tickers...", t0)
    adjusted_dir = cfg.project_path("data", "adjusted")
    price_lookup = _load_price_lookup(adjusted_dir, tickers)
    _log(f"  loaded {len(price_lookup)}", t0)

    # Re-simulate model tracks
    summaries = []
    for sp in score_paths:
        track = sp.stem.replace("scores_", "")
        df = pd.read_parquet(sp)
        df = df.rename(columns={f"resolution_date_{H}d": "resolution_date"})
        _log(f"resimulating model: {track} ({len(df)} candidate rows)", t0)
        out = sim.simulate(df, price_lookup, sim_cfg)
        out["equity"].to_parquet(out_dir / f"equity_{track}.parquet", index=False)
        out["blotter"].to_parquet(out_dir / f"blotter_{track}.parquet", index=False)
        s_summary = out["summary"]; s_summary["track"] = track
        summaries.append(s_summary)

    # Simple baselines: re-run from anchor_kept + engineered features
    if (src_dir / "engineered_features.parquet").exists() and (src_dir / "anchor_kept.parquet").exists():
        feats = pd.read_parquet(src_dir / "engineered_features.parquet")
        anchor_meta = pd.read_parquet(src_dir / "anchor_kept.parquet")
        for bn in rb.SIMPLE_BASELINES:
            _log(f"resimulating simple baseline: {bn}", t0)
            try:
                out = rb.run_simple_baseline(feats, anchor_meta, price_lookup,
                                              bn, sim_cfg, horizon=H)
                out["equity"].to_parquet(out_dir / f"equity_{bn}.parquet", index=False)
                out["blotter"].to_parquet(out_dir / f"blotter_{bn}.parquet", index=False)
                s_summary = out["summary"]; s_summary["track"] = bn
                summaries.append(s_summary)
            except Exception as e:
                _log(f"  {bn} failed: {e}", t0)

        # Random baselines
        _log(f"running {args.random_seeds} random seeds...", t0)
        rnd_summary = rb.run_random_seeds(anchor_meta, price_lookup, sim_cfg,
                                            horizon=H, n_seeds=args.random_seeds,
                                            base_seed=c["seeds"]["random_baseline_base_seed"])
        rnd_summary.to_parquet(out_dir / "random_seeds_summary.parquet", index=False)
    else:
        rnd_summary = pd.DataFrame()

    # Summary CSV + headline JSON
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "all_track_summary.csv", index=False)
    _log(f"summary:\n{summary_df.to_string()}", t0)

    # Stats
    headline = []
    rnd_eq = rnd_summary["end_equity"] if len(rnd_summary) else pd.Series([0.0])
    for r in summaries:
        headline.append({
            "track": r["track"],
            "end_equity": r["end_equity"],
            "cagr": r["cagr"],
            "max_dd": r["max_drawdown"],
            "sharpe": r["sharpe_annualized"],
            "vs_random_pct": float((rnd_eq <= r["end_equity"]).mean()) if len(rnd_eq) else 0.0,
        })

    n_trials_assumed = 108
    if summaries:
        best = max(summaries, key=lambda x: x["cagr"])
        eq_best = pd.read_parquet(out_dir / f"equity_{best['track']}.parquet")
        rets = st.daily_returns_from_equity(eq_best)
        n_obs = len(rets)
        sr = best["sharpe_annualized"]
        dsr = st.deflated_sharpe_ratio(observed_sr=sr, n_obs=n_obs, n_trials=n_trials_assumed)
    else:
        dsr = {}

    final = {
        "headline_label": label_key,
        "source_tag": args.source_tag,
        "resimulated": True,
        "simulator_config": {
            "trailing_stop_pct": sim_cfg.trailing_stop_pct,
            "use_almgren_chriss_impact": sim_cfg.use_almgren_chriss_impact,
            "halt_risk_enabled": sim_cfg.halt_risk_enabled,
            "min_adv_usd": sim_cfg.min_adv_usd,
        },
        "n_tickers_loaded": len(price_lookup),
        "summaries": headline,
        "random_baseline": {
            "n_seeds": int(len(rnd_summary)) if len(rnd_summary) else 0,
            "median_cagr": float(rnd_summary["cagr"].median()) if len(rnd_summary) else 0,
            "p95_end_equity": float(rnd_summary["end_equity"].quantile(0.95)) if len(rnd_summary) else 0,
            "p99_end_equity": float(rnd_summary["end_equity"].quantile(0.99)) if len(rnd_summary) else 0,
            "median_end_equity": float(rnd_summary["end_equity"].median()) if len(rnd_summary) else 0,
        },
        "deflated_sharpe": dsr,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "headline.json").write_text(json.dumps(final, indent=2, default=str))
    _log(f"DONE. Wrote {out_dir}/headline.json", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
