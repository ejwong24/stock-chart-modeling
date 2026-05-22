"""SMA250 random-baseline system-comparison test.

Implements the spec from SIMPLE_SMA250_RANDOM_SYSTEM_COMPARISON_TEST.docx.
The goal is NOT to evaluate the strategy — it's to verify that this
backtesting system agrees with another implementation on a trivial baseline
before comparing fancier model variants.

Spec:
- Pre-filter: Close > 1.5 × SMA250 (ONLY filter, no model, no ranking)
- Weekly Friday anchors (last trading day of week if Friday is a holiday)
- Random selection: pick up to 5 eligible non-already-held tickers per anchor
- Position size: 4% of current equity
- Hold: 20 trading days (Test A) AND 40 trading days (Test B)
- 200 seeds (0..199)
- Costs/slippage/commissions/ADV/liquidity ALL = 0
- Entry-date span: 2019-01-04 → (2026-03-20 for H=20, 2026-02-20 for H=40)
- Output: per-seed summary AND full trade blotter (with hold_days+seed columns)

Note: our simulator uses INTEGER share floors (not fractional), declared in
output. Other systems may use fractional and produce slightly higher equity.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg
from stock_chart import simulator as sim


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _load_price_lookup(adjusted_dir: Path, tickers) -> dict:
    out = {}
    for t in set(tickers):
        p = adjusted_dir / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        out[t] = df.sort_values("date").reset_index(drop=True)
    return out


def make_sim_config() -> sim.SimConfig:
    """Per the spec: zero costs, no liquidity filter, no advanced exits."""
    return sim.SimConfig(
        start_equity=100_000.0,
        max_position_pct=0.04,
        max_new_per_week=5,
        max_adv_pct=1.0,            # disabled (capped at equity*max_position_pct)
        min_adv_usd=0.0,            # NO ADV FILTER per spec
        slippage_bps_each_side=0.0, # NO SLIPPAGE per spec
        commission_per_share=0.0,   # NO COMMISSIONS per spec
        no_duplicate_ticker=True,   # spec: "duplicate overlapping ticker positions: not allowed"
        trailing_stop_pct=0.0,      # OFF
        use_almgren_chriss_impact=False,  # OFF
        halt_risk_enabled=False,    # OFF
    )


def run_one(anchor_meta: pd.DataFrame, price_lookup: dict, seed: int,
             horizon: int, sim_cfg: sim.SimConfig) -> dict:
    """Run one random-selection simulation; return blotter + summary."""
    rng = np.random.default_rng(seed)
    df = anchor_meta.copy()
    df["score"] = rng.random(len(df))
    df = df.rename(columns={f"resolution_date_{horizon}d": "resolution_date"})
    out = sim.simulate(
        df[["ticker", "anchor_date", "score", "anchor_close",
            "adv20_usd", "anchor_idx", "resolution_date"]],
        price_lookup, sim_cfg,
    )
    return out


def filter_anchors(labels_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per spec: entry dates 2019-01-04 → (2026-03-20 H=20, 2026-02-20 H=40).

    Filter to MA250-eligible anchors only (already done in our labels.parquet).
    """
    df = labels_df[labels_df["prefilter_ma250_15"] == 1].copy()
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])

    last_entry = pd.Timestamp("2026-03-20") if horizon == 20 else pd.Timestamp("2026-02-20")
    first_entry = pd.Timestamp("2019-01-04")
    df = df[(df["anchor_date"] >= first_entry) & (df["anchor_date"] <= last_entry)]

    # Drop rows whose forward window cannot fully resolve before the test end
    final_exit = pd.Timestamp("2026-04-20")
    res_col = f"resolution_date_{horizon}d"
    df = df[df[res_col] <= final_exit]

    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", default="full",
                     help="Existing run with labels.parquet + adjusted price data")
    ap.add_argument("--out-tag", default="sma250_test")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--horizons", type=str, default="20,40")
    ap.add_argument("--n-jobs", type=int, default=3,
                     help="Parallel seed workers (3 = leave 1 core for system)")
    args = ap.parse_args()

    t0 = time.time()
    horizons = [int(h) for h in args.horizons.split(",")]

    src_dir = cfg.project_path("reports", args.source_tag)
    out_dir = cfg.project_path("reports", args.out_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load anchors (MA250-prefiltered already)
    labels_path = src_dir / "labels.parquet"
    if not labels_path.exists():
        print(f"missing {labels_path}")
        return 2
    labels_df = pd.read_parquet(labels_path)
    _log(f"loaded labels: {len(labels_df)} rows", t0)

    # Load price lookup once (shared across both horizons)
    tickers = labels_df["ticker"].unique()
    _log(f"loading prices for {len(tickers)} tickers ...", t0)
    adjusted_dir = cfg.project_path("data", "adjusted")
    price_lookup = _load_price_lookup(adjusted_dir, tickers)
    _log(f"  loaded {len(price_lookup)}", t0)

    sim_cfg = make_sim_config()
    _log("sim config: zero costs, no ADV filter, no trailing stop, integer shares", t0)

    all_summaries = []

    def _process_seed(anchor_meta, price_lookup, seed, H, sim_cfg):
        out = run_one(anchor_meta, price_lookup, seed=seed, horizon=H, sim_cfg=sim_cfg)
        blot = out["blotter"]
        summary = out["summary"]
        row = {
            "hold_days": H, "seed": seed,
            "start_date": anchor_meta["anchor_date"].min(),
            "end_date": anchor_meta["anchor_date"].max(),
            "starting_equity": summary["start_equity"],
            "ending_equity": summary["end_equity"],
            "cagr": summary["cagr"],
            "max_drawdown": summary["max_drawdown"],
            "total_trades": summary["n_trades"],
            "win_rate": summary["win_rate"],
            "average_trade_return": summary["avg_trade_return"],
            "median_trade_return": summary["median_trade_return"],
            "profit_factor": summary["profit_factor"],
            "average_open_positions": summary["avg_positions"],
            "max_open_positions": summary["max_positions"],
        }
        if len(blot) > 0:
            tdf = blot.copy()
            tdf["hold_days"] = H
            tdf["seed"] = seed
            tdf["pnl"] = tdf["proceeds"] - tdf["cost"]
            tdf = tdf.rename(columns={
                "trade_return_pct": "trade_return",
                "cost": "position_value",
            })
            cols = ["hold_days", "seed", "ticker", "entry_date", "entry_price",
                    "exit_date", "exit_price", "shares", "position_value",
                    "trade_return", "pnl"]
            return row, tdf[cols]
        return row, None

    for H in horizons:
        anchor_meta = filter_anchors(labels_df, horizon=H)
        _log(f"H={H}: anchors after date-range filter = {len(anchor_meta)}", t0)
        _log(f"  running {args.n_seeds} seeds with {args.n_jobs} parallel workers ...", t0)

        results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
            delayed(_process_seed)(anchor_meta, price_lookup, seed, H, sim_cfg)
            for seed in range(args.n_seeds)
        )
        per_seed_summary = [r[0] for r in results]
        per_seed_trades = [r[1] for r in results if r[1] is not None]
        _log(f"H={H}: parallel run complete; {len(per_seed_summary)} summaries", t0)

        # Persist per-horizon outputs
        sum_df = pd.DataFrame(per_seed_summary)
        sum_path = out_dir / f"H{H}_per_seed_summary.csv"
        sum_df.to_csv(sum_path, index=False)
        _log(f"wrote {sum_path}", t0)

        if per_seed_trades:
            trades_df = pd.concat(per_seed_trades, ignore_index=True)
            trades_path = out_dir / f"H{H}_full_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            _log(f"wrote {trades_path}  ({len(trades_df)} trades total)", t0)

        all_summaries.append(sum_df)

    # Aggregate metrics per spec section 10
    aggregate = {}
    for H, sum_df in zip(horizons, all_summaries):
        s = {
            "horizon_trading_days": H,
            "n_seeds": int(len(sum_df)),
            "median_ending_equity": float(sum_df["ending_equity"].median()),
            "p5_ending_equity": float(sum_df["ending_equity"].quantile(0.05)),
            "p95_ending_equity": float(sum_df["ending_equity"].quantile(0.95)),
            "median_cagr": float(sum_df["cagr"].median()),
            "p5_cagr": float(sum_df["cagr"].quantile(0.05)),
            "p95_cagr": float(sum_df["cagr"].quantile(0.95)),
            "median_max_drawdown": float(sum_df["max_drawdown"].median()),
            "p5_max_drawdown": float(sum_df["max_drawdown"].quantile(0.05)),
            "p95_max_drawdown": float(sum_df["max_drawdown"].quantile(0.95)),
            "median_profit_factor": float(sum_df["profit_factor"].median()),
            "median_win_rate": float(sum_df["win_rate"].median()),
            "median_trade_count": float(sum_df["total_trades"].median()),
        }
        aggregate[f"H{H}"] = s
        _log(f"H={H} aggregates: median_eq=${s['median_ending_equity']:,.0f} "
             f"median_CAGR={s['median_cagr']*100:+.2f}% "
             f"median_DD={s['median_max_drawdown']*100:.2f}%", t0)

    aggregate["meta"] = {
        "spec": "SIMPLE_SMA250_RANDOM_SYSTEM_COMPARISON_TEST.docx",
        "system": "stock_chart_modeling rebuild (this repo)",
        "fractional_shares": False,
        "share_rounding": "int(target_usd // fill_price)",
        "starting_equity": 100_000,
        "n_seeds_per_horizon": args.n_seeds,
        "wall_seconds": round(time.time() - t0, 1),
        "rng": "numpy default_rng",
        "data_source": "yfinance auto_adjust=True",
        "universe_size": len(tickers),
    }
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, default=str))
    _log(f"DONE. Outputs in {out_dir}", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
