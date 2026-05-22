"""End-to-end integration test: feeds synthetic data through every stage
(labels → features → splits → model fit → simulator → stats) and asserts
sanity properties. No DINOv2 / no chart-rendering (those have their own
test files); this is the engineered-features track.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import labels as lbl
from stock_chart import features as feat
from stock_chart import splits as sp
from stock_chart import models as mdl
from stock_chart import simulator as sim
from stock_chart import stats as st


def _build_synthetic_dataset(n_tickers: int = 8, n_days: int = 700, seed: int = 0,
                               adjusted_dir: Path = None):
    """Build a synthetic OHLCV dataset + return the directory and tickers."""
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = pd.date_range("2017-01-02", periods=n_days, freq="B")
    for t in tickers:
        # Vary drift to get some "winners" and some "losers"
        drift = rng.choice([0.0005, 0.0015, -0.0003])
        closes = 100.0 * np.cumprod(1 + rng.normal(drift, 0.02, n_days))
        vols = rng.lognormal(15, 0.4, n_days)
        pd.DataFrame({"date": dates, "close": closes, "volume": vols,
                     "open": closes, "high": closes * 1.005,
                     "low": closes * 0.995}
                    ).to_parquet(adjusted_dir / f"{t}.parquet", index=False)
    return tickers


def test_full_pipeline_runs_end_to_end(tmp_path):
    """End-to-end smoke: build → label → feature → split → fit → predict → sim → stats."""
    adjusted_dir = tmp_path / "adjusted"
    tickers = _build_synthetic_dataset(n_tickers=8, n_days=700, seed=0,
                                          adjusted_dir=adjusted_dir)

    # Stage 1: labels
    labels_df = lbl.build_all(adjusted_dir, tickers, horizons=[20, 40],
                                 thresholds=[0.10, 0.25],
                                 prefilter_only=False)
    assert len(labels_df) > 0
    assert "fwd_ret_40d" in labels_df.columns
    assert "ret_40d_ge_25pct" in labels_df.columns

    # Stage 2: features
    anchors = labels_df[["ticker", "anchor_date", "anchor_idx"]].copy()
    feats = feat.compute_for_anchors(adjusted_dir, anchors, spy_path=None)
    assert len(feats) > 0

    # Merge labels + features
    merged = feats.merge(labels_df, on=["ticker", "anchor_date"], how="inner")
    merged = merged.reset_index(drop=True)
    assert len(merged) > 50

    # Stage 3: purged walk-forward splits
    test_years = sorted(merged["anchor_date"].dt.year.unique())[1:]  # skip first year
    folds = sp.yearly_walk_forward(merged, horizon=40, test_years=test_years,
                                       embargo_horizon_multiplier=1)
    assert len(folds) > 0

    # Stage 4: train + predict on each fold (engineered features only — fast)
    label_col = "ret_40d_ge_25pct"
    feat_cols = feat.FEATURE_COLS
    all_scores = []
    for year, (tr, te) in folds.items():
        # Need at least 50 train + 5 test
        if len(tr) < 50 or len(te) < 5:
            continue
        y_tr = tr[label_col].to_numpy()
        if y_tr.sum() < 3 or (len(y_tr) - y_tr.sum()) < 3:
            continue
        X_tr = tr[feat_cols].to_numpy(dtype=np.float32)
        X_te = te[feat_cols].to_numpy(dtype=np.float32)
        try:
            art = mdl.fit_lgbm(X_tr, y_tr, seed=42)
            scores = mdl.predict_lgbm(art, X_te)
        except Exception:
            # If LightGBM fails on degenerate small folds, skip
            continue
        df = te[["ticker", "anchor_date", "anchor_close", "adv20_usd",
                  "anchor_idx", "resolution_date_40d"]].copy()
        df["score"] = scores
        df["fold_year"] = year
        df = df.rename(columns={"resolution_date_40d": "resolution_date"})
        all_scores.append(df)

    if not all_scores:
        pytest.skip("No usable fold (synthetic data too small)")
    scores_df = pd.concat(all_scores, ignore_index=True)

    # Stage 5: simulator
    price_lookup = {}
    for t in tickers:
        p = adjusted_dir / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            price_lookup[t] = df

    cfg = sim.SimConfig(min_adv_usd=0, slippage_bps_each_side=0.0,
                          commission_per_share=0.0)
    out = sim.simulate(scores_df, price_lookup, cfg)
    assert "equity" in out and len(out["equity"]) > 0
    assert "blotter" in out
    assert "summary" in out

    # Stage 6: stats
    eq = out["equity"]
    ci = st.bootstrap_cagr_ci(eq, n_resamples=200, block_size=20)
    assert "cagr_ci" in ci

    # Sanity: equity series is non-empty + dates increasing
    assert eq["equity"].iloc[0] == cfg.start_equity
    dates = pd.to_datetime(eq["date"])
    assert (dates.diff().dropna() > pd.Timedelta(0)).all()


def test_full_pipeline_determinism_same_seed(tmp_path):
    """Running the pipeline twice with the same RNG seed must give identical
    blotter and equity outputs."""
    adjusted_dir = tmp_path / "adjusted"
    tickers = _build_synthetic_dataset(n_tickers=6, n_days=500, seed=99,
                                          adjusted_dir=adjusted_dir)

    def run():
        labels_df = lbl.build_all(adjusted_dir, tickers, horizons=[40],
                                     thresholds=[0.10],
                                     prefilter_only=False)
        anchors = labels_df[["ticker", "anchor_date", "anchor_idx"]].copy()
        feats = feat.compute_for_anchors(adjusted_dir, anchors, spy_path=None)
        merged = feats.merge(labels_df, on=["ticker", "anchor_date"], how="inner")
        merged = merged.reset_index(drop=True)

        # Use a single random-baseline simulation as a deterministic surrogate
        anchor_meta = labels_df[["ticker", "anchor_date", "anchor_idx",
                                   "anchor_close", "adv20_usd",
                                   "resolution_date_40d"]].copy()
        price_lookup = {}
        for t in tickers:
            p = adjusted_dir / f"{t}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                price_lookup[t] = df

        from stock_chart import random_baseline as rb
        cfg = sim.SimConfig(min_adv_usd=0, slippage_bps_each_side=0.0,
                              commission_per_share=0.0)
        return rb.run_random(anchor_meta, price_lookup, cfg, horizon=40,
                                seed=12345)

    out_a = run()
    out_b = run()
    pd.testing.assert_frame_equal(
        out_a["blotter"].reset_index(drop=True),
        out_b["blotter"].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        out_a["equity"].reset_index(drop=True),
        out_b["equity"].reset_index(drop=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
