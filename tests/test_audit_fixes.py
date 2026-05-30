"""Regression tests for the bug-audit fixes (this session).

Each test is written to FAIL on the pre-fix code and PASS on the fix.
Bugs (severity):
  N1  labels.label_one          — non-positive anchor close -> inf return ->
                                   false-positive binary label (data corruption)
  S1  simulator._close_due       — exit filled at a FUTURE bar when the ticker's
                                   calendar has a gap across exit_date (look-ahead)
  S2  simulator entry            — ticker in scores but not price_lookup opens a
                                   phantom position (trapped cash, no blotter row)
  M1  models.fit_*               — PCA n_components not bounded by n_samples (crash)
  M2  models._split_calib        — stratify on a singleton class (ValueError)
  R1  report_card.auto_fill      — KeyError on a summary missing 'track'
  N2  stats.bootstrap_cagr_ci    — np.log on non-positive equity -> -inf/nan
  N3  stats.post_tax_cagr        — negative compounding base -> complex -> TypeError
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import labels as lbl
from stock_chart import simulator as sim
from stock_chart import models as mdl
from stock_chart import report_card as rc
from stock_chart import stats as st


# ─── N1: labels — non-positive anchor close must not corrupt labels ─────

def test_labels_no_inf_returns_or_false_positive_from_zero_close():
    rng = np.random.default_rng(0)
    n = 700
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    # Inject a delisting-to-zero block well past the 252-day warmup; 3 weeks
    # guarantees at least one weekly anchor lands on a zero close.
    closes[400:421] = 0.0
    df = pd.DataFrame({"date": dates, "close": closes,
                        "volume": rng.lognormal(15, 0.4, n)})
    out = lbl.label_one(df, "ZERO", horizons=[20, 40], thresholds=[0.25])
    # No surviving anchor may have a non-positive anchor close.
    assert (out["anchor_close"] > 0).all()
    # Every forward-return column must be finite on surviving rows.
    for col in [c for c in out.columns if c.startswith("fwd_ret_")]:
        assert np.isfinite(out[col].to_numpy()).all(), f"{col} has non-finite"
    # And no binary label may be 1 purely because of an inf return.
    for col in [c for c in out.columns if c.startswith("ret_") and c.endswith("pct")]:
        # a real positive is fine; we just assert none are inf-driven, i.e. the
        # matching fwd_ret is finite wherever the label is 1
        H = col.split("_")[1]  # e.g. '40d'
        fwd = out[f"fwd_ret_{H}"].to_numpy()
        lab = out[col].to_numpy()
        assert np.isfinite(fwd[lab == 1]).all()


# ─── S1: simulator — no look-ahead exit fill across a price gap ─────────

def _gap_setup():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    rng = np.random.default_rng(1)
    closes_full = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 120))
    # Ticker B is complete (populates the master union calendar with all dates).
    b = pd.DataFrame({"date": dates, "close": closes_full})
    # Ticker A is MISSING dates[60:65] — a 5-bar gap that straddles its exit.
    keep = [i for i in range(120) if not (60 <= i < 65)]
    a = pd.DataFrame({"date": dates[keep], "close": closes_full[keep]})
    price_lookup = {"A": a, "B": b}
    # Only A is scored. anchor at dates[20], resolution at dates[62] (in the gap).
    scores = pd.DataFrame([{
        "ticker": "A", "anchor_date": dates[20], "score": 1.0,
        "anchor_close": float(closes_full[20]), "adv20_usd": 1e8,
        "anchor_idx": 20, "resolution_date": dates[62],
    }])
    return dates, scores, price_lookup


def test_simulator_no_lookahead_exit_across_gap():
    dates, scores, price_lookup = _gap_setup()
    cfg = sim.SimConfig(min_adv_usd=0, slippage_bps_each_side=0.0,
                          commission_per_share=0.0, halt_risk_enabled=False)
    out = sim.simulate(scores, price_lookup, cfg)
    eq = out["equity"].set_index("date")
    # resolution_date is dates[62], which is inside A's gap. The position must
    # stay OPEN through the gap and only close at the next real bar (dates[65]),
    # NOT be force-closed at dates[62] using a future price.
    assert int(eq.loc[dates[63], "n_positions"]) == 1, \
        "position closed early across the gap (look-ahead exit)"
    assert int(eq.loc[dates[64], "n_positions"]) == 1
    blot = out["blotter"]
    assert len(blot) == 1
    assert pd.Timestamp(blot.iloc[0]["exit_date"]) == dates[65]
    # After the real exit bar the position is closed.
    assert int(eq.loc[dates[66], "n_positions"]) == 0


# ─── S2: simulator — no phantom position for unpriced ticker ────────────

def test_simulator_skips_ticker_absent_from_price_lookup():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    rng = np.random.default_rng(2)
    real = pd.DataFrame({"date": dates,
                          "close": 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 120))})
    price_lookup = {"REAL": real}  # GHOST deliberately absent
    scores = pd.DataFrame([{
        "ticker": "GHOST", "anchor_date": dates[20], "score": 1.0,
        "anchor_close": 100.0, "adv20_usd": 1e8, "anchor_idx": 20,
        "resolution_date": dates[60],
    }])
    cfg = sim.SimConfig(min_adv_usd=0, slippage_bps_each_side=0.0,
                          commission_per_share=0.0, halt_risk_enabled=False)
    out = sim.simulate(scores, price_lookup, cfg)
    # No trade should have happened; no trapped capital; no dangling position.
    assert len(out["blotter"]) == 0
    assert int(out["equity"]["n_positions"].iloc[-1]) == 0
    assert abs(out["equity"]["cash"].iloc[-1] - cfg.start_equity) < 1e-6
    assert abs(out["summary"]["end_equity"] - cfg.start_equity) < 1e-6


# ─── M1: models — PCA clamps n_components to n_samples ──────────────────

def test_pca_clamps_to_n_samples():
    rng = np.random.default_rng(3)
    n = 30  # < pca_dim=64 and < image feature dim
    X_img = rng.normal(0, 1, (n, 384)).astype(np.float32)
    X_vol = rng.normal(0, 1, (n, 252)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    art = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=64, seed=0)
    preds = mdl.predict_lr_baseline(art, X_img, X_vol)
    assert preds.shape == (n,)
    assert np.isfinite(preds).all()


# ─── M2: models — stratify falls back on singleton class ────────────────

def test_split_calib_handles_singleton_class():
    rng = np.random.default_rng(4)
    n = 40
    X_img = rng.normal(0, 1, (n, 20)).astype(np.float32)
    X_vol = rng.normal(0, 1, (n, 20)).astype(np.float32)
    y = np.zeros(n, dtype=np.int8)
    y[0] = 1  # exactly one positive -> stratify would raise pre-fix
    art = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=8, seed=0)
    preds = mdl.predict_lr_baseline(art, X_img, X_vol)
    assert preds.shape == (n,)


# ─── R1: report_card — summary missing 'track' must not KeyError ────────

def test_auto_fill_summary_missing_track(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "headline.json").write_text(json.dumps(
        {"summaries": [{"end_equity": 100000, "cagr": 0.05, "sharpe": 0.5,
                         "max_dd": -0.2}]}  # no 'track' key
    ))
    out = rc.auto_fill_from_run(rd, tmp_path)
    assert isinstance(out, dict)
    assert out.get("tag") == "run"


# ─── N2: stats — bootstrap tolerates non-positive equity ────────────────

def test_bootstrap_cagr_ci_handles_zero_equity():
    eq = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=300, freq="B"),
        "equity": np.concatenate([np.linspace(100_000, 1.0, 150),
                                    np.zeros(150)]),  # blows up to 0
    })
    ci = st.bootstrap_cagr_ci(eq, n_resamples=200, block_size=20)
    for k in ("cagr_ci", "sharpe_ci", "maxdd_ci"):
        lo, med, hi = ci[k]
        assert np.isfinite([lo, med, hi]).all(), f"{k} has non-finite: {ci[k]}"


# ─── N3: stats — post_tax_cagr survives a wiped-out account ─────────────

def test_post_tax_cagr_account_wipeout_no_crash():
    # Two -100% trades of cost 80k each on 100k start, across 2 calendar years
    # => cumulative pnl = -160k < -start_equity (100k) => negative base pre-fix.
    blot = pd.DataFrame([
        {"ticker": "A", "exit_date": pd.Timestamp("2021-06-01"),
         "trade_return_pct": -1.0, "cost": 80_000.0},
        {"ticker": "B", "exit_date": pd.Timestamp("2022-06-01"),
         "trade_return_pct": -1.0, "cost": 80_000.0},
    ])
    out = st.post_tax_cagr(blot, start_equity=100_000.0)
    assert np.isfinite(out["pre_tax_cagr"])
    assert np.isfinite(out["post_tax_cagr"])
    assert out["post_tax_cagr"] <= 0.0


# ─── BETA1+BETA2: features — beta is computed (not dead) and date-aligned ─

from stock_chart import features as feat


def test_beta_is_not_dead_code():
    """beta_spy_63d must be ~1.0 when the 'market' series equals the stock's
    own series. Pre-fix the length guard never passed (63 closes -> 62 returns
    vs 63), so beta was hard-wired to 0.0 for every input."""
    rng = np.random.default_rng(7)
    closes = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, 300))
    vols = rng.lognormal(15, 0.4, 300)
    f = feat._compute_window_features(closes, vols, closes.copy(), 270)
    assert abs(f["beta_spy_63d"] - 1.0) < 1e-6, \
        f"beta should be 1.0 vs itself, got {f['beta_spy_63d']}"


def test_beta_spy_aligned_by_date_not_position(tmp_path):
    """compute_for_anchors must align SPY to the ticker BY DATE. TICK shares
    identical closes with SPY on shared dates but starts 50 bars later, so a
    date-aligned beta is ~1.0 while a position-aligned (buggy) beta is not."""
    n_spy = 350
    rng = np.random.default_rng(11)
    spy_dates = pd.date_range("2020-01-01", periods=n_spy, freq="B")
    spy_close = 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.011, n_spy))
    pd.DataFrame({"date": spy_dates, "close": spy_close,
                  "volume": rng.lognormal(18, 0.3, n_spy)}
                 ).to_parquet(tmp_path / "SPY.parquet", index=False)
    # TICK = SPY's last 300 bars (same dates, same closes) -> 50-bar offset.
    pd.DataFrame({"date": spy_dates[50:], "close": spy_close[50:],
                  "volume": rng.lognormal(15, 0.4, n_spy - 50)}
                 ).to_parquet(tmp_path / "TICK.parquet", index=False)
    # TICK row 270 == date spy_dates[320].
    anchor_df = pd.DataFrame({"ticker": ["TICK"],
                               "anchor_date": [spy_dates[320]],
                               "anchor_idx": [270]})
    out = feat.compute_for_anchors(tmp_path, anchor_df, spy_path=tmp_path / "SPY.parquet")
    assert len(out) == 1
    beta = float(out["beta_spy_63d"].iloc[0])
    assert abs(beta - 1.0) < 1e-6, f"date-aligned beta should be ~1.0, got {beta}"


# ─── I2: data_acq — duplicate dates are deduplicated ────────────────────

from stock_chart import data_acq as da


def test_normalize_dedups_duplicate_dates():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame({"Open": [1, 2, 2.5, 3], "High": [1, 2, 2.5, 3],
                        "Low": [1, 2, 2.5, 3], "Close": [1.0, 2.0, 2.5, 3.0],
                        "Volume": [10, 20, 25, 30]}, index=idx)
    out = da._normalize_one(df)
    assert out["date"].is_unique
    assert len(out) == 3
    # keep="last": the 2024-01-03 row keeps close 2.5, not 2.0
    row = out[out["date"] == pd.Timestamp("2024-01-03")]
    assert float(row["close"].iloc[0]) == 2.5


# ─── I1: embed_dinov2 — empty batch returns (0, 384), no crash ──────────

def test_embed_arrays_empty():
    from stock_chart.embed_dinov2 import embed_arrays
    out = embed_arrays(None, np.empty((0, 224, 224, 3), dtype=np.uint8))
    assert out.shape == (0, 384)


# ─── I3: lockbox — atomic write, no temp leak, claims preserved ─────────

def test_lockbox_atomic_write(tmp_path):
    from stock_chart import lockbox as lb
    reg = tmp_path / "lockbox.registry.json"
    lb.claim_lockbox(reg, horizon=40, threshold=0.25, model="lgbm")
    lb.claim_lockbox(reg, horizon=20, threshold=0.10, model="lr")
    # No temp file left behind, registry is valid JSON, both claims present.
    assert not (tmp_path / "lockbox.registry.json.tmp").exists()
    data = json.loads(reg.read_text())
    assert "40|0.25|lgbm" in data and "20|0.1|lr" in data
    # Re-claiming the same tuple still refuses.
    with pytest.raises(lb.LockboxError):
        lb.claim_lockbox(reg, horizon=40, threshold=0.25, model="lgbm")


# ─── ALIGN1: the volume-mask contract run_pipeline relies on ────────────

def test_volume_feature_mask_aligns_to_input_rows():
    """_volume_features_for_anchors must return a mask over the INPUT frame's
    rows, with vol_feats == out[mask] in the same order. run_pipeline's
    alignment fix depends on this: embs[mask], frame.iloc[mask], and vol_feats
    must be the same rows. Reproduce a dropped (non-last) row."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_pipeline", Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    price_lookup = {
        "A": pd.DataFrame({"date": dates, "close": 100 + np.arange(300),
                            "volume": rng.lognormal(15, 0.3, 300)}),
        "B": pd.DataFrame({"date": dates, "close": 50 + np.arange(300),
                            "volume": rng.lognormal(15, 0.3, 300)}),
    }
    # Three anchors; the MIDDLE one (B, ti=10) is below the 252 lookback and
    # must be dropped -> mask = [True, False, True].
    anchor_kept = pd.DataFrame({
        "ticker": ["A", "B", "A"],
        "anchor_idx": [260, 10, 280],
    })
    vol_feats, vmask = rp._volume_features_for_anchors(anchor_kept, price_lookup, lookback=252)
    assert list(vmask) == [True, False, True]
    assert vol_feats.shape[0] == 2  # == vmask.sum(), the True rows only
    assert len(anchor_kept.iloc[vmask]) == 2


# ─── F1: forward_pick can actually produce candidates ──────────────────

def _load_forward_pick():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "forward_pick", Path(__file__).resolve().parents[1] / "scripts" / "forward_pick.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_forward_pick_latest_candidates_emits_picks(tmp_path):
    """Pre-fix forward_pick derived candidates from label_one's trimmed anchor
    set, so the newest anchor's resolution was always < today -> ZERO picks
    forever. latest_candidates must return the latest tradable, prefilter-
    passing anchor per ticker regardless of forward-window resolvability."""
    fp = _load_forward_pick()
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    # UP: strong exponential uptrend -> close >> 1.5x MA250 (passes prefilter)
    up = 100.0 * np.cumprod(1 + np.full(n, 0.01))
    pd.DataFrame({"date": dates, "close": up,
                  "volume": rng.lognormal(15, 0.2, n)}
                 ).to_parquet(tmp_path / "UP.parquet", index=False)
    # FLAT: constant -> ratio 1.0 < 1.5 (fails prefilter)
    pd.DataFrame({"date": dates, "close": np.full(n, 100.0),
                  "volume": rng.lognormal(15, 0.2, n)}
                 ).to_parquet(tmp_path / "FLAT.parquet", index=False)

    c = {"labels": {"warmup_days": 252},
         "prefilter": {"ma_window": 250, "ma_extension": 1.5, "enabled": True}}
    today = dates[-1] + pd.Timedelta(days=1)
    cand = fp.latest_candidates(tmp_path, ["UP", "FLAT"], c, today)
    # UP qualifies, FLAT is filtered out by the prefilter.
    assert list(cand["ticker"]) == ["UP"]
    row = cand.iloc[0]
    assert int(row["anchor_idx"]) == n - 1          # the latest bar
    assert abs(float(row["anchor_close"]) - up[-1]) < 1e-6

    # With the prefilter disabled, BOTH tickers become candidates.
    c["prefilter"]["enabled"] = False
    cand2 = fp.latest_candidates(tmp_path, ["UP", "FLAT"], c, today)
    assert set(cand2["ticker"]) == {"UP", "FLAT"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
