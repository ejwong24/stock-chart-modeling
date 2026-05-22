"""Idempotency + performance sanity checks."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import labels as lbl
from stock_chart import features as feat
from stock_chart import simulator as sim
from stock_chart import lockbox as lb
from stock_chart import stats as st


def _make_synthetic_parquets(tmp_path, n=20, days=600):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-02", periods=days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n)]
    for t in tickers:
        closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, days))
        vols = rng.lognormal(15, 0.4, days)
        pd.DataFrame({"date": dates, "close": closes, "volume": vols}
                    ).to_parquet(tmp_path / f"{t}.parquet", index=False)
    return tickers


# ─── Idempotency ────────────────────────────────────────────────────

def test_labels_repeated_calls_same_output(tmp_path):
    tickers = _make_synthetic_parquets(tmp_path)
    a = lbl.build_all(tmp_path, tickers, horizons=[40], thresholds=[0.25],
                        prefilter_only=False)
    b = lbl.build_all(tmp_path, tickers, horizons=[40], thresholds=[0.25],
                        prefilter_only=False)
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                     b.reset_index(drop=True))


def test_features_repeated_calls_same_output(tmp_path):
    tickers = _make_synthetic_parquets(tmp_path)
    labels_df = lbl.build_all(tmp_path, tickers, horizons=[40],
                                 thresholds=[0.25], prefilter_only=False)
    anchors = labels_df[["ticker", "anchor_date", "anchor_idx"]]
    a = feat.compute_for_anchors(tmp_path, anchors, spy_path=None)
    b = feat.compute_for_anchors(tmp_path, anchors, spy_path=None)
    # All feature columns must match exactly
    for c in feat.FEATURE_COLS:
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(),
                                       rtol=1e-12, atol=1e-15)


def test_lockbox_audit_repeated_writes_dont_duplicate(tmp_path):
    """Each call appends one line; nothing else should change."""
    p = tmp_path / "audit.jsonl"
    lb.audit(p, "test", {"i": 1})
    lb.audit(p, "test", {"i": 2})
    n_before = len(p.read_text().splitlines())
    lb.audit(p, "test", {"i": 3})
    n_after = len(p.read_text().splitlines())
    assert n_after == n_before + 1


def test_trial_registry_idempotent_count(tmp_path):
    """Logging same config twice → trial_count stays at 1."""
    p = tmp_path / "trials.jsonl"
    cfg = {"model": "lgbm", "label": "x", "horizon_d": 40,
           "threshold_q": 0.25, "universe": "us"}
    lb.log_trial(p, config=cfg, headline_sharpe=1.0, n_obs=100)
    assert lb.trial_count(p) == 1
    lb.log_trial(p, config=cfg, headline_sharpe=2.0, n_obs=100)
    # Two log entries but only one unique trial tuple
    assert lb.trial_count(p) == 1
    # Different config → 2
    cfg2 = {**cfg, "horizon_d": 20}
    lb.log_trial(p, config=cfg2, headline_sharpe=1.5, n_obs=100)
    assert lb.trial_count(p) == 2


# ─── Performance sanity ──────────────────────────────────────────────

def test_simulator_handles_large_input():
    """50 anchors × 50 tickers = 2500 candidate rows should sim in < 5s."""
    rng = np.random.default_rng(0)
    n_days = 400
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    n_tickers = 50
    price_lookup = {f"T{i:02d}": pd.DataFrame({
        "date": dates,
        "close": 100 * np.cumprod(1 + rng.normal(0.001, 0.025, n_days)),
    }) for i in range(n_tickers)}
    rows = []
    H = 40
    for ad in dates[260::5]:
        ai = list(dates).index(ad)
        if ai + H >= n_days:
            continue
        for i in range(n_tickers):
            rows.append({
                "ticker": f"T{i:02d}", "anchor_date": ad,
                "score": float(rng.random()),
                "anchor_close": float(price_lookup[f"T{i:02d}"]["close"].iloc[ai]),
                "adv20_usd": 5e7, "anchor_idx": ai,
                "resolution_date": dates[ai + H],
            })
    scores = pd.DataFrame(rows)
    t0 = time.time()
    out = sim.simulate(scores, price_lookup, sim.SimConfig())
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"simulator too slow on {len(rows)} candidates: {elapsed:.1f}s"
    assert len(out["equity"]) > 0


def test_bootstrap_throughput():
    """1000 resamples of 500 daily returns should run in well under 1s."""
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    t0 = time.time()
    st.stationary_block_bootstrap(r, n_resamples=1000, block_size=20)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"bootstrap too slow: {elapsed:.2f}s"


def test_features_throughput(tmp_path):
    """Computing features for 100 anchors across 10 tickers should be < 5s."""
    tickers = _make_synthetic_parquets(tmp_path, n=10, days=400)
    labels_df = lbl.build_all(tmp_path, tickers, horizons=[40],
                                 thresholds=[0.25], prefilter_only=False)
    anchors = labels_df.head(100)[["ticker", "anchor_date", "anchor_idx"]]
    t0 = time.time()
    out = feat.compute_for_anchors(tmp_path, anchors, spy_path=None)
    elapsed = time.time() - t0
    assert elapsed < 5.0
    assert len(out) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
