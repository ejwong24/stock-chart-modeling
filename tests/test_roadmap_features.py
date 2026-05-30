"""Tests for the three roadmap features built this session:
  - stats.paired_gap_bootstrap  (model-vs-best-baseline significance)
  - scripts/settle_picks.py     (forward-pick realized-return loop)
  - manifest.write/verify_data_hashes  (reproducibility manifest)
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from stock_chart import stats as st
from stock_chart import manifest as mf


def _equity(dates, vals):
    return pd.DataFrame({"date": dates, "equity": np.asarray(vals, dtype=float)})


# ─── paired_gap_bootstrap ───────────────────────────────────────────────

def test_paired_gap_positive_when_model_dominates():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    base_r = rng.normal(0.0003, 0.01, 400)
    # model = baseline returns + a steady daily edge -> gap clearly > 0
    model_r = base_r + 0.001
    m_eq = _equity(dates, 100 * np.cumprod(1 + model_r))
    b_eq = _equity(dates, 100 * np.cumprod(1 + base_r))
    out = st.paired_gap_bootstrap(m_eq, b_eq, n_resamples=500, block_size=20)
    lo, med, hi = out["gap_ci"]
    assert lo > 0 and hi > 0, f"gap CI should be all-positive: {out['gap_ci']}"
    assert out["p_value_gap_le_0"] < 0.05
    assert out["gap_annualized"] > 0


def test_paired_gap_straddles_zero_when_equal():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1)
    r = rng.normal(0.0003, 0.01, 400)
    eq = _equity(dates, 100 * np.cumprod(1 + r))
    out = st.paired_gap_bootstrap(eq, eq.copy(), n_resamples=500, block_size=20)
    # identical curves -> daily diff is exactly 0 -> gap 0, p(gap<=0) == 1.0
    assert abs(out["gap_annualized"]) < 1e-9
    assert out["p_value_gap_le_0"] == pytest.approx(1.0)


def test_paired_gap_aligns_by_date():
    # baseline starts 50 bars later; only the overlapping dates are paired
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(2)
    r = rng.normal(0.0003, 0.01, 400)
    full = 100 * np.cumprod(1 + r)
    m_eq = _equity(dates, full)
    b_eq = _equity(dates[50:], full[50:])  # same values on shared dates
    out = st.paired_gap_bootstrap(m_eq, b_eq, n_resamples=300, block_size=20)
    assert out["n_obs"] == 349  # 350 shared rows -> 349 daily returns
    assert abs(out["gap_annualized"]) < 1e-9  # identical on shared dates


def test_paired_gap_short_input_returns_empty():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    eq = _equity(dates, [100, 101, 102])
    out = st.paired_gap_bootstrap(eq, eq.copy(), n_resamples=100)
    assert out["n_obs"] == 0
    assert out["gap_ci"] == (0.0, 0.0, 0.0)


# ─── settle_picks ───────────────────────────────────────────────────────

def _load_settle():
    spec = importlib.util.spec_from_file_location(
        "settle_picks", PROJECT / "scripts" / "settle_picks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_realized_return_resolvable():
    sp = _load_settle()
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    closes = 100.0 + np.arange(100)  # strictly increasing
    df = pd.DataFrame({"date": dates, "close": closes})
    res = sp.realized_return(df, dates[10], horizon=40)
    assert res is not None
    realized, exit_date, entry_close, exit_close = res
    assert entry_close == closes[10]
    assert exit_close == closes[50]
    assert realized == pytest.approx(closes[50] / closes[10] - 1)
    assert pd.Timestamp(exit_date) == dates[50]


def test_realized_return_not_yet_resolved():
    sp = _load_settle()
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame({"date": dates, "close": 100.0 + np.arange(100)})
    # anchor at idx 70, horizon 40 -> exit_idx 110 > len -> unresolved
    assert sp.realized_return(df, dates[70], horizon=40) is None


def test_realized_return_anchor_absent():
    sp = _load_settle()
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame({"date": dates, "close": 100.0 + np.arange(50)})
    assert sp.realized_return(df, pd.Timestamp("1999-01-01"), horizon=40) is None


def test_settle_all_idempotent(tmp_path):
    sp = _load_settle()
    adjusted = tmp_path / "adjusted"; adjusted.mkdir()
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    pd.DataFrame({"date": dates, "close": 100.0 + np.arange(100),
                  "volume": np.full(100, 1e6)}).to_parquet(adjusted / "AAA.parquet", index=False)
    picks_dir = tmp_path / "picks"; picks_dir.mkdir()
    pd.DataFrame([{"ticker": "AAA", "anchor_date": dates[10].date().isoformat(),
                   "anchor_close": 110.0, "adv20_usd": 1e8, "score": 0.9,
                   "pick_generated": "2024-01-15", "horizon_days": 40,
                   "track": "lgbm_engineered"}]).to_csv(picks_dir / "2024-01-15.csv", index=False)
    realized = tmp_path / "realized" / "realized.csv"

    s1 = sp.settle_all(picks_dir, realized, adjusted, horizon=40)
    assert s1["newly_settled"] == 1
    assert realized.exists()
    # Second run settles nothing new (idempotent) and leaves no .tmp behind.
    s2 = sp.settle_all(picks_dir, realized, adjusted, horizon=40)
    assert s2["newly_settled"] == 0
    assert not realized.with_name(realized.name + ".tmp").exists()
    assert len(pd.read_csv(realized)) == 1


# ─── data_hashes ────────────────────────────────────────────────────────

def _make_parquets(d: Path, n=3):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                      "close": [1.0 + i] * 5}).to_parquet(d / f"T{i}.parquet", index=False)


def test_write_and_verify_clean(tmp_path):
    adj = tmp_path / "adjusted"; _make_parquets(adj, 3)
    manifest = tmp_path / "data_hashes.json"
    payload = mf.write_data_hashes(adj, manifest)
    assert payload["file_count"] == 3
    assert manifest.exists()
    r = mf.verify_data_hashes(adj, manifest)
    assert len(r["ok"]) == 3
    assert r["changed"] == [] and r["missing"] == [] and r["added"] == []


def test_verify_detects_drift(tmp_path):
    adj = tmp_path / "adjusted"; _make_parquets(adj, 3)
    manifest = tmp_path / "data_hashes.json"
    mf.write_data_hashes(adj, manifest)
    # Mutate one file (yfinance restatement), add one, remove one.
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                  "close": [999.0] * 5}).to_parquet(adj / "T0.parquet", index=False)
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                  "close": [7.0] * 5}).to_parquet(adj / "T9.parquet", index=False)
    (adj / "T2.parquet").unlink()
    r = mf.verify_data_hashes(adj, manifest)
    assert r["changed"] == ["T0.parquet"]
    assert r["missing"] == ["T2.parquet"]
    assert r["added"] == ["T9.parquet"]


def test_verify_missing_manifest(tmp_path):
    adj = tmp_path / "adjusted"; _make_parquets(adj, 1)
    r = mf.verify_data_hashes(adj, tmp_path / "nope.json")
    assert "error" in r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
