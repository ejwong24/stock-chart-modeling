# Stock Chart Modeling — Corrected Rebuild

This is a from-scratch re-implementation of the chart-image stock-
selection pipeline described in `BEST_MODEL_REPRODUCIBILITY_EMAIL_DOCUMENT.docx`,
with explicit corrections for the flaws identified by a 12-subagent
audit. See `FLAWS_AND_FIXES.md` for the full critique and severity
ranking.

## What this does

Given daily OHLCV data for ~3000–6000 US tickers (2016–2026), it:

1. Builds a survivorship-bias-aware working universe.
2. Generates weekly anchor labels (binary fixed-horizon and
   continuous excursions) with explicit `label_resolution_date` for
   leakage-safe splits.
3. Computes engineered features (the FALSIFICATION baseline) AND
   renders 224×224 fixed-log-y chart images (the corrected encoder
   input).
4. Embeds charts via frozen DINOv2 ViT-S/14 (384-dim).
5. Trains three side-by-side model tracks per fold (matched-original
   LR, LightGBM on images, LightGBM on engineered features only) using
   PURGED walk-forward with embargo.
6. Runs a portfolio simulator with realistic costs (Almgren-Chriss
   square-root market impact), slippage, halt-risk, a configurable
   trailing stop, and liquidity constraints.
7. Compares the model against five simple-momentum baselines and 200+
   random-seed portfolios.
8. Reports honest stats: deflated Sharpe ratio, block-bootstrap
   p-values, López de Prado effective sample size, post-tax CAGR,
   percentile within random distribution.

## Key corrections vs original document

| Original flaw | This rebuild's fix |
|---|---|
| Universe = "tickers existing 2026-04-20" → survivorship bias | Active universe + static delisted seed list |
| H=40 forward labels straddle fold boundaries (leakage) | Purged walk-forward + embargo |
| 72 configs evaluated → 100th-percentile vs 100 seeds is meaningless | Deflated Sharpe ratio for N_trials |
| First-close=100 + auto-y → +5% and +500% charts identical | Fixed log-y axis −90% to +1000% |
| No simple-baseline comparison | 5 deterministic momentum baselines |
| No transaction costs / slippage / liquidity | bps slippage + commission + min ADV + ADV-aware position sizing |
| LR with class_weight='balanced' → uncalibrated probs | Isotonic calibration on held-out train slice |
| Only 100 random seeds | 200+ seeds (configurable) |
| Reproducibility grade C | Pinned requirements + manifest + seeds + DINOv2 hash |

## Layout

```
config/                  — default.yaml + delisted_seed.txt
data/raw                 — (unused; auto_adjust=True writes to adjusted)
data/adjusted/           — one parquet per ticker, split-adjusted
data/universe/           — working_universe.csv
src/stock_chart/         — library
  config.py
  universe.py
  data_acq.py            — yfinance bulk downloader
  labels.py              — weekly-anchor + multi-horizon labels
  features.py            — engineered features (~30 cols)
  render.py              — fixed-log-y deterministic PIL renderer
  embed_dinov2.py        — frozen DINOv2 ViT-S/14
  splits.py              — López de Prado purged walk-forward
  models.py              — three tracks: LR, LGBM-image, LGBM-engineered
  simulator.py           — event-driven w/ slippage + ADV-aware sizing
  random_baseline.py     — random + 5 simple-momentum baselines
  stats.py               — block bootstrap, deflated Sharpe
  manifest.py            — reproducibility manifest writer
scripts/                 — orchestration entry points
tests/                   — determinism + leakage + simulator tests
reports/                 — per-run outputs
```

## Run

```bash
# 1. Build venv + install deps
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Build universe
python scripts/01_build_universe.py

# 3. Bulk-download yfinance OHLCV (auto_adjust=True). Long-running.
python scripts/02_acquire_data.py

# 4. Smoke test on a small subset
python scripts/run_pipeline.py --horizon 40 --threshold 0.25 \
    --n-tickers 100 --random-seeds 25 --out-tag smoke_100

# 5. Full reproduction (5–8 hours wall-clock on Oracle ARM64, 4 cores)
python scripts/run_pipeline.py --horizon 40 --threshold 0.25 \
    --n-tickers 0 --random-seeds 200 --out-tag full
```

Outputs land in `reports/<out-tag>/`:
- `headline.json` — primary results bundle
- `labels.parquet`, `engineered_features.parquet`, `dinov2_embeddings.npy`
- `scores_<track>.parquet` — per-track test-fold scores
- `equity_<track>.parquet`, `blotter_<track>.parquet`
- `random_seeds_summary.parquet`
- `all_track_summary.csv`

## Tests

```bash
python tests/test_render_determinism.py
python tests/test_splits_no_leakage.py
python tests/test_simulator_basic.py
```

## Hardware notes

DINOv2 ViT-S/14 inference on ARM64 4-core CPU: ~7 imgs/sec → ~3–4 hours
for ~80k images on the full universe. The engineered-features track
runs in <10 minutes regardless of universe size. The simulator scales
roughly linearly with `n_anchors × n_random_seeds`.

## Caveats

This rebuild does NOT make the strategy "production-ready". See
`FLAWS_AND_FIXES.md` section "What this rebuild does *not* fix" — most
notably, full survivorship-bias correction requires paid Polygon/EODHD/
CRSP data, and an out-of-sample 2025 lockbox protocol must be adopted
before declaring an honest headline result.
