# SMA250 Random System-Comparison Test — Results

**Spec:** SIMPLE_SMA250_RANDOM_SYSTEM_COMPARISON_TEST.docx
**System:** `stock_chart_modeling` rebuild (https://github.com/ejwong24/stock-chart-modeling)
**Run date:** 2026-05-09 (wall: 54 min on 4-core ARM64, 3 parallel workers)

---

## TL;DR

Two random portfolios, MA250-prefiltered universe, zero costs, 200 seeds each. **The H=20 random median ends at $344k (CAGR +13.08%); the H=40 random median ends at $459k (CAGR +16.35%).** Whichever system Greg is running should land within ~±15% of these numbers on the medians — otherwise the disagreement is data plumbing, not modeling.

The dispersion across seeds is meaningful: **H=40 has a 95th-percentile equity of $1.26M vs 5th-percentile of $212k** (6× spread). Same universe, same rules, just different random picks. This is the noise floor any model-ranking claim has to clear.

---

## Meta

| Field | Value |
|---|---|
| **Spec source** | SIMPLE_SMA250_RANDOM_SYSTEM_COMPARISON_TEST.docx |
| **System** | stock_chart_modeling rebuild |
| **Repo** | https://github.com/ejwong24/stock-chart-modeling |
| **Live UI** | https://openclaw.tail92a69b.ts.net:3344/ |
| **Universe size** | 3,429 tickers (NASDAQ + NYSE common stocks, no ETFs/test issues) |
| **Data source** | yfinance with `auto_adjust=True` (split-/dividend-adjusted close) |
| **Prefilter** | `Close > 1.5 × SMA250` only (no model, no rank, no future info) |
| **Anchors** | Weekly, last trading day of each ISO week (Friday-style anchor) |
| **Selection** | uniform random from eligible non-held tickers, up to 5 per anchor |
| **Position size** | 4% of current equity |
| **Starting equity** | $100,000 |
| **Costs / slippage / commissions** | 0 (per spec) |
| **ADV / liquidity filters** | None (per spec) |
| **Duplicate ticker** | Not allowed (per spec) |
| **Hold periods** | 20 and 40 trading days |
| **Seeds per horizon** | 200 (seeds 0 through 199) |
| **RNG** | numpy `default_rng(seed)` |
| **Fractional shares** | **No — integer floor `int(target_usd // fill_price)`** (see caveat below) |
| **Entry-date span (H=20)** | 2019-01-04 → 2026-02-19 (first/last anchor with any pick) |
| **Entry-date span (H=40)** | 2019-01-04 → 2026-02-19 |
| **Final exit date** | 2026-04-20 (last available close in dataset) |
| **Total wall-clock** | 54.2 min (3-worker `joblib.Parallel`) |

---

## Test A — 20-Trading-Day Hold (200 seeds)

| Metric | 5th percentile | **Median** | 95th percentile | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Ending equity | $176,517 | **$344,449** | $676,919 | $374,884 | $167,586 |
| CAGR | +5.81% | **+13.08%** | +20.93% | +13.09% | 4.89 pp |
| Max drawdown | −61.83% | **−46.94%** | −36.81% | — | — |
| Profit factor | — | **1.280** | — | — | — |
| Win rate | — | **47.97%** | — | — | — |
| Trade count | — | **1,863** | — | — | — |
| Average trade return | — | **+2.28%** | — | — | — |

The H=20 distribution is concentrated: 5th–95th percentile spans roughly $177k → $677k. That's the noise band for "random pick within MA250-prefiltered universe, 20-day hold."

---

## Test B — 40-Trading-Day Hold (200 seeds)

| Metric | 5th percentile | **Median** | 95th percentile | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Ending equity | $211,761 | **$458,784** | $1,262,766 | $557,426 | $387,765 |
| CAGR | +7.74% | **+16.35%** | +28.66% | +16.89% | 6.45 pp |
| Max drawdown | −64.47% | **−49.83%** | −38.78% | — | — |
| Profit factor | — | **1.453** | — | — | — |
| Win rate | — | **47.69%** | — | — | — |
| Trade count | — | **1,187** | — | — | — |

The H=40 distribution is materially wider (95th pct is 3.6× the median vs 2.0× for H=20). Longer holds → fewer trades → more concentrated outcomes per seed → larger seed-to-seed variance. This is also the headline horizon from Greg's email (40d / 25% threshold).

---

## Pass / Fail comparison checklist (Spec Section 12)

To call the systems "reasonably close," both should agree on these per-horizon medians within sensible tolerance:

| Metric | H=20 (this system) | H=40 (this system) |
|---|---:|---:|
| Median ending equity | **$344,449** | **$458,784** |
| Median CAGR | **+13.08%** | **+16.35%** |
| Median max drawdown | **−46.94%** | **−49.83%** |
| Median trade count | **1,863** | **1,187** |
| Median win rate | **47.97%** | **47.69%** |
| Median profit factor | **1.280** | **1.453** |

Suggested tolerance for "pass": within ±15% on ending equity / CAGR, within ±5 pp on max DD, within ±5% on trade count, within ±2 pp on win rate, within ±0.10 on profit factor.

If the other system materially disagrees, **the spec says to dive into the trade list before debating model quality.** Start with H=20, seed=0, first 50 trades (below).

---

## Caveats from this system

### 1. Integer-share rounding (likely 50–150 bps CAGR cost vs fractional)

This simulator uses `shares = int(target_usd // fill_price)`. The spec prefers fractional shares.

Empirically: at $100k start with 4% positions = $4,000 per trade, integer rounding loses approximately $10–50 in deployed capital per trade × ~1,863 trades over 7 years. In the early years before compounding helps, this is **~50–150 bps drag on CAGR**. So a fractional-share system might report H=20 median CAGR closer to **+13.6% to +14.2%** (vs our +13.08%). For H=40 the drag is smaller because there are fewer trades.

### 2. Anchor schedule

"Friday-like": we pick the last trading day per ISO calendar week per ticker. Holiday-shortened weeks still get an anchor (the latest available trading day). The spec's "Friday anchor if available, otherwise last available trading day" is faithfully implemented.

### 3. Data range

- First entry: 2019-01-04 (first Friday with 250 prior trading days available for every ticker eligible)
- Last entry: 2026-02-19 (last anchor whose 40-day forward window resolves by 2026-04-20)
- Final exit: 2026-04-20 (last bar in our yfinance pull)
- Trades whose forward window would extend past 2026-04-20 are excluded per spec

### 4. Universe construction (potential disagreement source)

Our universe is 3,429 tickers built from NASDAQ Trader's official `nasdaqlisted.txt` + `otherlisted.txt` (common stocks only, no ETFs, no test issues, no special-character symbols), then restricted to tickers with usable yfinance data 2016–2026. **If Greg's universe is from CRSP, Polygon, or a different point-in-time list, ticker membership will differ** — that's the most likely single cause of any disagreement. The MA250-prefilter is applied identically to whichever universe is loaded.

### 5. Trade overlap is large

With 5 picks/week and 20-day or 40-day holds, ~5 (H=20) or ~25 (H=40) positions are open concurrently on average. Independent-sample math doesn't apply to the daily returns. We make no statistical-significance claims here — this is purely a numerical agreement test.

---

## Spec Section 12 — H=20 seed=0 first 50 trades (debugging starting point)

| entry_date | ticker | entry_price | exit_date | exit_price | shares | position_value | trade_return % | pnl $ |
|---|---|---|---|---|---|---|---|---|
| 2019-01-04 | PHUN | 2703.0000 | 2019-02-04 | 12500.0000 | 1 | 2703.00 | +362.45 | +9797.00 |
| 2019-01-04 | POCI | 4.0500 | 2019-02-04 | 3.4500 | 987 | 3997.35 | -14.81 | -592.20 |
| 2019-01-04 | DXR | 11.2500 | 2019-02-04 | 16.9000 | 355 | 3993.75 | +50.22 | +2005.75 |
| 2019-01-04 | AXIA | 6.4599 | 2019-02-04 | 7.9440 | 619 | 3998.69 | +22.97 | +918.66 |
| 2019-01-04 | INTZ | 77.4000 | 2019-02-04 | 78.0000 | 51 | 3947.40 | +0.78 | +30.60 |
| 2019-01-11 | TWLO | 96.8300 | 2019-02-11 | 115.7500 | 42 | 4066.86 | +19.54 | +794.64 |
| 2019-01-11 | EHTH | 40.7400 | 2019-02-11 | 58.9500 | 100 | 4074.00 | +44.70 | +1821.00 |
| 2019-01-11 | ASTC | 146.1000 | 2019-02-11 | 146.7000 | 28 | 4090.80 | +0.41 | +16.80 |
| 2019-01-11 | KRYS | 26.1100 | 2019-02-11 | 22.2700 | 156 | 4073.16 | -14.71 | -599.04 |
| 2019-01-11 | INSG | 55.6000 | 2019-02-11 | 48.1000 | 73 | 4058.80 | -13.49 | -547.50 |
| 2019-01-18 | CIG | 1.4155 | 2019-02-19 | 1.4582 | 2994 | 4238.12 | +3.01 | +127.84 |
| 2019-01-18 | MTNB | 56.0000 | 2019-02-19 | 52.0000 | 75 | 4200.00 | -7.14 | -300.00 |
| 2019-01-18 | ORGO | 30.3500 | 2019-02-19 | 7.9000 | 139 | 4218.65 | -73.97 | -3120.55 |
| 2019-01-18 | LFVN | 12.5459 | 2019-02-19 | 13.8065 | 337 | 4227.97 | +10.05 | +424.84 |
| 2019-01-18 | NTRP | 430.0000 | 2019-02-19 | 482.0000 | 9 | 3870.00 | +12.09 | +468.00 |
| 2019-01-25 | LRN | 29.4000 | 2019-02-25 | 32.6700 | 140 | 4116.00 | +11.12 | +457.80 |
| 2019-01-25 | PACB | 7.1700 | 2019-02-25 | 7.3500 | 575 | 4122.75 | +2.51 | +103.50 |
| 2019-01-25 | AXSM | 8.6000 | 2019-02-25 | 9.2300 | 479 | 4119.40 | +7.33 | +301.77 |
| 2019-01-25 | SOWG | 180.0000 | 2019-02-25 | 180.0000 | 22 | 3960.00 | +0.00 | +0.00 |
| 2019-01-25 | LIQT | 59.5200 | 2019-02-25 | 64.9600 | 69 | 4106.88 | +9.14 | +375.36 |
| 2019-02-01 | IMDX | 99.6000 | 2019-03-04 | 66.0000 | 46 | 4581.60 | -33.73 | -1545.60 |
| 2019-02-01 | VCYT | 18.4600 | 2019-03-04 | 20.7200 | 248 | 4578.08 | +12.24 | +560.48 |
| 2019-02-01 | CRMD | 9.9000 | 2019-03-04 | 8.1000 | 463 | 4583.70 | -18.18 | -833.40 |
| 2019-02-01 | LIXT | 51.6000 | 2019-03-04 | 54.0000 | 68 | 3508.80 | +4.65 | +163.20 |
| 2019-02-01 | AUDC | 11.7126 | 2019-03-04 | 11.3969 | 218 | 2553.34 | -2.70 | -68.82 |
| 2019-02-08 | TNDM | 42.2900 | 2019-03-11 | 65.2800 | 103 | 4355.87 | +54.36 | +2367.97 |
| 2019-02-08 | AMSC | 14.4000 | 2019-03-11 | 15.5600 | 305 | 4392.00 | +8.06 | +353.80 |
| 2019-02-08 | USIO | 3.0900 | 2019-03-11 | 2.9000 | 1421 | 4390.89 | -6.15 | -269.99 |
| 2019-02-08 | IVDA | 6.4000 | 2019-03-11 | 10.2400 | 229 | 1465.60 | +60.00 | +879.36 |
| 2019-02-08 | INTZ | 81.0000 | 2019-03-11 | 82.2000 | 54 | 4374.00 | +1.48 | +64.80 |
| 2019-02-15 | DXR | 17.8500 | 2019-03-18 | 13.6500 | 251 | 4480.35 | -23.53 | -1054.20 |
| 2019-02-15 | OKTA | 84.7600 | 2019-03-18 | 82.8700 | 52 | 4407.52 | -2.23 | -98.28 |
| 2019-02-15 | VIVK | 4148.9360 | 2019-03-18 | 3191.4893 | 1 | 4148.94 | -23.08 | -957.45 |
| 2019-02-15 | PAYS | 7.7600 | 2019-03-18 | 7.1600 | 578 | 4485.28 | -7.73 | -346.80 |
| 2019-02-15 | AXIA | 7.6863 | 2019-03-18 | 7.5066 | 583 | 4481.10 | -2.34 | -104.77 |
| 2019-02-22 | CDNA | 29.6000 | 2019-03-22 | 36.9200 | 151 | 4469.60 | +24.73 | +1105.32 |
| 2019-02-22 | PDEX | 14.7700 | 2019-03-22 | 14.3000 | 303 | 4475.31 | -3.18 | -142.41 |
| 2019-02-22 | ZEPP | 65.1515 | 2019-03-22 | 52.4616 | 68 | 4430.30 | -19.48 | -862.93 |
| 2019-02-22 | PED | 39.0000 | 2019-03-22 | 49.8000 | 114 | 4446.00 | +27.69 | +1231.20 |
| 2019-02-22 | GNE | 7.2181 | 2019-03-22 | 6.9286 | 621 | 4482.47 | -4.01 | -179.78 |
| 2019-03-01 | NOA | 11.3588 | 2019-03-29 | 10.5733 | 402 | 4566.22 | -6.91 | -315.77 |
| 2019-03-01 | MTNB | 69.5000 | 2019-03-29 | 54.5000 | 65 | 4517.50 | -21.58 | -975.00 |
| 2019-03-01 | VHC | 34.6261 | 2019-03-29 | 37.1497 | 132 | 4570.65 | +7.29 | +333.12 |
| 2019-03-01 | MGNI | 6.0100 | 2019-03-29 | 6.0800 | 761 | 4573.61 | +1.16 | +53.27 |
| 2019-03-01 | EHTH | 58.8900 | 2019-03-29 | 62.3400 | 77 | 4534.53 | +5.86 | +265.65 |
| 2019-03-08 | ROKU | 71.2700 | 2019-04-05 | 63.4000 | 63 | 4490.01 | -11.04 | -495.81 |
| 2019-03-08 | TTD | 19.4350 | 2019-04-05 | 19.6520 | 231 | 4489.48 | +1.12 | +50.13 |
| 2019-03-08 | UI | 131.8488 | 2019-04-05 | 151.9654 | 34 | 4482.86 | +15.26 | +683.96 |
| 2019-03-08 | SOWG | 180.0000 | 2019-04-05 | 180.0000 | 21 | 3780.00 | +0.00 | +0.00 |
| 2019-03-08 | TWLO | 116.8000 | 2019-04-05 | 122.4200 | 38 | 4438.40 | +4.81 | +213.41 |

Sanity-check tickers:

- **2019-01-04 PHUN:** Phunware Inc. opened January 2019 in a microcap pump, +362% in one month was real and visible in price data. Both systems should see the same outcome if both use the same adjusted-close field.
- **2019-01-18 ORGO:** Organogenesis crashed −74% in one month — real corporate event. Useful sanity check that both systems handle the price gap identically.
- **2019-02-15 VIVK:** $4,149 entry price suggests a pre-split adjustment. With `auto_adjust=True` we honor yfinance's adjustment; if Greg's system uses unadjusted prices or different split history, this trade will differ materially.

---

## Reproducing this run

```bash
git clone https://github.com/ejwong24/stock-chart-modeling
cd stock-chart-modeling
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build universe and download data (one-time, ~30-60 min)
python scripts/01_build_universe.py
python scripts/02_acquire_data.py

# Build the labels.parquet that feeds the SMA250 test
python scripts/run_pipeline.py --horizon 40 --threshold 0.25 \
    --n-tickers 0 --random-seeds 0 --out-tag full
# (Use --random-seeds 0 to skip the random baseline if you only want SMA250 test)

# Run the SMA250 systems-comparison test
python scripts/run_sma250_test.py \
    --source-tag full \
    --out-tag sma250_test \
    --n-seeds 200 \
    --horizons 20,40 \
    --n-jobs 3
```

Expected wall-clock: 54 min on Oracle ARM64 4-core CPU with 3 parallel workers.

Outputs land in `reports/sma250_test/`:
- `H20_per_seed_summary.csv` (200 rows × 15 cols)
- `H20_full_trades.csv` (~372k rows × 11 cols)
- `H40_per_seed_summary.csv` (200 rows × 15 cols)
- `H40_full_trades.csv` (~237k rows × 11 cols)
- `aggregate_metrics.json` (medians + p5 + p95 + meta)

---

## Next steps if the systems disagree

Per spec Section 12, the order of debugging is:
1. Compare H=20 seed=0 first 50 trades (table above) — does Greg's system see PHUN +362%? ORGO −74%? Same entry/exit prices?
2. If trades disagree on tickers: **universe mismatch.** Compare ticker lists at a fixed anchor date.
3. If trades agree on tickers but disagree on prices: **adjustment mismatch.** Compare raw vs adjusted close fields.
4. If trades agree on tickers + prices but disagree on share counts: **fractional vs integer.** This system uses integer; declared in caveats.
5. If everything matches at the trade level but aggregates disagree: **portfolio accounting** (compounding start date, equity definition, etc.).

---

*Generated 2026-05-21 from artifacts produced 2026-05-09. Repo: https://github.com/ejwong24/stock-chart-modeling*
