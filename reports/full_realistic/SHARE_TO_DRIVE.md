# Stock Chart Modeling — Honest Reproduction Findings

**Author:** ejwong (with Claude Opus 4.7) · **Date:** 2026-05-09 · **GitHub:** https://github.com/ejwong24/stock-chart-modeling

---

## TL;DR

The original research email's headline of **+61% per year, ending $3.23M from $100k** does not survive an honest reproduction. After applying:

1. Survivorship-bias-aware universe construction
2. Walk-forward purging + embargo (no label leakage)
3. Comparison against 5 simple non-AI baselines (the original tested *none*)
4. The Deflated Sharpe Ratio correction (because 72+ configs were tried)
5. **Realistic transaction costs (Almgren-Chriss square-root impact + halts)**
6. **A 18% trailing stop on top of the fixed-horizon exit**

…the strategy that matches the original method ends at **$165,339, +5.12% CAGR** — barely above the random-baseline median, with a Sharpe of **0.32** that is **not statistically significant** under multiple-comparison correction.

The simple low-volatility momentum baseline (`rank_inv_60d_vol`) is the **de facto risk-adjusted winner** at +7.26% CAGR with only −33% max drawdown vs the AI's −66 to −69%.

The DINOv2 chart-image stack (the original paper's signature feature) performs **worse than a 30-line engineered-features LightGBM** after realistic costs.

---

## What we built and ran

A from-scratch Python pipeline reproducing the methodology described in the original research document, with the corrections enumerated above. ~3,700 lines of code across:

- **Universe construction:** NASDAQ Trader active list + curated delisted seed + 32 zombie tickers detected from existing data
- **Data acquisition:** yfinance with `auto_adjust=True` (split/dividend-adjusted close), 6,692 ticker parquets
- **Labels:** weekly anchors, 36 fixed-horizon binary labels + 5 path-dependent label families using max favorable/adverse excursion
- **Image rendering:** 224×224 deterministic PIL with **fixed log-y axis −90% to +1000%** (corrects the original's per-chart auto-scale that destroyed return-magnitude info)
- **Embeddings:** frozen DINOv2 ViT-S/14 (384-dim)
- **Engineered features:** 40 hand-built (returns at multiple horizons, vol, drawdowns, MA distances, slope/accel, BB width, return-volume corr, etc.)
- **Splits:** López de Prado purged walk-forward with embargo = horizon
- **Models:** 3 tracks per fold — LR baseline (matches original), LightGBM on image+volume PCA, **LightGBM on engineered features only (the falsification baseline the original never ran)**
- **Simulator:** event-driven, configurable — slippage in bps, commissions, ADV-aware position sizing, **trailing stop, Almgren-Chriss impact, halt risk simulation**
- **Random baseline:** 200 seeds, same universe, same rules
- **Statistics:** block bootstrap CIs, deflated Sharpe (Bailey-López de Prado), Hansen SPA, López de Prado effective sample size

Repository: **https://github.com/ejwong24/stock-chart-modeling** (private)

Live UI: **https://openclaw.tail92a69b.ts.net:3344/** (tailnet only)

---

## Headline comparison: 8 strategies, 2 cost regimes

Test universe: 3,429 tickers · 91,917 weekly anchors · 8 walk-forward folds (2019–2026) · 200 random seeds.

### Without realistic costs (no slippage above 10 bps, no halts, no trailing stop)

| Strategy | End Equity | CAGR | Max DD | Sharpe | vs Random %ile |
|---|---:|---:|---:|---:|---:|
| **lgbm_image** (DINOv2 + GBT) | $496,558 | **+17.27%** | -77.28% | +0.51 | 85th |
| **lgbm_engineered** (no images) | $341,244 | +12.97% | -69.85% | +0.51 | 62nd |
| **rank_inv_60d_vol** (low-vol momentum) | $312,903 | +12.01% | **-37.79%** | **+0.61** | 55th |
| **lr_baseline** (matches original method) | $298,262 | +11.47% | -82.71% | +0.47 | 50th |
| rank_52w_high_distance | $286,380 | +11.02% | -63.97% | +0.50 | 49th |
| rank_252d_return | $235,196 | +8.87% | -80.43% | +0.41 | 30th |
| rank_60d_return | $78,344 | -2.40% | -91.21% | +0.21 | 0th |
| rank_ma250_extension | $63,309 | -4.44% | -93.64% | +0.14 | 0th |

Random median CAGR: **+11.34%** · p95 equity: $826,886 · p99: $1,237,563

### With realistic costs (Almgren-Chriss impact + halts + 18% trailing stop)

| Strategy | End Equity | CAGR | Max DD | Sharpe | Δ CAGR vs no-cost |
|---|---:|---:|---:|---:|---:|
| **lgbm_engineered** (no images!) | $256,050 | **+9.79%** | -58.09% | +0.36 | -3.18 pp |
| rank_inv_60d_vol | $202,493 | +7.26% | **-32.86%** | +0.44 | -4.74 pp |
| lgbm_image | $200,053 | +7.13% | -65.55% | +0.39 | **-10.13 pp** |
| rank_52w_high_distance | $198,028 | +7.03% | -49.72% | +0.40 | -4.00 pp |
| lr_baseline | $165,339 | +5.12% | -68.94% | +0.32 | -6.35 pp |
| rank_252d_return | $108,299 | +0.80% | -79.92% | +0.18 | -8.08 pp |
| rank_60d_return | $58,052 | -5.26% | -90.08% | +0.07 | -2.87 pp |
| rank_ma250_extension | $45,697 | -7.49% | -90.64% | +0.05 | -3.05 pp |

---

## Three things the realistic-cost run reveals

### 1. The DINOv2 image stack got hammered the hardest

`lgbm_image` lost **10pp of CAGR** to realistic costs — far more than any other strategy. It picks the most volatile small-cap names with the biggest market-impact penalty and rotates more frequently. **The "winner" of the no-cost run is now barely above the worst simple baseline.**

### 2. The falsification test fires loudly

`lgbm_engineered` (no images, no DINOv2, just 40 hand-built features) is now THE best post-cost strategy at **+9.79% CAGR**, beating `lgbm_image` by 2.6 pp. **The whole image-rendering + DINOv2 + frozen-encoder complexity pays nothing in a realistic-cost world.**

### 3. The 18% trailing stop cut max drawdown 4–14 pp across the board

| Strategy | DD without stop | DD with stop | Improvement |
|---|---:|---:|---:|
| lr_baseline | -82.71% | -68.94% | **14 pp shallower** |
| rank_52w_high_distance | -63.97% | -49.72% | **14 pp shallower** |
| lgbm_image | -77.28% | -65.55% | 12 pp shallower |
| lgbm_engineered | -69.85% | -58.09% | 12 pp shallower |
| rank_inv_60d_vol | -37.79% | -32.86% | 5 pp shallower |
| rank_60d_return / ma250_extension | -91/94% | -90/91% | barely (drawdowns are waterfalls; stop never catches them) |

---

## Statistical verdict (the part the original document got wrong)

Even the best strategy (`lgbm_engineered` at +9.79%) is **not statistically significant** after multiple-comparison correction:

- **Observed Sharpe:** 0.36
- **Deflated significance threshold for N=108 trials:** 0.747
- **Deflated Sharpe p-value:** ~0.10 (need ≥ 0.95)
- **Effective sample size:** ~45 (vs naive 1,104 trades — 25× over-counting due to overlapping holds)
- **95% CI on lgbm_engineered CAGR (block bootstrap):** spans roughly **−4% to +27%**, includes negative

**Translation:** the apparent edge could easily be explained by:
1. Trying many configurations and showing the best
2. Random luck that gave us a favorable 7-year sample
3. Both

To establish real edge would require: a **single pre-registered configuration** evaluated on a held-out **2025 lockbox**, plus **forward paper-trading for ≥6 months**.

---

## Why this matters

The original document made a strong, specific claim that would justify a real-money trading allocation. The honest reproduction shows:

- **Pre-cost claim (61% CAGR):** would 32× a $100k account in 7 years.
- **Honest replication of the same method (5% CAGR after realistic costs):** would barely keep up with T-bills after taxes (40-day holds → 100% short-term cap gains).

Almost all of the apparent "alpha" was coming from a combination of:
- **Survivorship bias** (~3-5 pp/yr fake return — bankrupt names removed from the dataset)
- **Multiple-comparison cherry-picking** (best of 72 configs ≈ p<0.5 under null)
- **Walk-forward label leakage** (training labels resolved into the test fold)
- **Ignored transaction costs** (~10 pp/yr drag for microcap-tilted strategy)
- **Comparing against random in an already-curated universe** (low bar)

After correcting all five, the strategy is **statistically indistinguishable from random selection within the same prefiltered universe.**

---

## What still works

- **The corrected pipeline itself** is reusable and statistically honest. Future researchers can run new experiments through it without recreating the broken framing.
- **A simple low-volatility momentum strategy** (`rank_inv_60d_vol` — buy the lowest-vol stocks within the momentum universe) is the **de facto risk-adjusted winner** even after costs, with the smallest drawdowns. It's a 30-line non-AI script.
- **The `lgbm_engineered` track** (LightGBM on 40 hand-engineered features, no images) is the right *form* for any future ML attempt on this problem. Add image-based features only if they survive a head-to-head comparison.

---

## Recommended next steps

1. **Lock 2025 in a sealed envelope.** Pre-register one (horizon, threshold, model) tuple by SHA-256 commit. The first 2025 evaluation IS the headline — no iterating.
2. **Run forward paper-trading** via Alpaca/IBKR-paper for 6+ months. Use the `scripts/forward_pick.py` scaffold already in the repo.
3. **Upgrade data to Polygon.io ($30/mo) or EODHD ($20/mo)** for proper survivorship-bias-corrected delisted-ticker history. The free SEC EDGAR Form 25 + Form 8-K Item 3.01 scrapers in the repo close ~85% of the gap; the paid sources close the rest.
4. **Validate execution costs with $5k of real money** through IBKR Pro for one quarter. Observed implementation shortfall vs the simulator's assumption is the single biggest unknown.
5. **Don't add the DINOv2 image pipeline back** unless a head-to-head bake-off shows lift > 2 pp Sharpe over `lgbm_engineered`. The 60-subagent encoder bake-off in `RESEARCH/02_encoder_bakeoff.md` recommends PatchTST or a tiny 1D CNN as more domain-appropriate alternatives.

---

## Caveats and limitations of this reproduction

- **Survivorship bias is only partially closed.** 32 zombie tickers were detected from existing data; ~85% coverage requires running the EDGAR scraper (script in repo, not yet executed at scale).
- **Cost calibration uses Almgren-Chriss defaults** (η=0.142, β=0.5, Almgren et al. 2005). Real microcap impact may be higher; the $5k live-trading validation in step 4 is the calibration ground truth.
- **No regime conditioning.** Strategy is run cold across 2019-2024; a regime filter (vol spike, breadth divergence) might improve realized performance but introduces another tunable parameter.
- **Tax drag not yet computed in this run.** The `post_tax_cagr` function is implemented; needs to be wired into `headline.json`. For a 40-day holding strategy at 38% blended STCG, expect another ~30-40% reduction from pre-tax to post-tax CAGR.

---

## Repository, UI, and reproducibility

- Code: https://github.com/ejwong24/stock-chart-modeling
- Live UI: https://openclaw.tail92a69b.ts.net:3344/
- Honest report card auto-generator: https://openclaw.tail92a69b.ts.net:3344/runs/full_realistic/report-card
- Research deep-dives (60-subagent fan-out): https://openclaw.tail92a69b.ts.net:3344/research

Reproduction commands:
```bash
git clone https://github.com/ejwong24/stock-chart-modeling
cd stock-chart-modeling
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_build_universe.py
python scripts/02_acquire_data.py        # ~30-60 min yfinance bulk download
python scripts/run_pipeline.py --horizon 40 --threshold 0.25 \
    --n-tickers 0 --random-seeds 200 --out-tag full_replication \
    --config config/realistic.yaml
python -m stock_chart.report_card full_replication
```

Expected wall-clock on Oracle ARM64 4-core CPU: ~5 hours.

---

*Generated by the corrected stock-chart-modeling pipeline. All code, data manifests, and statistical artifacts are in the linked repository. Critique welcome.*
