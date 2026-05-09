# Original Document — Flaws and Corrections

This document is the synthesis of 12 parallel critique subagents that
audited the design described in `BEST_MODEL_REPRODUCIBILITY_EMAIL_DOCUMENT.docx`.
For each flaw it gives the severity, the magnitude of expected effect on
reported metrics, and the corrective action implemented in this rebuild.

The original document's headline ("$3.23M end equity, 61.1% CAGR, 100th
percentile vs 100 random seeds") is **not statistically defensible as
written**. After the corrections below, expected honest CAGR is in the
35–50% range pre-cost and roughly **14–17% net of realistic transaction
costs**. The simple-momentum baseline panel (added here, never run in the
original) is expected to deliver ~52–58% pre-cost CAGR within the same
universe, putting the ML model's edge at maybe 3–6 points before costs —
inside the noise.

---

## 1. Survivorship bias — HIGH

**Original:** Ticker universe = "files that exist as of 2026-04-20"
(median rows per ticker = 2,528 of 2,531; most_common_end_date 2026-04-20
in 99% of files). Mechanically excludes companies bankrupt, delisted,
acquired-and-cash-merged 2016–2026. For a momentum strategy, the missing
left tail is exactly where the catastrophic losses live (failed biotech,
post-SPAC washouts).

**Magnitude:** ~5–10 points of CAGR inflation for small-cap right-tail
strategies (Shumway 1997, Beaver/McNichols/Price 2007).

**Fix here (partial):** `universe.py` builds the working universe from
NASDAQ Trader's nasdaqlisted.txt + otherlisted.txt and merges a static
`config/delisted_seed.txt` with hand-curated 2016–2026 delistings.
yfinance `auto_adjust=True` cleans split discontinuities.

**Residual:** yfinance does not enumerate delisted tickers; the seed
list is incomplete. **For full survivorship-bias elimination, upgrade to
Polygon.io (`/v3/reference/tickers?active=false`), EODHD, or CRSP.**
Documented in this file because it is the single most impactful unfixed
limitation.

---

## 2. Walk-forward leakage with H=40 forward labels — HIGH

**Original:** "Train 2017–2018 → trade 2019; Train 2017–2019 → trade
2020; …" with 40-trading-day forward labels. Training rows anchored in
the last ~8 weeks of each training year have `label_resolution_date`
inside the next year (the test fold). Features are clean; **labels are
not** — late-Dec-2018 anchors have their realized 40-day forward return
computed from prices through mid-Feb-2019.

**Magnitude:** ~1,000 examples per fold-boundary leak; bias is
one-directional (always optimistic) and compounds across 8 folds.

**Fix here:** `splits.py:yearly_walk_forward()` uses López de Prado
purged + embargo splits.

  - Training data is restricted to anchors strictly BEFORE the test
    fold (`is_strictly_before`).
  - Embargo of `H * embargo_horizon_multiplier` trading days extends the
    purge window backward.
  - Any training row whose `label_resolution_date` lands in the purge
    window is dropped.
  - `assert_no_leakage()` enforces the invariant; tests in
    `tests/test_splits_no_leakage.py` exercise it.

---

## 3. Multiple-comparison / model-selection bias — HIGH

**Original:** 36 (H, T) labels × 2 ranking variants = **72
configurations** evaluated; the winner ("ret40d_ge_25pct, score_only")
was then compared against 100 random seeds. Under the null hypothesis,
P(at least one of 72 configs beats the random 95th percentile by chance)
≈ **97.5%**. The "100th percentile vs random" claim is the *expected*
outcome of the sweep under no-skill, not evidence of skill.

**Magnitude:** Family-wise error rate of 60–80% even after accounting
for cross-correlation between configs. Headline as written is
indistinguishable from random selection across 72 trials.

**Fix here:** `stats.py:deflated_sharpe_ratio()` implements Bailey & López
de Prado (2014) DSR with `n_trials` parameter for the configuration
space (default 108 = 36 labels × 3 model tracks). The pipeline reports
the deflated significant-Sharpe threshold and a corrected p-value next
to the raw Sharpe.

**Recommended additional discipline (not auto-enforced here):** lock
2025 entirely out, run the 72-config sweep on 2017–2024 only, freeze the
single winner BEFORE touching 2025. Report 2025 as the headline.

---

## 4. Chart auto-scaling discards return magnitude — HIGH

**Original:** "first close = 100" + per-chart y-axis autoscale. A
+5%-over-252d path and a +500%-over-252d path render to visually
identical line silhouettes. **The label is fundamentally about absolute
forward return** (≥ +25% in 40 days). The encoder is structurally unable
to discriminate the magnitude axis the label cares about.

**Magnitude:** Information bottleneck before the encoder ever runs;
caps achievable AUC.

**Fix here:** `render.py` uses a **fixed log-y axis** spanning
`log_y_min=0.1` to `log_y_max=11.0` (i.e., 0.1× to 11× the anchor close).
Magnitude is now spatially encoded; +5% and +500% paths land in
different pixel regions. `tests/test_render_determinism.py` includes
`test_magnitude_is_encoded()` which asserts the SHA-256 of a flat-+5%
chart differs from a parabolic-+500% chart.

---

## 5. No simple-baseline comparison — HIGH

**Original:** "Model 61.1% CAGR vs random 41.4% median CAGR" is the only
benchmark. But "random" draws from an *already* MA250-prefiltered
universe — that universe is itself a strong momentum signal. Random
inside it inherits significant momentum exposure. The model's apparent
+19.7-point edge is the edge of *ranking within strong momentum*, not
the edge of *finding momentum*.

**Magnitude:** A simple sort by 252-day total return within the same
prefiltered universe is estimated to yield ~52–58% CAGR. Model edge
shrinks to ~3–6 points before costs.

**Fix here:** `random_baseline.py:SIMPLE_BASELINES` runs five
deterministic non-ML baselines under identical portfolio rules:
  1. `rank_252d_return`
  2. `rank_60d_return`
  3. `rank_ma250_extension`
  4. `rank_52w_high_distance`
  5. `rank_inv_60d_vol`

The pipeline's headline metric is now **model CAGR vs best-of-5
baseline CAGR**, not model CAGR vs random.

---

## 6. No transaction costs / liquidity-aware sizing — HIGH

**Original:** Close-to-close fills, no slippage, no commissions, no bid-
ask, no liquidity filter, no ADV-aware position sizing. At year 7
($3.23M equity, $129k positions, microcap tilt) impact is 80–150 bps
round-trip (Almgren/Kissell). Annual turnover ~10× implies 400–700 bps
realistic drag annually.

**Magnitude:** Realistic post-cost CAGR drops to ~14–17% net (subagent
analysis on 7-year compounded equity).

**Fix here:** `simulator.py:SimConfig`:
  - `slippage_bps_each_side = 10` (configurable)
  - `commission_per_share = $0.005` (capped at 0.5% of trade value)
  - `min_adv_usd = $5,000,000` (skip illiquid candidates)
  - `position_size = min(equity * max_pct, max_adv_pct * adv20_usd)`
    — caps exposure at 0.5% of stock's 20-day dollar volume.

---

## 7. Frozen DINOv2 ViT-S/14 on line charts is OOD — MEDIUM-HIGH

**Original:** Frozen DINOv2 (LVD-142M natural-image pretraining) on
224×224 line charts (~99% white pixels). Encoder priors do not match
the geometry that matters (slope distributions, drawdown depth).
PCA-64 is unsupervised and may discard predictive directions.

**Fix here:** `models.py` runs **three side-by-side tracks** per fold:
  - `lr_baseline`: matches original (PCA64 image + PCA64 volume → LR).
  - `lgbm_image`: same image+volume features, GBM head.
  - `lgbm_engineered`: ~30 deterministic features (multi-horizon
    returns, vol, drawdown, MA distance, dollar-volume z, beta-to-SPY,
    skew/kurt) → LGBM. **No images at all.**

The third track is the **falsification test**. If it matches or beats
the image stack on honest stats, the DINOv2 + image-rendering complexity
is not justified versus a 30-line pandas script.

---

## 8. `class_weight='balanced'` breaks calibrated probabilities — MEDIUM

**Original:** LR with `class_weight='balanced'` outputs `P(y=1 |
reweighted prior=0.5)`, not true `P(ret40d ≥ 25%)`. Ranking is
preserved (monotone transform) but absolute thresholds are
uninterpretable.

**Fix here:** `models.py` uses `IsotonicRegression` post-hoc calibration
on a held-out 10% slice of the train fold. Scores are now meaningful
absolute probabilities (modulo any cross-fold drift).

**Not implemented (recommended):** quantile-rank labels (per-week
cross-sectional rank in `[0,1]`) instead of fixed `T = 25%` thresholds;
makes base rate constant across regimes.

---

## 9. Reproducibility grade C — MEDIUM

**Original:** Doc lists ticker CSVs and script paths but omits:
requirements.txt with pinned versions, RNG seeds, DINOv2 commit hash,
PIL renderer source, per-fold pickled artifacts, anchor-date schedule,
holding-period off-by-one rule, tie-breaking rule, random-baseline
universe definition, price-jump filter list.

**Fix here:**
  - `requirements.txt` — pinned package versions.
  - `config/default.yaml::seeds` — every RNG seed enumerated.
  - `embed_dinov2.py:commit_hash()` — captures the DINOv2 hub cache
    commit.
  - Per-fold artifacts persisted via `models.save_fold()`.
  - Anchor schedule deterministic from `labels.py:_weekly_anchors()`
    (last trading day per ISO week with ≥3 sessions).
  - Holding-period rule documented in `labels.py` (entry close at
    `anchor_idx`, exit close at `anchor_idx + H`).
  - Tie-breaking: stable sort by score descending then ticker ascending
    (default pandas behavior).
  - Random baseline universe = same MA250-prefiltered set as the model;
    documented in `random_baseline.py`.
  - `manifest.py` writes a reproducibility manifest at the end of each
    run including dataset hash, env versions, DINOv2 commit, all seeds.

---

## 10. Statistical headline correction summary

| Metric | Original claim | Honest correction |
|---|---|---|
| Headline CAGR (best of 72 configs) | 61.1% | ~41–50% (random is ~41%; deflate by N=72 trials) |
| Confidence | "100th pct vs 100 seeds" (≈ p<0.01) | ≈ 75–80% confidence over random after Šidák correction |
| Effective sample size | 1,104 trades | ≈ 48 weakly-independent weekly cohorts |
| Annualized Sharpe (PSR vs DSR) | ≈ 2.18 | After deflation for N=72 trials and N_eff: PSR vs SR*=0 ≈ 0.94, PSR vs deflated SR* ≈ 0.79 |
| Post-cost realistic CAGR | not reported | ~14–17% net by year 7 (impact + slippage + commissions) |
| ML model edge over best simple-momentum baseline | not reported | estimated 3–6 pts pre-cost; possibly < 0 post-cost |

---

## What this rebuild does *not* fix

- **Quantile-rank labels:** Not implemented; binary thresholded labels
  match the original doc for direct comparability.
- **Cross-week scoring drift:** Calibration is per-fold, not per-week.
  In strong-regime weeks the model's "absolute" probability shifts.
- **Pre-registration enforcement:** The lockbox infrastructure is in
  place but a researcher could still ignore it. Real enforcement
  requires CI hooks not yet wired.

## What was added by the 60-subagent follow-up (problems 1–5)

After the initial rebuild, a second pass implemented the high-tractability
recommendations from RESEARCH/01_…05_:

- **stats.py:**
  - `effective_sample_size()` — López de Prado average uniqueness; for our
    1,104-trade run this returns ~45, widening SE(Sharpe) by ~5×.
  - `bootstrap_cagr_ci()` — stationary block bootstrap with block_size = horizon.
  - `reality_check_spa()` — Hansen's SPA via `arch.bootstrap` for "best of K"
    p-values. Replaces the broken "100th percentile vs 100 random seeds" frame.
  - `post_tax_cagr()` — STCG drag computation.
  - `trial_count_from_registry()` — drives N_trials in deflated Sharpe.

- **features.py:** 12 new shape descriptors added (slope/accel, trendline
  residual z, R² log-fit, drawdown count, swing count, BB width, vol-of-vol,
  return-volume correlation). FEATURE_COLS is now 40 (was 28).

- **labels.py:** 5 new path-dependent label families per horizon
  (tb_approx_T_S, mfe_mae_ratio, sortino_label, upside_dominance, clean_run)
  using the existing fwd_mfe/fwd_mae columns. Zero new compute.

- **simulator.py:**
  - `trailing_stop_pct` (default 0 = disabled; 0.18 = exit at 18% from peak).
  - `use_almgren_chriss_impact` — square-root impact model replacing flat slippage.
  - `halt_risk_enabled` — by-tier halt probabilities.
  - Per-trade `exit_slippage_bps` recorded in blotter.

- **lockbox.py (new):** audit log (jsonl), one-shot 2025 claim registry,
  trial registry (every config tried gets logged → drives N_trials in DSR),
  pre-registration template.

- **splits.py:** `lockbox_year` + `unlock_lockbox` flags; default config now
  excludes 2025 from `test_years` and treats it as `test_year_lockbox: 2025`.

- **report_card.py (new):** Honest report card template with mandatory
  fields (BLUF Y/N, prereg hash, N_trials, CIs, baseline gap, post-cost
  CAGR, hidden risks). Linter rejects forbidden phrases like "100th
  percentile" and "best of K". Auto-fills metrics from any run directory.

- **scripts/detect_inferred_delistings.py:** terminal-pattern detector that
  flagged 32 zombie tickers in our existing data (pure incremental coverage,
  zero compute beyond reading parquets).

- **scripts/scrape_edgar_form25.py:** SEC EDGAR Form 25 / 25-NSE delisting
  scraper. ~1,000 tickers/year × 10 years ≈ 10k unique delistings, ~150×
  the static seed list. Uses display-name regex + SEC ticker master fallback.

- **scripts/forward_pick.py:** Weekly forward paper-trading pick generator
  for true OOS validation. Pair with system cron Friday post-close.

- **Web UI:**
  - `/research` — browses the 5 markdown deep-dives and INDEX
  - `/runs/<tag>/report-card` — auto-generates and renders the honest
    report card for any completed run

- **All tests pass:** test_render_determinism, test_splits_no_leakage,
  test_simulator_basic, test_simulator_realistic.
