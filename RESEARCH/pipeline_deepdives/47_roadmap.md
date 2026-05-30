# Feature gaps & roadmap — what's missing and what to build next

This is the honest accounting page. Everything earlier in the `/story/09` series describes what the pipeline *does*; this page is the inverse — what it does not yet do, what's stubbed, and what's documented-as-future. Each entry is grounded in the current tree (`src/stock_chart/`, `scripts/`, `tests/`), not aspiration.

The table is sorted by value-to-effort ratio. Effort is S/M/L (a few hours / a day or two / a multi-day project). Value is H/M/L for how much the fix moves the central question — *does this model actually beat a simple baseline, out of sample, after costs?*

## The prioritized table

| Gap | Why it matters | Approach | Effort | Value |
|---|---|---|---|---|
| 1. Settle/realize loop for forward picks is unimplemented | Forward paper-trading is the one result that survives every objection — but picks are never scored | Daily cron: 40 trading days after each pick file, look up realized return, append to `data/forward_realized/` | M | H |
| 2. No paired bootstrap on the model-vs-best-baseline GAP | This is *the* claim; `gap_ci` is a TODO string, so the headline gap has no significance | Paired block bootstrap on per-period return differences; fill `gap_ci`/`gap_significant` in `report_card` | M | H |
| 3. No `data_hashes.json` | "Bit-identical repro" only holds if parquet is byte-identical, but yfinance drifts | Commit SHA-256 manifest of `data/adjusted/*.parquet`; flag drift on restore; snapshot to NAS/S3 | S | M |
| 11. SPY-beta doc states a now-false "BNTX beta = 0.00" | A flagship deep-dive teaches a bug artifact as fact | Add a correction banner linking the [beta-zero post-mortem](/story/09/42_beta_zero_bug) | S | M |
| 4. Unsupervised PCA, not supervised (PLS/LDA) | PCA may discard the image signal; the falsification verdict could be PCA's fault | Add a PLS-64 rerun of the image track for a fair comparison | M | M-H |
| 5. Flat-bps spread, not liquidity-tiered | Small caps have wider spreads; flat bps understates their cost | `spread_bps` scaled by ADV tier | S | M |
| 7. `run_pipeline.main` has no end-to-end test | The row-alignment landmine lived in untested orchestration | Synthetic end-to-end test of the real `main`, or extract a tested alignment helper | S-M | M |
| 8. Long runs aren't resumable | A 6–10 hr run that dies restarts from scratch | Per-fold checkpoint files; resume from last completed fold | M | M |
| 6. Universe is approximate, not point-in-time | Survivorship residual remains; weakens rigor claims | Build a true PIT universe (listing/delisting calendar) | L | H (rigor) |
| 9. SPA failure is silent | `reality_check_spa` returns an error dict; report shows a placeholder with no reason | Log SPA failures + the failure reason | S | L-M |
| 10. Wash-sale ignored; lockbox TOCTOU; EDGAR no retry/backoff | Minor correctness/robustness edges | Note wash-sale assumption; harden claim-check; add retry w/ backoff on 5xx | S | L (each) |

## 1. Settle/realize loop (the big missing half)

`scripts/forward_pick.py` exists and now works (see [forward paper-trading harness](/story/09/46_forward_pick_harness)) — it freezes a model trained through today, emits next-Monday's top-5, and writes `data/forward_picks/<date>.csv`. Its docstring promises a companion `settle_picks.py` that "looks up realized returns 40 trading days later, writing `data/forward_realized/`." That script **does not exist**. So the harness generates picks into a void: nothing ever scores them against realized outcomes.

This matters more than anything else on the list. As the docstring itself says, after 26 weeks you'd have ~130 fully out-of-sample trades evaluated under real execution — "the only result that survives all methodological objections." Walk-forward, embargo, [deflated Sharpe](/story/09/10_deflated_sharpe), and bootstrap all defend against *backtest* overfitting; live forward picks sidestep that entire debate. Right now that defense is unbuilt. The fix is small in scope, medium in care: a daily cron that walks `data/forward_picks/`, finds any pick file ≥40 trading days old and unsettled, looks up the realized adjusted return per ticker (same [trading calendar](/story/09/37_trading_calendar) the simulator uses), and appends settled rows idempotently.

## 2. Paired bootstrap for the GAP (the unproven headline)

`report_card.py` emits the central comparison — model CAGR minus best-baseline CAGR — and every generated card carries the literal string `gap_ci = "(needs paired bootstrap; see stats.py)"`. The point estimate is there, but the confidence interval is a TODO and `gap_significant` is never decided.

This is the load-bearing claim of the whole project: does the model beat the *best* [simple baseline](/story/09/20_simple_baselines), not just *a* baseline? The honest report card prints a number and then admits it can't say whether the number is distinguishable from zero. `stats.py` already has the [block-bootstrap](/story/09/22_block_bootstrap_params) machinery; what's missing is the *paired* variant operating on per-period return differences (model − best baseline), which controls for shared market exposure and is far tighter than two independent CIs. The output should feed [multiple-comparison adjustment](/story/09/23_multiple_comparison_landscape), since "best baseline" is itself a max over K candidates.

## 3. Reproducibility manifests

[Reproducibility seeds](/story/09/12_reproducibility_seeds) pins every RNG, and [disaster recovery](/story/09/17_disaster_recovery) documents a full restore — but bit-identical reproduction only holds if `data/adjusted/*.parquet` is byte-identical, and yfinance silently restates history. It's a small script: hash the tree, commit the manifest, diff against it on restore, fail loudly on drift. Pair with a periodic NAS/S3 snapshot so the *exact* bytes survive upstream mutation.

## 4. Supervised reduction vs PCA

The image track runs DINOv2 embeddings through [PCA](/story/09/02_pca_math) before modeling. PCA is unsupervised: it keeps high-variance directions and discards low-variance ones — but the predictive image signal may live precisely in a low-variance direction PCA throws away. That means the [falsification](/story/05_falsification) verdict ("the image track loses") might be partly PCA's fault, not the embeddings'. A PLS-64 (or LDA) rerun — supervised reduction toward the label — would make the comparison fair. Until then, the falsification stands but with an asterisk. See [DINOv2 architecture](/story/09/01_dinov2_architecture).

## Suggested next sprint

The four highest value/effort-ratio items, in order:

1. **Build the settle loop** (`settle_picks.py` + cron) — turns a dead-end harness into the project's strongest evidence.
2. **Paired-bootstrap the GAP** — proves or kills the central claim; the machinery already half-exists.
3. **`data_hashes.json`** — cheap insurance against yfinance drift quietly breaking "reproducible."
4. **Beta-doc correction banner** — a one-line fix that stops a flagship page from teaching a bug.

Together these are roughly one focused sprint, and they close the two gaps that most undercut credibility (no live scoring, no significance on the headline gap) plus two near-free integrity fixes. A natural continuation of [the hardening story](/story/09/39_hardening_story).
