# Feature gaps & roadmap — what's missing and what to build next

This is the honest accounting page. Everything earlier in the `/story/09` series describes what the pipeline *does*; this page is the inverse — what it does not yet do, what's stubbed, and what's documented-as-future. Each entry is grounded in the current tree (`src/stock_chart/`, `scripts/`, `tests/`), not aspiration.

The table is sorted by value-to-effort ratio. Effort is S/M/L (a few hours / a day or two / a multi-day project). Value is H/M/L for how much the fix moves the central question — *does this model actually beat a simple baseline, out of sample, after costs?*

## The prioritized table

> **Update:** items 1, 2, 3, and 11 below are now **✅ shipped** (this session).
> The table keeps them for context; the prose under each notes what landed.

| Gap | Why it matters | Approach | Effort | Value |
|---|---|---|---|---|
| 1. ✅ Settle/realize loop for forward picks | Forward paper-trading is the one result that survives every objection — but picks were never scored | **Shipped:** `scripts/settle_picks.py` settles resolvable picks idempotently into `data/forward_realized/realized.csv` | M | H |
| 2. ✅ Paired bootstrap on the model-vs-best-baseline GAP | This is *the* claim; `gap_ci` was a TODO string | **Shipped:** `stats.paired_gap_bootstrap` + wired into `report_card` (`gap_ci`/`gap_significant`) | M | H |
| 3. ✅ `data_hashes.json` | "Bit-identical repro" only holds if parquet is byte-identical, but yfinance drifts | **Shipped:** `manifest.write/verify_data_hashes` + `scripts/check_data_hashes.py --write/--verify` | S | M |
| 11. ✅ SPY-beta doc stated a now-false "BNTX beta = 0.00" | A flagship deep-dive taught a bug artifact as fact | **Shipped:** correction banner on [38_spy_beta](/story/09/38_spy_beta) linking the [beta-zero post-mortem](/story/09/42_beta_zero_bug) | S | M |
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

## Suggested next sprint — ✅ delivered

The four highest value/effort-ratio items were built this session:

1. ✅ **The settle loop** (`scripts/settle_picks.py`) — grades resolvable forward picks 40 trading days later, idempotently, into `data/forward_realized/realized.csv`. Wire it to a daily cron to start accumulating live out-of-sample trades.
2. ✅ **Paired-bootstrap the GAP** (`stats.paired_gap_bootstrap`) — the central claim now has a CI and p-value. On the `full` run the +5.26% headline gap has a 95% CI of roughly **[−15%, +49%]** annualized with **p(gap≤0) ≈ 0.27** — i.e., *not* distinguishable from zero even before multiple-comparison adjustment. Exactly the honest verdict the project exists to produce.
3. ✅ **`data_hashes.json`** (`scripts/check_data_hashes.py`) — `--write` to snapshot, `--verify` to detect yfinance drift on restore.
4. ✅ **Beta-doc correction banner** — done on [38_spy_beta](/story/09/38_spy_beta).

What remains for a future sprint: PLS-vs-PCA fairness rerun, liquidity-tiered spread, a true point-in-time universe, an end-to-end test of `run_pipeline.main`, resumable runs, and SPA-failure logging. A natural continuation of [the hardening story](/story/09/39_hardening_story).
