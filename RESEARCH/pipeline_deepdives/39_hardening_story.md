# The hardening story — how 21 bugs were found and fixed across 18 passes

This pipeline did not arrive correct. It arrived *plausible* — it ran end to end, produced equity curves, printed report cards, and never threw — and then we spent eighteen passes proving that "runs without error" and "is right" are different claims. Seventeen of those passes were incremental: write tests against a component, watch something fail, fix it, repeat. The eighteenth was different in kind — a dedicated multi-agent bug **audit** whose only job was to assume the code was wrong and try to prove it. Across the whole effort the test suite grew from **15 to 334 tests**, and the bug count tells a story that has very little to do with crashes.

## The most dangerous bugs were silent

If you scan the audit's findings, the pattern that should worry you is not the crashes. It is the four bugs that produced *wrong numbers without complaint*: look-ahead in the exit fill, false-positive label corruption, a dead feature, and a row-misalignment landmine. None of these threw. All of them would have quietly biased the research conclusion. A pipeline that crashes tells you it is broken. A pipeline that returns a confident, leakage-tainted [Sharpe ratio](/story/09/27_sharpe_ratio) does not.

The [look-ahead exit bug](/story/09/40_lookahead_exit_bug) is the cleanest example. The [simulator loop](/story/09/06_simulator_loop) advanced on the union calendar across all tickers, so when an individual ticker's calendar had a gap straddling its `exit_date`, the position filled at a *future* bar that the union calendar had already reached — the simulator could see a price the strategy could not have traded on. The fix was to defer the close until `today` reaches the ticker's real bar. This is exactly the failure mode the [walk-forward embargo](/story/09/08_walkforward_embargo) work exists to prevent, leaking back in through the execution layer instead of the training split.

The label corruption (see [the label grid](/story/09/19_label_grid)) was subtler. When an anchor's close was non-positive — delisted names, heavily adjusted prices — the forward return computed as `inf`, and `inf >= threshold` silently minted a **false-positive** binary label. The model was being trained, on a handful of corrupted rows, to associate broken price history with "winners." Fixed by guarding `ret`/`log_ret` and dropping `anchor_close <= 0` rows.

## When a test that only checks "is it finite" lets a dead feature live

The single most instructive bug is [why beta was zero](/story/09/42_beta_zero_bug). `beta_spy_63d` was computed from a 63-close window, which yields 62 returns, not 63 — an off-by-one that meant the internal length guard *never passed*. So beta returned `0.0` for **every stock, for the entire life of the project**. Worse, SPY was aligned to the stock by row position rather than by date, so even had the guard passed, the regression would have been against misaligned market returns. The feature was dead code wearing a feature's name. See [SPY beta](/story/09/38_spy_beta) for the corrected 64-close slice and date-based alignment.

Here is the uncomfortable part: this survived seventeen passes of testing. Why? Because the tests asserted that beta was *finite* and *not NaN* — and `0.0` is gloriously finite. A constant-zero column passes every smoke test you can write about its dtype and its range. It fails only a test that asserts a *value*: beta of a high-beta name against itself should be ≈1, beta of SPY against SPY should be 1.0. The lesson generalized into our [testing philosophy](/story/09/45_testing_philosophy): finiteness assertions are necessary and nearly worthless. Value assertions are what catch dead features.

## The audit findings, by severity

These are the 11 bugs the dedicated audit surfaced after the 17 incremental passes had already run.

| Severity | File | Bug | Fix |
|---|---|---|---|
| Critical (silent) | `labels.py` | Non-positive anchor close → `inf` return → `inf>=threshold` minted false-positive labels | Guard `ret`/`log_ret`; drop `anchor_close<=0` rows ([labels](/story/09/19_label_grid)) |
| Critical (silent) | `simulator.py` | Exit filled at a future bar via the union calendar when a ticker's calendar gapped its exit date — look-ahead | Defer close until `today` reaches the real bar ([look-ahead](/story/09/40_lookahead_exit_bug)) |
| Critical (silent) | `features.py` | `beta_spy_63d` dead: off-by-one length guard never passed (beta `0.0` for every stock) + SPY aligned by position not date | 64-close slice + date alignment ([SPY beta](/story/09/38_spy_beta)) |
| Critical (silent) | `run_pipeline.py` | Row-alignment landmine: `merged.iloc[:len(embs)]` positional prefix + mask over the wrong frame would misalign image features with labels if any anchor dropped | Derive `embs`, `vol_feats`, kept frame from one canonical `anchor_kept` + one mask ([row-alignment landmine](/story/09/41_row_alignment)) |
| High | `simulator.py` | Ticker in `scores` but absent from `price_lookup` opened a phantom position (cash debited, never closed, never in [blotter](/story/09/34_blotter_equity_summary)) | Skip tickers not in `price_lookup` at entry ([phantom positions](/story/09/49_phantom_positions)) |
| High | `models.py` | PCA `n_components` clamped to `n_features` only → crash on 20–63-row folds | `min(pca_dim, n_features, n_samples)` ([degenerate folds](/story/09/48_degenerate_folds)) |
| Medium | `models.py` | `train_test_split(stratify=y)` crashed on a singleton class | Fall back to unstratified ([degenerate folds](/story/09/48_degenerate_folds)) |
| Medium | `stats.py` | `bootstrap_cagr_ci`: `np.log` on non-positive equity → `-inf`/`nan` poisoned the CIs | Clip equity to `1e-9` ([bootstrap](/story/09/22_block_bootstrap_params)) |
| Medium | `stats.py` | `post_tax_cagr`: negative compounding base → complex number → `float(complex)` `TypeError` on a wiped-out account | Floor base at 0 ([post-tax](/story/09/25_frictions_beyond_impact)) |
| Low | `simulator.py` | Forced end-of-sim close used flat slippage, not participation-scaled impact | Use Almgren-Chriss impact for consistency ([Almgren-Chriss](/story/09/09_almgren_chriss)) |
| Low | `report_card.py` | `KeyError` on a summary dict missing `'track'` — the one non-`.get` access | Use `.get` |

## The earlier seventeen passes

The incremental passes were the warm-up, and they followed the same grain: cover a component, find the edge it never handled. Among them — `effective_sample_size` crashed on an empty blotter (see [effective sample size](/story/09/24_effective_sample_size)); the label MFE/MAE sign convention was inverted; the simulator silently passed a NaN ADV and separately crashed on a NaN `anchor_close`; the EDGAR ticker regex dropped SPAC multi-class share lines; `report_card` threw a `KeyError` on a partial summary; `data_acq` crashed on empty/`None`/missing-`Close` frames; the chart renderer divided by zero on a degenerate log-y axis; `embed_dinov2` crashed on an empty batch; the README had drifted from the code; and `splits` crashed on tz-aware timestamps (relevant to [reproducibility seeds](/story/09/12_reproducibility_seeds) and the [walk-forward](/story/09/08_walkforward_embargo) machinery). Even the *cross-linker* that wires these deep-dives together had its own bugs — an infinite loop and a non-idempotency defect — fixed and tested like everything else.

## What made the fixes trustworthy

Two disciplines did the real work. First, **every fix shipped with a regression test written to fail before the fix** — not a test that happens to pass now, but one we confirmed turned red against the buggy code first. That is the only way to know a test tests anything. Second, the audit used **adversarial multi-agent verification in default-refute mode** (see [the audit methodology](/story/09/51_audit_methodology)): each candidate bug had to survive an agent whose job was to argue it was a false alarm. That filter is why the table above contains 11 *real* bugs and not 30 nervous ones — phantom findings got refuted, genuine leakage and corruption did not. Crashes are free to find. Silent wrongness costs you an audit, value-level assertions, and a verifier that assumes you're wrong until you prove otherwise. The full known-limitations list lives in the [feature-gaps roadmap](/story/09/47_roadmap).
