# Glossary — every term and metric across the deep-dive series

Sixty-some pages, one vocabulary. The deep-dives assume you already know what an *anchor* is, what *purging* protects, why a *deflated* Sharpe is the only Sharpe worth quoting. This is the decoder ring: every load-bearing term, one or two sentences, linked to the page that owns it. Alphabetical; where pages share a concept, both link.

## A

**ADV20** — Twenty-day average dollar volume, the liquidity yardstick that caps position size before its own [Almgren-Chriss impact](/story/09/09_almgren_chriss) eats the edge; it gates the [position-size formula](/story/09/32_position_size_formula). [/story/09/33_adv20_metric]

**Allocation budget** — The total capital the strategy may deploy at once, split across simultaneous positions; it turns a ranked pick list into an actual portfolio. See [position sizing](/story/09/32_position_size_formula). [/story/09/36_allocation_budget]

**Almgren-Chriss impact** — The square-root market-impact model: trading a fraction of ADV moves the price against you in proportion to roughly the square root of participation. The dominant friction at size. [/story/09/09_almgren_chriss]

**Anchor / [weekly anchor](/story/09/18_weekly_anchors)** — A single `(ticker, date)` point from which features look backward and labels resolve forward; the atomic unit, sampled weekly to decorrelate adjacent observations. [/story/09/18_weekly_anchors]

## B

**Beta (beta_spy_63d)** — A stock's regression slope against SPY over a trailing window, one of the [40 engineered features](/story/09/03_engineered_features). Notorious for the [bug](/story/09/42_beta_zero_bug): an off-by-one returned `0.0` for every stock for the project's entire life. [/story/09/38_spy_beta]

**[Blotter](/story/09/34_blotter_equity_summary) / equity curve / summary** — The three artifacts of a run: the per-trade ledger, the portfolio-value-over-time curve, and the rolled-up scorecard ([CAGR](/story/09/28_cagr_drawdown_calmar), [Sharpe](/story/09/27_sharpe_ratio), and friends). [/story/09/34_blotter_equity_summary]

**Block bootstrap (stationary)** — A resampling scheme drawing contiguous *blocks* of returns rather than independent days, preserving autocorrelation; the basis for honest confidence intervals on the [gap test](/story/09/52_paired_gap_test). [/story/09/22_block_bootstrap_params]

## C

**CAGR / max drawdown / Calmar** — Compound annual growth rate, max drawdown (worst peak-to-trough loss), and Calmar (CAGR over max drawdown) — the return-vs-pain triad complementing [Sharpe](/story/09/27_sharpe_ratio). [/story/09/28_cagr_drawdown_calmar]

**Calibration ([isotonic](/story/09/30_isotonic_calibration))** — A monotone post-hoc transform mapping raw scores onto honest probabilities, so a "0.7" means close to a 70% hit rate; isotonic because it assumes nothing beyond monotonicity. [/story/09/30_isotonic_calibration]

**Cost stack** — The full ledger of frictions on every trade — [slippage, spread, commission](/story/09/56_cost_stack) plus [Almgren-Chriss impact](/story/09/09_almgren_chriss) and [other frictions](/story/09/25_frictions_beyond_impact) — so net return is what a real account keeps. [/story/09/25_frictions_beyond_impact]

## D

**Data flow contract** — The structural guarantee that one row means the same thing at every stage, raw parquet to simulator exit; the antidote to silent [leakage](/story/09/40_lookahead_exit_bug) and misalignment. [/story/09/60_data_flow_contract]

**Deflated Sharpe Ratio (DSR)** — A Sharpe adjusted downward for the number of strategies tried before finding this one; it answers "skill, or best of many coin-flips?" — the headline [multiple-comparison](/story/09/23_multiple_comparison_landscape) defense. [/story/09/10_deflated_sharpe]

**DINOv2 / ViT-S/14** — The self-supervised vision transformer (small variant, 14-pixel patches) that embeds a rendered price chart into a feature vector — the image half of the model. [/story/09/01_dinov2_architecture]

## E

**Effective sample size** — The number of *independent* observations after accounting for overlap and autocorrelation, usually far below the raw row count; it deflates every significance claim that assumes IID data. [/story/09/24_effective_sample_size]

**Embargo / purged walk-forward** — The leakage guard on the time-series split: *purge* drops training rows whose forward label resolves inside the test fold; the *embargo* extends that exclusion by a buffer of trading days. [/story/09/08_walkforward_embargo]

**Engineered features (the 40)** — The 40 hand-built numeric features (momentum, volatility, [beta](/story/09/38_spy_beta), volume shape) computed strictly from the lookback window — the tabular complement to the [DINOv2](/story/09/01_dinov2_architecture) embedding. [/story/09/03_engineered_features]

## F

**Falsification test** — A deliberately destructive experiment designed to *break* the result — shuffle labels, randomize picks — to confirm the edge disappears when it should. If it doesn't kill the signal, the signal was never real. [/story/05_falsification]

**[Fixed log-y axis](/story/09/29_fixed_log_y_axis)** — Always plotting equity on a fixed logarithmic y-axis so equal percentage moves look equal and no run flatters itself with a rescaled or linear axis. [/story/09/29_fixed_log_y_axis]

## G

**Gap (model vs best baseline) + paired bootstrap** — The signed margin between the model and the strongest [simple baseline](/story/09/20_simple_baselines), measured *paired* on the same anchors and run through a [block bootstrap](/story/09/22_block_bootstrap_params). This gap, not raw return, is the verdict's currency. [/story/09/52_paired_gap_test]

## L

**LightGBM / gradient boosting** — The gradient-boosted decision-tree library that consumes the combined [image embedding](/story/09/01_dinov2_architecture) and [engineered features](/story/09/03_engineered_features) and emits a score; the tabular learner. [/story/09/31_lightgbm_internals]

**Lockbox / trial registry** — The lockbox is a held-out slice touched exactly once, at the end; the trial registry logs every variant evaluated against it, so [deflation](/story/09/10_deflated_sharpe) knows the true attempt count. [/story/09/16_trial_registry]

**Log vs simple returns** — [Log returns](/story/09/35_log_vs_simple_returns) add across time and are symmetric; simple returns compound and aggregate across a portfolio. Mixing them is a classic silent error, so the pipeline is explicit about each. [/story/09/35_log_vs_simple_returns]

**Look-ahead / leakage** — Any path by which future information reaches a present decision — an exit on a bar the strategy couldn't have traded, a label bleeding across a split boundary. The [exit-fill bug](/story/09/40_lookahead_exit_bug) is the canonical case; the [data-flow contract](/story/09/60_data_flow_contract) the systemic defense. [/story/09/40_lookahead_exit_bug], [/story/09/60_data_flow_contract]

## M

**[MA250 prefilter](/story/09/07_ma250_prefilter)** — A 250-day moving-average screen on the [universe](/story/09/57_survivorship_accounting) before anchors are drawn, so picks and the random baseline sample from the same pre-trended pool. [/story/09/07_ma250_prefilter]

**MFE / MAE** — Maximum favorable and adverse excursion: the best and worst unrealized moves a position reaches before its [resolution date](/story/09/19_label_grid). They shape the label grid and inform [trailing-stop](/story/09/11_trailing_stop_interactions) design. [/story/09/19_label_grid]

**Multiple-comparison correction** — The methods (Bonferroni, Holm, SPA, and the [Deflated Sharpe](/story/09/10_deflated_sharpe)) that pay the statistical tax for testing many strategies, so the survivor isn't just the luckiest of the litter. [/story/09/23_multiple_comparison_landscape]

## P

**PCA** — Principal component analysis, the linear dimensionality reduction that compresses the [DINOv2](/story/09/01_dinov2_architecture) embedding before it reaches [LightGBM](/story/09/31_lightgbm_internals). [/story/09/02_pca_math]

**Position sizing / allocation budget** — How a ranked pick becomes a dollar amount: size capped by [ADV20](/story/09/33_adv20_metric) and impact, then fitted into the total [allocation budget](/story/09/36_allocation_budget). [/story/09/32_position_size_formula], [/story/09/36_allocation_budget]

## R

**Reproducibility / seeds / data_hashes** — Pinning every RNG seed (numpy, torch, sklearn, LightGBM, baseline) and SHA-256-fingerprinting every input artifact in the manifest, so any run is bit-for-bit re-derivable. [/story/09/12_reproducibility_seeds], [/story/09/54_data_hashes]

**Resolution date** — The trading-calendar date on which an anchor's forward label *becomes known*; it drives both the [purge/embargo](/story/09/08_walkforward_embargo) and exit timing in the simulator, and must be a real calendar date, not a naive offset. [/story/09/19_label_grid]

## S

**Settle loop / forward paper-trading** — The live, no-takebacks track: the [forward-pick harness](/story/09/46_forward_pick_harness) commits picks on a date and the settle loop resolves them only as real future prices arrive. [/story/09/46_forward_pick_harness], [/story/09/53_settle_loop]

**Sharpe ratio** — Annualized mean excess return over its volatility; the baseline risk-adjusted metric. Quote only after [deflation](/story/09/10_deflated_sharpe) and on the [effective sample size](/story/09/24_effective_sample_size). [/story/09/27_sharpe_ratio]

**Simple baselines (the 5)** — Five deliberately dumb strategies (random picks, buy-and-hold SPY, momentum, and the like) the model must beat to justify its complexity; the [gap](/story/09/52_paired_gap_test) is measured against the best. [/story/09/20_simple_baselines]

**Slippage / spread / commission** — The per-trade frictions below the impact model — slippage (fill vs intended price), bid-ask spread, commission — stacked into the full [cost stack](/story/09/56_cost_stack). [/story/09/56_cost_stack], [/story/09/25_frictions_beyond_impact]

**Survivorship bias** — Testing only on names that survived to today, inflating returns and shrinking drawdowns; corrected here by reconstructing a graveyard of delisted tickers to ~85–90% coverage. [/story/09/57_survivorship_accounting]

## T

**Trailing stop** — A stop that ratchets up as the position rises, capping give-back from the peak; its interaction with [MFE/MAE](/story/09/19_label_grid) and horizon reshapes realized returns. [/story/09/11_trailing_stop_interactions]

**Trial registry** — See [lockbox / trial registry](/story/09/16_trial_registry): the count of variants scored against the lockbox, the denominator [Deflated Sharpe](/story/09/10_deflated_sharpe) divides by. [/story/09/16_trial_registry]

## V

**[Validation modes](/story/09/21_validation_modes)** — The three ways a result is graded: **walk-forward** (purged cross-validation), **lockbox** (a one-touch held-out slice), and **forward** ([paper-trading](/story/09/53_settle_loop)) — each closing a loophole the others can't. [/story/09/21_validation_modes]

---

Two best entry points: [the verdict](/story/09/55_the_verdict) for what the effort concluded, and [the hardening story](/story/09/39_hardening_story) for how the pipeline earned its trust.
