# The full multiple-comparison landscape — Bonferroni, Holm, DSR, SPA, Reality Check

When you try 36 model configurations on the same dataset and report only the best, you have not discovered an edge — you have discovered the maximum of 36 noisy estimates. The true Sharpe of the "winner" is biased upward, and a naive p-value on it is meaningless. Multiple-comparison corrections fix this, but they make very different assumptions about how the trials are related. Below is the full landscape, sorted from most conservative to least.

## 1. Bonferroni correction — the blunt instrument

**Math sketch.** To control family-wise error rate (FWER) at alpha over K tests, require each individual p-value to satisfy `p_i ≤ alpha / K`. For alpha = 0.05 and K = 36, the threshold is `p ≤ 0.00139`.

**When to use it.** Anywhere you need a one-line, defensible, no-assumption correction. It is provably valid for *any* dependence structure.

**When it fails.** Whenever trials are correlated, which in our pipeline is always: 36 configs on the same price history with overlapping features produce Sharpe estimates that are not 36 independent draws. Bonferroni treats them as if they were, so it overcorrects dramatically.

**Pipeline use.** Sanity floor only. If a strategy survives Bonferroni, we can stop arguing.

## 2. Holm-Bonferroni (and Holm-Sidák) — the cheap upgrade

**Math sketch.** Sort the K p-values ascending. Compare each to `alpha / (K - i + 1)`, stopping at the first failure. Strictly more powerful than Bonferroni at no cost.

**When to use it.** Few trials (K ≤ 20), genuinely independent or nearly so.

**When it fails.** Still ignores correlation. For our K = 36–108 range with heavy correlation it's too strict.

**Pipeline use.** Only in unit tests with genuinely independent diagnostics.

## 3. White's Reality Check (1999) — the first bootstrap method

**Math sketch.** Form the loss-differential vector `d_k = L(benchmark) - L(strategy_k)` for each k. Stationary block bootstrap to get the null distribution of `max_k mean(d_k)`. The Reality Check p-value is the bootstrap quantile of the observed best.

**When to use it.** Correlated trials, non-normal Sharpe distribution.

**When it fails.** The null includes *every* strategy, including obviously terrible ones that pull the null mean down. Conservative whenever the candidate pool has a long tail of losers.

**Pipeline use.** Historical — retired from the headline JSON in favor of SPA.

## 4. Hansen's Superior Predictive Ability — White's RC, recentered

**Math sketch.** Same bootstrap setup as RC, but recenter the loss-differentials: only strategies that are not statistically dominated by the benchmark contribute to the null.

**When to use it.** Many correlated trials with heterogeneous quality — our default case.

**When it fails.** The BCS pre-test can degenerate when loss-differential variance is near zero, or when block-bootstrap windows wrap pathologically.

**Pipeline use.** Implemented in `src/stock_chart/stats.py` via the `arch` package's `SPA` class.

## 5. Deflated Sharpe Ratio (Bailey & López de Prado 2014) — analytical

**Math sketch.** Assume Sharpe estimates across N trials are approximately `N(0, sigma_SR^2)` under the null. The expected maximum of N i.i.d. standard normals is roughly `sqrt(2 ln N) - (gamma + ln ln N) / (2 sqrt(2 ln N))`. DSR adjusts the observed best Sharpe by this expected max.

**When to use it.** Many correlated trials with moderate skew/kurtosis. Fast, closed-form.

**When it fails.** Heavy-tailed distributions break the normal approximation.

**Pipeline use.** Always reported in `headline.json`.

## Comparison table

| Method | Year | Handles correlation? | Conservatism | Cost |
|---|---|---|---|---|
| Bonferroni | 1936 | No | Highest | O(1) |
| Holm-Bonferroni | 1979 | No | High | O(K log K) |
| White's RC | 1999 | Yes | High | O(B·K) |
| Hansen's SPA | 2005 | Yes | Moderate | O(B·K) |
| Deflated Sharpe | 2014 | Implicitly | Moderate | O(N) |

## Decision tree

- **Few uncorrelated trials (K ≤ 20):** Holm-Bonferroni is fine and simple.
- **Many correlated trials, moderate skew/kurt:** DSR is the analytical choice. Fast, defensible.
- **Many correlated trials, heavy skew/kurt:** SPA is more robust.
- **Want both:** Use DSR + SPA. Disagreements signal looking at the return distribution.

## What our pipeline actually does

1. **Always report DSR.** Cheap and never crashes.
2. **Report SPA when it succeeds.** Falls back to DSR alone on degenerate inputs.
3. **Headline verdict uses the more conservative of the two.**
4. **Always quote N_trials.** A correction without N is unreplayable.

**Concrete numbers from the most recent sweep:** with N = 108 trials, the best observed annualized Sharpe was **0.510**. DSR threshold for alpha = 0.05 was **~0.75**, so not significant after deflation. SPA returned p ≈ **0.18**. Both methods agree: **no edge survives multiple-comparison correction** on this sweep.

That null result is the headline, and it is more important than any of the per-config numbers behind it.

> See [/story/09/10](/story/09/10) for the DSR derivation and [/story/06_statistics](/story/06_statistics) for the broader chapter.
