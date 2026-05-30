# Sharpe ratio — definition, interpretation, and the dozen ways people get it wrong

## What the number actually is

The Sharpe ratio is the workhorse of risk-adjusted return measurement. The definition:

```
Sharpe = (E[r_strategy] - r_f) / std(r_strategy - r_f)
```

where `r_f` is the risk-free rate (typically a T-bill yield). For daily data, the standard annualization is:

```
Sharpe_annual = Sharpe_daily * sqrt(252)
```

The `sqrt(252)` factor assumes returns are independent and identically distributed (IID) across the 252 trading days in a year — a load-bearing assumption we'll dismantle below.

## What it's measuring

Sharpe is asking: **how much excess return do I get per unit of volatility risk?** It collapses the joint distribution of returns into a single mean-over-spread number. A Sharpe of 1.0 means the strategy's expected annual excess return equals one annual standard deviation of returns. Higher is better — but only along the volatility axis. Sharpe is mute on everything else (skew, drawdown, lock-up, regime stability).

## Concrete computation in our pipeline

For the `lgbm_engineered` strategy in our [chapter-9 reproduction](/story/09_pipeline_walkthrough), the equity curve gives:

```
daily_mean_return     = 0.000159
daily_std_return      = 0.005000
daily_sharpe          = 0.000159 / 0.005000 = 0.0319
annualized_sharpe     = 0.0319 * sqrt(252)  = 0.510
```

The daily std of 0.005 is low because this is a **portfolio-level** equity curve, not an individual-stock return series. Diversification has already done its work by the time we measure the curve.

## Conventions this pipeline uses

A few choices that matter, all defensible but worth stating explicitly:

- **Risk-free rate = 0.** This is a long-only equity selection strategy. The honest benchmark is the universe-equal-weight return, not T-bills. Subtracting T-bills here would inflate Sharpe by free-money the strategy isn't actually capturing — see "Wrong benchmark" below.
- **Annualization factor = sqrt(252).** Standard, but see the IID caveat.
- **Daily returns from the equity curve via `pct_change()`.** Simple returns, not [log returns](/story/09/35_log_vs_simple_returns) — see [Log vs simple returns](/story/09/35_log_vs_simple_returns). For sub-1% daily moves the difference is negligible (<0.5bp).

## What Sharpe doesn't capture

Three big omissions, all of which bite in practice:

1. **Skew.** A Sharpe of 1.0 from symmetric returns is very different from a Sharpe of 1.0 from "win 95% of months, then take a -50% month once." Sharpe averages over the shape of the loss tail.
2. **Tail risk.** Maximum drawdown, conditional VaR, and time-to-recovery are all invisible to Sharpe. A strategy with Sharpe 1.0 and 60% max drawdown is unholdable for most institutional allocators, regardless of what the ratio says. See [CAGR + drawdown + Calmar](/story/09/28_cagr_drawdown_calmar).
3. **Capital lock-up and turnover.** A strategy that holds one position for 5 years with a stably rising price has the **same Sharpe** as one that churns 5-day positions and ends up at the same place.

## Interpretation thresholds

Rough but useful priors when staring at a backtest:

| Annualized Sharpe | Reading |
|---|---|
| < 0.3 | Roughly noise; cannot distinguish from random |
| ~0.5 | Marginally interesting; needs CI to take seriously |
| ~1.0 | Institutionally tradable if it survives stress tests |
| ~2.0 | Rare; suggests a real edge if it survives DSR |
| 3.0+ | Almost certainly overfit, data-snooped, or has a data error |

A reported Sharpe above 3 should trigger skepticism, not celebration.

## The famous mistakes

The dozen ways the headline number misleads, in roughly increasing severity:

- **No N_trials adjustment.** If you tried 108 model variants and reported the best Sharpe, you're reporting a maximum-of-N order statistic, not a Sharpe. See [Deflated Sharpe Ratio](/story/09/10_deflated_sharpe).
- **No effective-N adjustment.** Overlapping windows, autocorrelated returns, and short test sets all shrink the *effective* sample size below the nominal day count. See [Effective sample size](/story/09/24_effective_sample_size). For our test set, effective-N is roughly 1/25th of nominal, which widens the Sharpe standard error by ~5x.
- **No skew/kurt adjustment.** Lo (2002) derived the standard error of Sharpe under non-normal returns: it widens with negative skew and excess kurtosis. Our return distributions are slightly left-skewed and fat-tailed.
- **Annualization assumes IID.** Daily strategy returns are *not* IID — they autocorrelate when positions overlap. The `sqrt(252)` factor *understates* true annual volatility whenever autocorrelation is positive, inflating Sharpe.
- **Wrong benchmark.** Subtracting T-bills from a long-only equity strategy's returns silently credits the strategy with the equity risk premium itself. Compare to the universe equal-weight return instead.
- **Survivorship bias in the universe.** If delisted tickers are missing from your panel, your Sharpe is measuring the return of a hand-picked subset of winners. See [universe construction](/story/09/14_universe_construction).
- **Look-ahead leakage in features.** Any feature that peeks at future data will inflate Sharpe in-sample and collapse out-of-sample. See [walk-forward embargo](/story/09/08_walkforward_embargo).
- **Transaction costs ignored.** A Sharpe of 1.0 gross can be a Sharpe of -0.2 net for a high-turnover strategy. See [Frictions](/story/09/25_frictions_beyond_impact).

## Putting a confidence interval on Sharpe

The right tool is a **[stationary block bootstrap](/story/09/22_block_bootstrap_params)** of the daily return series, which respects autocorrelation structure. See [Block bootstrap parameters](/story/09/22_block_bootstrap_params) for our block-length selection. For `lgbm_engineered`:

```
nominal Sharpe   = 0.510
95% bootstrap CI = [-0.18, +1.21]
```

That CI straddles zero. Before any of the formal corrections, the headline 0.510 is already statistically indistinguishable from no skill.

## Verdict for our reproduction

A nominal Sharpe of 0.510 reads as "marginally interesting" against the thresholds table. But:

1. The [Deflated Sharpe Ratio](/story/09/10_deflated_sharpe) threshold at N=108 trials is approximately **0.75** — our nominal value doesn't clear it.
2. The [effective-N correction](/story/09/24_effective_sample_size) widens the standard error roughly **5x** versus the naive estimate.
3. The [block bootstrap CI](/story/09/22_block_bootstrap_params) already includes zero before either correction.

Applying both corrections produces an interval that comfortably contains 0. The honest read on `lgbm_engineered` is that the headline 0.510 is *consistent with no edge at all* once you account for trial multiplicity and the true precision of the test set.
