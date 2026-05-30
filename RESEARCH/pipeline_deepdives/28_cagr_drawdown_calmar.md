# CAGR, maximum drawdown, and the Calmar ratio

A single number cannot summarize an equity curve. The headline scorecard for our `lgbm_engineered` run pairs return with pain — and pain takes more than one shape. This deep-dive walks through CAGR, maximum drawdown, and the Calmar ratio, then explains why we report all three with confidence intervals from a [block bootstrap](/story/09/22_block_bootstrap_params).

## CAGR: the constant rate that connects the endpoints

Compound Annual Growth Rate is the single constant rate that would compound the starting equity into the ending equity over the actual elapsed time.

```
CAGR = (end_equity / start_equity)^(1 / years) - 1
```

Worked example. Suppose $100k grows to $103k over 7 years:

```
CAGR = (103000 / 100000)^(1/7) - 1
     = (1.03)^(1/7) - 1
     = 0.0042
     = 0.42%
```

For our `lgbm_engineered` run, $100k grew to $146k over 8 years:

```
CAGR = (146000 / 100000)^(1/8) - 1
     ≈ +4.3% per year
```

CAGR is brutally endpoint-sensitive. One bad year at the end can demolish a decade of compounding, and the metric tells you nothing about the path taken. That is why we never publish CAGR alone — it lives next to a [Sharpe ratio](/story/09/27_sharpe_ratio) (which prices in variance) and a max drawdown (which prices in the worst stretch).

## Maximum drawdown

Max drawdown is the largest peak-to-trough decline in the equity curve, as a percentage of the peak. The algorithm is a single pass:

```python
def max_drawdown(equity):
    peak = equity[0]
    worst = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak  # always <= 0
        if dd < worst:
            worst = dd
    return worst  # most negative value over the whole series
```

For `lgbm_engineered`, max_dd is **-44%**. The peak landed around mid-2022; the trough came in January 2023 after the rate-shock bear market.

Max DD captures something Sharpe cannot: the pain of holding through the worst stretch. A strategy can have a tidy Sharpe and still be unholdable if the worst trough is deep enough to trigger a forced exit — by the investor, the risk committee, or the broker.

## Calmar ratio: CAGR per unit of worst-case pain

The Calmar ratio (sometimes called MAR) is annualized return divided by the magnitude of the worst drawdown.

```
Calmar = CAGR / |max_dd|
```

For our run:

```
Calmar = 0.043 / 0.44 ≈ 0.10
```

Calmar conventions: below 0.5 is grim, 0.5–1.0 is what institutional momentum managers target, >1.0 is exceptional. At 0.10 we are well below the bar. The virtue of Calmar is that it is hard to game by tail-clipping — unlike Sharpe, which can be inflated by truncating losers, Calmar specifically *rewards* the absence of deep troughs. This makes it complementary to the [Deflated Sharpe](/story/09/10_deflated_sharpe), which guards against multiple-testing inflation of the *mean* but says nothing about the worst path.

## What max_dd misses

Depth is only one dimension of drawdown pain. Three things max_dd does not see:

- **Time-under-water.** How long were we below the prior peak? Our 44% DD took about 6 months to bottom and roughly 15 months to recover the previous high — a 21-month stretch of "this is not working."
- **Number of drawdowns.** A track record with four separate -20% drawdowns has a different psychological texture than one with a single -44%.
- **Path shape.** A slow, grinding descent vs a violent flash-crash leg down feel different. Same depth, different experience.

## Confidence intervals on these metrics

Point estimates lie. We resample the daily return series with [block bootstrap](/story/09/22_block_bootstrap_params) (block length chosen from the autocorrelation structure — see also [effective sample size](/story/09/24_effective_sample_size)) and recompute each metric on every replicate. For `lgbm_engineered`:

- **CAGR 95% CI:** `[-4.2%, +12.6%]`
- **max_dd 95% CI:** `[-67%, -28%]`

These are wide enough that the honest reading is "we cannot rule out that the strategy loses money, and we cannot rule out a future drawdown near -70%."

## Sortino: the downside-only cousin

The Sortino ratio is Sharpe with downside deviation in the denominator instead of full-sample standard deviation. Up-volatility is no longer penalized. Our pipeline uses a path-dependent training label called `sortino_label_40d` built from per-trade MFE/MAE — see [Label grid](/story/09/19_label_grid).

## The composite scorecard

No single metric tells you the answer. The honest report card forces multiple lenses on the same equity curve:

- **CAGR** (point + 95% CI)
- **Sharpe** (point + 95% CI) — see [Sharpe ratio](/story/09/27_sharpe_ratio)
- **max_dd** (point + 95% CI)
- **[Deflated Sharpe](/story/09/10_deflated_sharpe) threshold + p-value** — see [Deflated Sharpe](/story/09/10_deflated_sharpe)
- **[Effective N](/story/09/24_effective_sample_size) and SE widening** — see [Effective sample size](/story/09/24_effective_sample_size)
- **Post-tax, post-friction CAGR** — see [Frictions beyond impact](/story/09/25_frictions_beyond_impact)
- **Best [simple baseline](/story/09/20_simple_baselines) + gap** — see [Simple baselines](/story/09/20_simple_baselines)

Read together, these tell a coherent story. Read individually, any one of them can mislead.
