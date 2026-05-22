# Deflated Sharpe Ratio — the math behind the threshold

The Sharpe ratio is the most-cited number in quant finance and also one of the easiest to fool yourself with. If you try enough models, one of them will post an impressive Sharpe purely by luck. Bailey and Lopez de Prado's 2014 *Deflated Sharpe Ratio* (DSR) is a closed-form correction for exactly that selection bias. This is the math our pipeline implements in `src/stock_chart/stats.py`, walked through end to end.

## The intuition: the best of N is not the truth

Imagine N traders, each flipping coins with zero edge. Their realized Sharpe ratios will scatter around zero. If you pick the **best one** and report only that number, you've conflated two things: real skill, and the upper tail of a noisy distribution. The bigger N, the further into the tail you reach.

DSR asks a precise question: under the null hypothesis of zero skill, how high a Sharpe should I *expect* the luckiest of N trials to post? Anything below that bar is consistent with noise. Anything above it is *evidence* — proportional to how far above.

## Expected max of N standard normals

Treat each trial's Sharpe (rescaled to its standard error) as a draw from `N(0, 1)`. The expected maximum of N i.i.d. standard normals has no closed form, but for moderate N it's well approximated using extreme-value theory:

```
if N <= 1: E[max] = 0
elif N == 2: E[max] = sqrt(2 ln N)
else:
    a = sqrt(2 ln N)
    E[max] ~= a - (ln(ln N) + gamma) / a    # gamma ~ 0.5772
```

The leading term `sqrt(2 ln N)` is the classic Gumbel-tail result for the maximum of normals. The `gamma` (Euler-Mascheroni constant) correction comes from the limiting Gumbel distribution's mean. The crucial qualitative point: the max grows like `sqrt(log N)`, *much* slower than `sqrt(N)`. You can run a hundred models and the luck-only ceiling barely doubles vs. running ten. But it does grow, and ignoring that growth is the bias DSR fixes.

## The variance of an individual trial's Sharpe

Each trial's Sharpe also has noise. Lo (2002) gives the variance of a Sharpe estimator with adjustments for non-normal returns:

```
SR_daily = SR_annualized / sqrt(252)
var(SR) ~= (1 - skew * SR_daily + (kurt - 1)/4 * SR_daily^2) / (n_obs - 1)
sigma_max = sqrt(var(SR))
```

This `sigma_max` is the standard deviation of a single trial's estimated Sharpe under the null. The threshold is then:

```
SR_threshold_daily = E[max_under_null] * sigma_max
```

## Concrete numbers from our reproduction

Our pipeline tried `N = 108` configs (36 labels x 3 model tracks). The best observed annualized Sharpe was 0.510.

```
SR_daily        = 0.510 / sqrt(252) ~= 0.0321
n_obs           ~= 2000     # 8 years * 252 daily returns
skew, kurt      = 0, 3      # baseline assumption
var(SR)         ~= 1 / 1999 ~= 5.00e-4
sigma_max       ~= sqrt(5.00e-4) ~= 0.0224

a               = sqrt(2 * ln(108)) ~= 3.06
correction      = (ln(ln 108) + 0.5772) / 3.06 ~= 0.46
E[max]          ~= 3.06 - 0.46 ~= 2.60

SR_thr_daily    ~= 2.60 * 0.0224 ~= 0.0583
SR_thr_annual   ~= 0.0583 * sqrt(252) ~= 0.925
```

The actual threshold reported in our results was `0.747`. The discrepancy comes from the skew/kurt terms being non-trivial in our real returns; the `0.925` here is a rough sanity check, not an exact reproduction.

## Why skew and kurt matter

A Sharpe of 1.0 built from Gaussian returns is not the same animal as a Sharpe of 1.0 built from a distribution with left skew and a fat right tail. The second one rests on a handful of outliers — kill those trades and the strategy is mediocre.

Look at the PSR denominator:

```
PSR = Phi( (SR_daily - SR_threshold_daily) * sqrt(n_obs - 1)
          / sqrt(1 - skew * SR_daily + (kurt - 1)/4 * SR_daily^2) )
```

For negative skew (`skew < 0`) the `-skew * SR_daily` term becomes positive and *inflates* the denominator — shrinking the test statistic. Same for excess kurtosis (`kurt > 3`). DSR is being conservative: it doesn't trust a Sharpe more than it has to when the return distribution is non-Gaussian.

Our reproduced strategies are heavily fat-tailed and slightly left-skewed: a few huge winners carry the mean, and the worst days are larger than a Gaussian would predict. The fat-tail penalty hits the test statistic, not the threshold.

## What PSR is doing geometrically

Three steps:

1. **Shift**: subtract the expected-max-by-luck from the observed Sharpe.
2. **Standardize**: divide by `sigma_max` (with the skew/kurt deflation in the denominator) to get a z-score.
3. **CDF**: pass through `Phi` to get a probability.

`PSR > 0.95` means: under the null of zero skill, you'd see a Sharpe this high in fewer than 5% of repeated experiments at this N. That's the bar.

## The verdict for our reproduction

```
SR_daily          ~= 0.0321
SR_threshold      ~= 0.0583
PSR               ~= 0.23
```

The best of 108 models is *below* the luck-only ceiling. PSR of 0.23 is nowhere near 0.95. We don't have evidence of an edge — and that's before we even run the random-baseline test (a separate falsification).

## The most common misuse

The single most frequent Sharpe-reporting sin: quoting a number without quoting `N_trials`. The original research document tested 72 configurations and presented the best one. The reported Sharpe was never deflated by 72. Under DSR with `N = 72`, the threshold lands in the same neighborhood as our `N = 108` case — high enough to disqualify most of what got published.

This is exactly the pattern DSR was designed to catch: best-of-N selection masquerading as a single experiment. Always report N. Always deflate.

> See also: [/story/06_statistics](/story/06_statistics) — interactive walkthrough of the variance, threshold, and PSR for our 108-config sweep.
