# Log returns vs simple returns — when each is correct, and what our pipeline does

In a backtesting pipeline, "return" is not a single concept. It's at least three: a label, an input to a bootstrap, and a metric for [Sharpe](/story/09/27_sharpe_ratio). Our pipeline uses both simple returns and log returns, and the choice in each spot is deliberate.

## The two definitions

Given consecutive prices `P[t-1]` and `P[t]`:

```
simple return:  r_s = (P[t] - P[t-1]) / P[t-1] = P[t] / P[t-1] - 1
log return:     r_l = ln(P[t] / P[t-1]) = ln(P[t]) - ln(P[t-1])
```

Simple return asks *"how many cents per dollar did I make?"* Log return asks *"what continuously-compounded rate produced this price change?"*

## They agree for small moves

For `r << 1`, the Taylor expansion of `ln(1+r)` gives `r_l ≈ r_s`. They diverge as returns grow:

| Simple | Log     |
|--------|---------|
| +10%   | +0.0953 |
| +50%   | +0.4055 |
| +100%  | +0.6931 |
| -50%   | -0.6931 |

Notice the asymmetry: a -50% simple loss has a log magnitude of 0.693, while a +50% gain has 0.405. That's the math telling you a 50% drawdown needs a 100% recovery to break even (`0.5 × 2.0 = 1.0`).

## Why log returns are mathematically nicer

- **Additive over time.** `r_l(2-period) = r_l(1) + r_l(2)`. Simple returns compound multiplicatively: `(1+r_s,1)(1+r_s,2) - 1`. Anything involving sums over time — bootstraps, moving averages — is cleaner in log space.
- **Closer to Gaussian.** Empirically, daily equity log returns are roughly normal (with fat tails); simple returns are bounded below at -1 and skewed.
- **Symmetry around zero.** `r_l` has range `(-∞, ∞)`, which plays well with Brownian-motion models.

## Why simple returns are practically nicer

- **Direct dollar interpretation.** "+10% simple" means a literal dime per dollar.
- **Portfolio aggregation.** A portfolio return is the weighted sum of *simple* component returns. Log returns do not aggregate cross-sectionally.
- **Labels are simple-return questions.** "Did this stock rise more than 25% in 40 days?" is a binary on `P[t+40]/P[t] - 1 >= 0.25`. Using logs would force every threshold to be translated.

## Where our pipeline uses each, and why

**Labels — simple returns.** The forward-return label is a question about realized percentage gain:

```python
fwd_ret_40d = close[t+40] / close[t] - 1
label_40d_ge_25pct = (fwd_ret_40d >= 0.25).astype(int)
```

See [Label grid](/story/09/19_label_grid).

**Bootstrap CI — log returns.** Our [stationary block bootstrap](/story/09/22_block_bootstrap_params) resamples daily portfolio returns. With logs we can *sum* resampled days to get period returns:

```python
log_rets = np.diff(np.log(equity))   # input to resampler
# inside one bootstrap iteration:
period_log_ret = log_rets[block_indices].sum()
period_simple  = np.expm1(period_log_ret)
```

Doing this with simple returns would require multiplicative compounding inside every bootstrap iteration — slower and more numerically fragile.

**Sharpe — simple returns.** `equity.pct_change()` gives simple daily returns. Since daily moves are typically under 5%, the simple/log gap is negligible:

```python
daily_rets = equity.pct_change().dropna()
sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
```

See [Sharpe ratio](/story/09/27_sharpe_ratio).

**Equity curve — multiplicative.** `equity[t] = equity[t-1] × (1 + r_s[t])` is the natural recursion. We never take logs at the equity level — only when feeding the bootstrap.

## The trap: never sum daily simple returns

```python
# WRONG
monthly_ret = daily_rets.sum() * 21

# RIGHT
monthly_ret = (1 + daily_rets).prod() - 1
# or equivalently
monthly_ret = np.expm1(np.log1p(daily_rets).sum())
```

For a steady 1% daily return, `(1.01)^21 - 1 ≈ 0.232`, but `21 × 0.01 = 0.21` — a 10% relative error that grows with horizon. This is the single most common returns bug in homegrown backtesters.

## The historical norm

Academic finance (CAPM, [Sharpe ratio](/story/09/27_sharpe_ratio), Black–Scholes, GARCH) almost universally uses log returns because additivity and approximate normality make the math tractable. Practitioners often default to simple returns because they match account statements. Both are defensible — our pipeline uses each where it's natural: **simple for labels and reported metrics, log for the bootstrap engine that needs additivity**.
