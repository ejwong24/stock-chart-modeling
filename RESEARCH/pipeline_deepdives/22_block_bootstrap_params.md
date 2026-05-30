# Block bootstrap parameters — block size, n_resamples, stationary vs circular

The point estimate of a backtest tells you almost nothing. A Sharpe of 1.2 on a 2000-day series could be drawn from a true distribution centered anywhere from 0 to 2. The honest answer is a confidence interval, and the only credible way to produce one from a single observed equity curve is the block bootstrap. This page documents the choices baked into `src/stock_chart/stats.py`.

## Why a block bootstrap, not IID

The IID bootstrap resamples individual days with replacement. That is fine when observations are independent — but daily equity returns are emphatically not. A high-volatility regime persists for weeks. A bear market lasts months. Drawdowns cluster. If you shuffle individual days, you destroy every one of these structures and end up with a synthetic series whose statistics look much cleaner than reality. CIs come out wildly too tight, sometimes by a factor of 3.

The block bootstrap preserves local autocorrelation by sampling **contiguous chunks** of the series. Each chunk carries its own volatility regime, drawdown shape, and short-term momentum with it.

## The stationary block bootstrap

We use Politis & Romano (1994):

1. Pick a uniform random starting index in `[0, N)`.
2. Sample consecutive days for a **geometric** number of steps with mean `block_size`.
3. When the geometric draw expires, jump to a new random start and repeat.
4. Continue until the synthetic series has length `N`.

The key feature is that block lengths are **random, not fixed**. This makes the resampled series stationary in distribution — the marginal distribution at every position is identical, which is not true for the fixed-length (moving) block bootstrap.

## Why `block_size = 40`

Our pipeline holds positions over a roughly 40-trading-day horizon (`H=40`). Returns inside that window are correlated — the same trade is open, the same volatility regime is in force. Returns 40+ days apart are roughly independent. The right block size is the **autocorrelation length** of the returns. So `block_size = 40` tracks the natural correlation timescale of the data.

## Sensitivity to `block_size`

- Too small (e.g. 5): blocks don't capture full volatility regimes, CIs come out too tight.
- Too large (e.g. 200): too few independent blocks per resample, sampling variance explodes.
- Sweet spot: the autocorrelation length.

For `lgbm_engineered`:

| block_size | CAGR 95% CI | Sharpe 95% CI | max_dd 95% CI |
|------------|-------------|---------------|---------------|
| 10         | [-2.8%, +11.1%] | [-0.05, +1.08] | [-61%, -31%] |
| 20         | [-3.6%, +12.0%] | [-0.13, +1.16] | [-64%, -29%] |
| **40**     | [-4.2%, +12.6%] | [-0.18, +1.21] | [-67%, -28%] |
| 60         | [-4.5%, +12.9%] | [-0.21, +1.24] | [-68%, -27%] |

CAGR CI shifts under 2pp across the sweep — conclusions are robust.

## Why `n_resamples = 10000`

Monte Carlo error scales as `1 / sqrt(n_resamples)`. At `n = 10000`, MC error on a 95% percentile CI endpoint is ~0.5 percentage point — well below the natural data uncertainty (CI width is 15+ pp). Going to 100,000 shrinks MC error 3x at 10x cost. 10,000 is the sweet spot.

## Alternatives we don't use

- **Circular block bootstrap.** Wraps the series end-to-start. For our 2000-day series the difference is negligible; we prefer stationary for cleaner distributional properties.
- **Moving block bootstrap.** Fixed-length blocks, random starts. Similar empirical properties but the resampled series is not stationary.
- **Wild bootstrap.** Multiplies residuals by random ±1 weights. Useful for heteroskedastic regression but does not preserve autocorrelation — wrong tool for equity curves.

## Concrete numbers for `lgbm_engineered`

- **CAGR 95% CI:** `[-4.2%, +12.6%]` (point estimate `+4.3%`)
- **Sharpe 95% CI:** `[-0.18, +1.21]` (point estimate `+0.51`)
- **max_dd 95% CI:** `[-67%, -28%]` (point estimate `-44%`)

These are wide. That is the honest answer to "what is the model's true CAGR" — the point estimate sits well within the noise band.

## Performance

10,000 resamples of a 2000-day return series runs in ~1.5 seconds. `tests/test_idempotency_perf.py::test_bootstrap_throughput` asserts `< 2 seconds`.

---

> **See also:** [/story/09/10](/story/09/10) — the [Deflated [Sharpe Ratio](/story/09/27_sharpe_ratio)](/story/09/10_deflated_sharpe). [/story/06_statistics](/story/06_statistics) — the broader framework.
