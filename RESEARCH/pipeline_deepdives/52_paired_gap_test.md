# The paired-bootstrap gap test — finally measuring the central claim

For most of this project's life, the report card had a hole in the middle. The one question that justifies the whole effort — *does the model beat the best simple momentum baseline?* — was answered with a single point number, like "+5.26% CAGR," and a `gap_ci` field that literally said `TODO`. We printed the difference and quietly hoped you wouldn't ask whether it was real. This page is about the function that finally closes that hole: `stats.paired_gap_bootstrap`.

## Why paired, not two independent CIs

The naive way to compare model and baseline is to bootstrap a confidence interval for each curve separately and eyeball whether they overlap. That is wrong, and not by a little. The model and the [best simple baseline](/story/09/20_simple_baselines) trade the same universe over the same period; on any given day they are both long the same broad market move. A market-wide +2% day inflates *both* equity curves. Two independent CIs treat that shared move as independent noise in each, so each interval is wide — and the comparison loses almost all of its power.

The fix is to difference *before* you bootstrap. We align the curves by date and take the per-day model-minus-baseline simple-return difference:

```python
j = m.merge(b, on="date", suffixes=("_m", "_b")).sort_values("date")
rm = j["equity_m"].astype(float).pct_change().to_numpy()[1:]
rb = j["equity_b"].astype(float).pct_change().to_numpy()[1:]
d = rm - rb          # paired daily gap; shared market exposure cancels
```

The merge-on-date is deliberate: the two equity series do not necessarily share a row index, and aligning by position instead of date would silently compare mismatched days (the [row-alignment](/story/09/41_row_alignment) trap). Once differenced, the common market factor is gone and what remains is the *daily edge*.

## Why block, not iid

You cannot resample the daily gap series as if each day were independent. With a ~40-day holding horizon, positions overlap heavily, so today's gap is correlated with yesterday's and tomorrow's. An iid bootstrap would shatter that autocorrelation, understate the variance, and hand you a falsely tight CI. We use a Politis–Romano stationary block bootstrap instead — resampling contiguous geometric-length blocks so the within-horizon dependence survives:

```python
p = 1.0 / max(block_size, 1)
while t < n:
    start = int(rng.integers(0, n))
    L = int(rng.geometric(p))          # geometric block length, mean = block_size
    take = min(L, n - t)
    ...
ann[i] = float(d[idx].mean() * 252.0)
```

`block_size` defaults to 40 — the holding horizon, the natural autocorrelation length of overlapping holds. The reasoning lives in [block bootstrap params](/story/09/22_block_bootstrap_params).

## Annualized arithmetic gap, not CAGR-difference

Each resample reduces to `mean(daily gap) × 252` — the **annualized arithmetic** alpha. This is *not* the same object as "model CAGR minus baseline CAGR," and the difference matters. CAGR is a geometric, path-dependent endpoint statistic; subtracting two CAGRs gives a number whose sampling distribution is awkward. The mean daily gap is a clean linear functional of the paired series, so its bootstrap distribution is well-behaved and interpretable as "the annualized edge per dollar exposed." The report card still *prints* the CAGR-difference as the headline `gap_point`, but the **significance** verdict rides on the arithmetic gap. See [Sharpe](/story/09/27_sharpe_ratio) and [CAGR / drawdown / Calmar](/story/09/28_cagr_drawdown_calmar).

## The one-sided p-value

The null is directional: H0 is *gap ≤ 0* (the model does not beat the baseline). So the p-value is the fraction of bootstrap resamples in which the annualized gap landed at or below zero:

```python
"p_value_gap_le_0": float((ann <= 0).mean())
```

A large p means a meaningful share of plausible worlds have the model *losing*. No parametric assumption — a direct read of the resampled distribution.

## The honest result

On the real `full` run the verdict is uncomfortable. The point gap is +5.26% CAGR, but the paired bootstrap returns a 95% CI of roughly **[-15.1%, +49.3%] annualized** with **p(gap ≤ 0) ≈ 0.27**. The interval straddles zero by a wide margin; better than a quarter of resampled worlds have the model behind. The headline edge is **not distinguishable from zero** — and that's *before* any multiple-comparison penalty. This number drives [the verdict](/story/09/55_the_verdict).

## The caveat that makes it worse

Even p ≈ 0.27 is optimistic. The "best baseline" we compare against is itself a **max over K** simple baselines — we picked the one that looked strongest. That selection inflates the apparent gap, so the raw paired p still needs a [multiple-comparison adjustment](/story/09/23_multiple_comparison_landscape), via the [deflated Sharpe ratio](/story/09/10_deflated_sharpe) or Hansen's SPA. The report card's `gap_significant` flag refuses to ever claim a clean win:

```python
if p < 0.05 and lo > 0:
    gap_significant = f"Y (raw paired p={p:.3f}; NOT yet N_trials-adjusted)"
else:
    gap_significant = f"N (paired p(gap≤0)={p:.3f}; CI straddles 0)"
```

## How it's wired in

`auto_fill_from_run` reads the best track's equity and the best `rank_*` baseline's equity, calls `paired_gap_bootstrap(model_eq, bl_eq, n_resamples=2000, block_size=40)`, and renders the result into the `gap_ci`/`gap_significant` cells. The old `TODO` is gone. Four regression tests pin the behavior: a dominant model → CI entirely above zero; equal curves → CI straddles zero with p ≈ 1.0; misaligned dates forced to align by the merge; short inputs → empty result. See [the hardening story](/story/09/39_hardening_story) and the [roadmap](/story/09/47_roadmap).
