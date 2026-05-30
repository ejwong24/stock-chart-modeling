# Numerical stability patterns across the pipeline

Quantitative pipelines fail in a particular way: not with a stack trace, but with a `NaN`, an `inf`, or a complex number that slips through a comparison and silently corrupts a training label, a backtest summary, or a rendered image. This deep-dive surveys the numerical-robustness techniques used across `src/stock_chart` — and the bugs that motivated each one. Every example is real code.

## Division guards: epsilon floors over conditionals

The most common hazard is dividing by a quantity that can be zero. `features.py` defends every ratio with a tiny additive epsilon in the denominator:

```python
f["log_dollar_vol_z252"] = float((log_dv[-1] - log_dv.mean()) / (log_dv.std() + 1e-9))
f["vol_ratio_20_252"]   = float(v[-20:].mean() / (v.mean() + 1e-9))
f["vol_21d_div_252d"]   = f["vol_21d"] / (f["vol_252d"] + 1e-9)
f["beta_spy_63d"]       = float(cm[0, 1] / (cm[1, 1] + 1e-12))
```

The path-dependent labels in `labels.py` use `clip` instead of addition, which protects against both zero and tiny negatives:

```python
abs_mae = mae.abs().clip(lower=EPS)          # EPS = 1e-4
out[f"mfe_mae_ratio_{H}d"]  = (mfe / abs_mae).astype(np.float32)
out[f"sortino_label_{H}d"]  = (ret.clip(lower=0) / abs_mae).astype(np.float32)
```

Why the epsilon and not `if denom == 0: return default`? A conditional only catches *exactly* zero, leaving subnormal denominators to produce `1e308`-scale ratios. A floor of `1e-9`/`1e-12`/`1e-4` is dimensionally negligible against real volatilities and variances, branchless, and vectorizes cleanly over a whole pandas column.

## log/sqrt of non-positive arguments

`log(x)` is `-inf` at zero and `nan` below it; `sqrt(x)` is `nan` below zero. Three fixes guard the argument *before* the call.

`stats.bootstrap_cagr_ci` floors the equity curve:

```python
eq = np.clip(eq, 1e-9, None)
r = np.diff(np.log(eq))
```

A single zero/negative equity point — a catastrophic blow-up — would make `np.log` emit `-inf`, and because that `-inf` feeds the resampled returns array, it poisoned *every* bootstrap statistic, not just the one resample containing it. ([label corruption](/story/09/43_label_corruption) is the analogous data-side failure.)

`labels.py` guards `anchor_close <= 0` so a non-positive anchor never reaches `np.log`, and `render.py` clips ratios before the log and synthesizes a finite axis range when the price window is degenerate (see [fixed log-y axis](/story/09/29_fixed_log_y_axis)):

```python
log_ratios = np.log(np.clip(ratios, 1e-6, None))
if y_hi <= y_lo or not np.isfinite(y_hi - y_lo):
    y_hi = y_lo + 1.0   # synthesize a unit range so the math is finite
```

## Negative base to a fractional power: the complex-number trap

CAGR compounding raises an equity multiple to `1/years`. If cumulative losses drive the account *below* its starting value by more than 100%, the base goes negative — and a negative float to a fractional power in Python is not an error, it is a `complex`. `float(complex)` then raises `TypeError`, crashing `stats.post_tax_cagr`. The fix floors the base:

```python
pre_base  = max(1 + df["pnl_dollars"].sum() / start_equity, 0.0)
post_base = max(equity / start_equity, 0.0)
pre  = float(pre_base ** (1 / years) - 1)
post = float(post_base ** (1 / years) - 1)
```

`0 ** x - 1 == -1.0` is not a hack — a wiped-out account *is* a -100% CAGR, so zero is the semantically correct floor. See [post-tax CAGR / frictions](/story/09/25_frictions_beyond_impact) and [CAGR/drawdown](/story/09/28_cagr_drawdown_calmar).

## Reductions on constant or length-1 series

`std()` and `var()` are zero on a constant series and undefined on a single element, which detonates any Sharpe, skew, or beta that divides by them. The pattern is to guard the spread before using it. `_annualized_sharpe`:

```python
if len(daily_returns) == 0 or daily_returns.std() == 0:
    return 0.0
```

The bootstrap resample guard, `features.py` skew/kurtosis, and the beta covariance all do the same (`if rs.std() > 0 else 0.0`). See [Sharpe](/story/09/27_sharpe_ratio) and [block bootstrap](/story/09/22_block_bootstrap_params).

## NaN/Inf containment at the module boundary

`features.compute_for_anchors` routes every output column through `_safe` before returning:

```python
def _safe(s, fill=0.0):
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill)
...
for c in FEATURE_COLS:
    out[c] = _safe(out[c])
```

This is a *containment boundary*: no matter what a per-window computation produces, the public API never emits a non-finite value into the model. The tradeoff is real and worth stating plainly — silently zeroing a non-finite value can mask a genuine bug. That is exactly how the beta bug hid: an off-by-one made the length guard never pass, beta fell through to `0.0`, and `_safe` made the zero look like a legitimate feature value rather than a defect. See [why beta was zero](/story/09/42_beta_zero_bug). Containment should coexist with *alerting*, not replace it.

## NaN comparison semantics

`NaN >= x` is `False` and `NaN < x` is `False`. This cuts both ways. The simulator depends on it as a *trap to defend against* — a NaN ADV would silently slip past the `adv < min_adv_usd` filter (the comparison is `False`, so it isn't rejected) and crash later, so the code tests finiteness explicitly:

```python
if not np.isfinite(adv) or adv < cfg.min_adv_usd:
    continue
```

`labels.py` relies on the same semantics defensively: `(ret >= T)` on a NaN return yields `0`, keeping a meaningless row out of the positive class rather than flipping it. The rule: a NaN must never *silently pass* a comparison filter. See [the simulator loop](/story/09/06_simulator_loop).

## Integer vs float

Shares are integers; dollars are floats. The simulator floors to whole shares with `//`, deliberately leaving residual cash uninvested rather than over-spending:

```python
shares = int(target_usd // fill)
if shares <= 0:
    continue
```

Control flow never tests float equality; it uses `<= 0`, `> cash`, and `>= T` thresholds so accumulated rounding error can't make a `==` branch flake. See [position size](/story/09/32_position_size_formula).

## The checklist

- **Every divide** → guard the denominator (`+1e-9`/`+1e-12`, or `clip(lower=eps)`).
- **Every log/sqrt** → guard the argument's sign before the call.
- **Every fractional power** → floor the base at zero.
- **Every reduction** (`std`/`mean`/`cov`/`var`) → guard empty and constant inputs.
- **NaN never silently passes a comparison filter** → test `np.isfinite` first.
- **Contain non-finite at module boundaries, but ALERT — don't just zero** — a value that's unexpectedly non-finite is a bug, not a default.

These patterns are the connective tissue of [the hardening story](/story/09/39_hardening_story); the tests that pin them down are [the testing philosophy](/story/09/45_testing_philosophy).
