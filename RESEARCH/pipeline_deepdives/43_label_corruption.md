# Non-positive prices and label corruption — when inf becomes a buy signal

A supervised model is only as honest as its labels. This chapter is about a single missing positivity check in `src/stock_chart/labels.py` that could silently stamp a **buy** label onto exactly the stocks the project goes out of its way to include — the dead ones. The bug is fixed now, but it is worth dissecting because it sits at the intersection of three things this series cares about: floating-point arithmetic, survivorship-aware data, and the falsifiability of the whole experiment.

## The bug

`label_one` turns a price series into forward-return labels. For each [weekly anchor](/story/09/18_weekly_anchors) and each horizon `H`, it computed the forward return as a plain ratio:

```python
ret = end_close / anchor_close - 1.0      # original
log_ret = np.log(end_close / anchor_close)
```

There was no guard on `anchor_close`. If the adjusted close at an anchor day was `0.0`, then `end_close / 0.0` is `inf` (IEEE-754, no exception by default), and `ret = inf - 1.0 = inf`. The binary labels in the [36-label grid](/story/09/19_label_grid) are built with a `>=` comparison:

```python
out[f"ret_{H}d_ge_{t_str}"] = (ret >= T).astype(np.int8)
```

And `inf >= 0.25` is `True`. So a meaningless division produced a confident `1` — a **false positive training target**. No exception, no NaN, no warning. The poisoned label was written straight to parquet and handed to the model.

What makes this especially nasty is that the path-statistics code *did* have the guard. Inside the MFE/MAE loop there is an explicit `if closes[a] <= 0:` branch. The author clearly knew zero closes were possible. The return and label lines simply didn't get the same treatment — a classic case of a defensive check applied in one place and forgotten three lines up.

## Why a close can be ~0 in the first place

This is not a contrived edge case. As covered in [data acquisition](/story/09/13_data_acquisition), prices come from yfinance with `auto_adjust=True`, which back-propagates cumulative split and dividend factors through the entire history. On a long-lived, heavily-split or richly-dividend-paying name — and especially on an illiquid or near-delisting ticker — that cumulative factor can drive early adjusted closes to round to `0.0` in float64, or a feed glitch on a near-zero penny stock can emit a literal zero. And by design, [universe construction](/story/09/14_universe_construction) *deliberately retains delisted tickers* to defeat survivorship bias. The very names most likely to carry a zero close are the ones we most insisted on keeping.

## The asymmetry that matters

The fix is not "drop anything with a zero price." There is a real distinction between the two ends of the return:

- **A non-positive *anchor* close is meaningless.** You cannot buy a stock at $0. A return measured from zero is undefined, not extreme. These rows are dropped.
- **A non-positive *end* close is real.** It is a genuine **−100% total loss**: `ret = 0/anchor − 1 = −1`. That is one of the most valuable observations the survivorship-corrected universe can offer — the model *should* learn what a wipeout looks like. We keep it as `ret = −1` and only sanitize its `log_ret`, since `log(0) = −inf`.

So the patch encodes that asymmetry directly:

```python
with np.errstate(divide="ignore", invalid="ignore"):
    ret = end_close / anchor_close_arr - 1.0
    log_ret = np.log(end_close / anchor_close_arr)
ret = np.where(anchor_close_arr > 0, ret, np.nan)        # bad anchor -> NaN
log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan) # clean -inf from zero end
...
out = out[out["anchor_close"] > 0].reset_index(drop=True) # drop bad anchors
```

`NaN >= T` is `False`, so even before the row is dropped the label is already `0` rather than a phantom `1`. The drop is the belt to that suspenders.

## Why this threatened the project's entire thesis

The point of a survivorship-aware universe is to measure skill *honestly* — to stop the model from looking good only because the losers were deleted from history. This bug did the opposite, with surgical precision. It injected false-positive "this was a buy" labels onto precisely the delisted and heavily-adjusted names. A model trained on that data could post **inflated apparent skill on the corrected universe** — the one slice we trust most — by learning to "predict" winners that were never winners at all. That is the exact failure mode [the falsification chapter](/story/05_falsification) exists to catch: a result that looks like signal but is an artifact of label construction.

## The regression test

The test pins the properties the fix guarantees. A synthetic series gets a block of zero closes forced into it (a corporate-action artifact), then we assert the invariants: no surviving anchor has `anchor_close <= 0`; every `fwd_ret` is finite; and no binary label is `1` where the underlying finite return doesn't justify it.

## Lesson

Every division by market data needs a positivity guard, because `inf >= threshold` is a **silent label-flip**, not a crash you'll notice. Reach for `np.errstate` and `np.where` rather than trusting that "prices are positive" — adjusted prices are a derived quantity and derived quantities go to zero. And when you guard one consumer of a value (the MFE/MAE loop), grep for every *other* consumer before you move on. See [numerical stability patterns](/story/09/44_numerical_stability) for the family this belongs to, and [the hardening story](/story/09/39_hardening_story) for how it was found.
