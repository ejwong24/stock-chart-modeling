# Why weekly anchors, not daily — autocorrelation, practical rebalancing, and the dataset size knob

The choice of anchor cadence is one of those quiet methodological decisions that doesn't *look* important in a config file but ends up driving everything: how big our training set is, how honest our confidence intervals are, how much our walk-forward embargo eats, and whether the live system can actually be traded by a human. We picked **weekly** — one labeled row per ticker per ISO calendar week, on Fridays — and the rest of this note explains why.

## The choice

Each ticker contributes **one `(anchor_date, label)` row per ISO calendar week** that has at least 3 trading days. That's roughly **52 anchors per stock per year**. Across our universe of ~6,500 stocks and 8 years of history, that's about **2.6M anchors** before any filtering. After the MA250 prefilter, we land at **~200k anchors** — the actual training set the model sees.

## Why not daily anchors

The naïve move is "more data is always better." That fails badly here because we're using a **40-trading-day forward return** as the label. Two consecutive daily anchors would share **39 of their 40 forward days**:

- **Autocorrelation is near 1.** `cov(label[t], label[t+1])` is close to `var(label)` — thousands of "rows" that are really one observation in a trench coat.
- **Effective sample size collapses.** Daily anchors with H=40 give each row uniqueness ≈ 1/40. Nominal 1M rows → effective ~25k.
- **Bootstrap CIs widen dramatically** once you do the overlap accounting properly.
- **Walk-forward embargo balloons.** Every label touches 40 forward days.

Daily anchors look like 200× the data on paper. In effective-N terms they're maybe 2–3× — and you've paid for it with worse leakage hygiene.

## Why not monthly anchors

The other end of the spectrum is monthly. That would cut the dataset by **~4×** versus weekly: roughly **50k anchors** instead of 200k. Two problems:

1. **Overfitting risk goes up.** 50k rows across ~6,500 tickers and 8 years is thin for a tree-based model.
2. **Worse alignment with the trading style.** Monthly rebalancing is a different strategy.

## The compromise

Weekly hits the sweet spot:

- **~5× reduction in autocorrelation vs daily** (uniqueness goes from ~1/40 to ~1/8 with H=40 and a 5-day stride)
- **4–5× more data than monthly**, which matters for tree models that need enough leaves to generalize
- **Aligns with how the strategy will actually be traded** — most retail and swing-trading rebalance cadences are weekly

## The López de Prado effective sample size

For our weekly anchors with H=40 trading days held:

- **Naive trade count:** 1,104 trades
- **Effective N:** ~44 (because ~25 concurrent positions on any given day)
- **SE(Sharpe) widening factor:** `sqrt(1104 / 44) ≈ 5.0`

That widening factor is what feeds the deflated Sharpe machinery downstream.

## Execution implications of the weekly grain

The simulator assumes **Friday close fills** at `anchor_close`. In practice a human executes sometime Friday afternoon or at Monday open. This is **slightly optimistic** versus realistic execution — Monday open is typically a few bps to a few tens of bps off Friday close — but close enough.

## The ISO-week rule

We require at least **3 trading days in the week** to emit an anchor. This drops holiday-shortened weeks like Thanksgiving. `tests/test_labels.py::test_label_one_weekly_anchor_rule` confirms that the key `(year * 100 + iso_week)` is unique per ticker.

## Forward paper trading matches

The live forward-paper system runs **every Friday** and emits next week's top-5 picks. Train, backtest, and live all share one clock.

---

> **See also:** [/story/09/10](/story/09/10) — uses the effective-N from this section as a deflator. [/story/09/06](/story/09/06) — the simulator that consumes the weekly anchor.
