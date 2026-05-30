# Portfolio construction — from scores to the weekly five-name book

Every Friday the model finishes its job and hands off an artifact that is, by itself, untradable: a long table of `(ticker, anchor_date, score)` rows — one probability-like number per name per week. The score says "this chart pattern resolved up H days later in the training data." It does not say how many shares to buy, whether the name is liquid enough to fill, or whether you already own it. Turning scores into a book is a **separate, deterministic, rule-based stage** that lives entirely in `simulate()`. The model proposes; the funnel disposes.

## The Friday funnel, in order

The simulator's outer loop walks the master calendar one day at a time. On a day that carries anchors (an entry Friday), it pulls that day's candidate slice and pushes each row through seven gates:

**1. Rank by calibrated score, descending.** The first thing `simulate()` does is sort by `(anchor_date, score desc)`, and each anchor group is re-sorted on the way in. Order is everything here. We rank *before* we filter, so that when later gates remove a name, they remove it **from the top of the queue** and the next-best candidate slides up. Note that [isotonic calibration](/story/09/30_isotonic_calibration) sits upstream of this sort and is irrelevant to it: isotonic regression is monotonic, so it is **rank-preserving**. Calibrating the scores does not reshuffle a single position in the queue. (That fact is also a quiet confession — see gate 7.)

**2. ADV liquidity filter.** Drop any candidate whose `adv20_usd < $5M`. In code this is the `adv < cfg.min_adv_usd` skip, guarded by an `np.isfinite` check because a NaN ADV compares `False` against everything and would otherwise sail through. This keeps the strategy out of names it could never actually trade without becoming the tape. See [ADV20](/story/09/33_adv20_metric).

**3. No-duplicate-ticker.** If `no_duplicate_ticker` is set and the ticker is already in `open_tickers_set`, skip it. We will not stack a second BNTX on the BNTX we already hold. Same-day rotation is still allowed — a name that *exits* today can be re-entered today — but concurrent doubling-up is forbidden.

**4. Halt-risk fill failure.** When enabled, each surviving candidate draws against a tier-dependent halt rate (`micro` 1.8%, `small` 0.8%, …). A failed draw means the entry simply never fills — no position, no P&L, the capital stays in cash. This is the probabilistic-miss member of the [frictions beyond impact](/story/09/25_frictions_beyond_impact) family.

**5. Position sizing.** The target dollar size is `min(equity × 4%, max_adv_pct × ADV, cash)`. The equity cap bounds single-name risk; the ADV cap (0.5% of $-volume in the shipped config) bounds your market footprint; the cash cap is the hard wall of what you have. Below a $100 floor the trade is dropped as noise. Shares are floored to whole units after slippage, and if commission tips you over cash, the size is shaved again. Details in the [position-size formula](/story/09/32_position_size_formula).

**6. Cap at five new per week.** `new_count >= cfg.max_new_per_week` breaks the candidate loop. Five fresh names, every Friday, no more. Combine that with ~40-day holds: a name entered this Friday is still open for roughly eight weekly cycles, so 5 × ~8 ≈ **25 concurrent positions**, at 4% nominal each → ~100% deployed. The 5 / 4% / 40d triple is not arbitrary — it is solved jointly so the book fills exactly to its [allocation budget](/story/09/36_allocation_budget) without leverage and without chronic idle cash.

**7. Cohort overlap.** Because holds span eight entry Fridays, **eight weekly cohorts are always in flight at once**. That is the structural risk the whole funnel is built around — and where the BNTX lesson bites. One outlier inside a single cohort can dominate the book's path, because the gates control *count* and *footprint*, not *correlation between simultaneously-held bets* (see the [chapter 9 walkthrough](/story/09_pipeline_walkthrough)).

## Why the order, and why these numbers

Rank-then-filter is the load-bearing choice. If you filtered first and ranked the survivors, the five names you buy would still be your five best survivors — but any cap applied mid-stream would bite the wrong end of the list. By ranking globally up front, every gate is a top-down sieve: the book is always "the best names that also clear liquidity, aren't already held, and fit the budget," in that priority.

The overlapping-cohort design is exactly why naive trade counts overstate your real bets. Twenty-five concurrent positions drawn from eight cohorts of the same regime are **correlated**, so the strategy's [effective sample size](/story/09/24_effective_sample_size) is far below its nominal trade count — a Sharpe computed as if trades were independent flatters itself. The funnel limits *names*; it cannot manufacture *independence*. See also [regime dependence](/story/09/58_regime_dependence) for why the pool the funnel draws from is itself regime-dependent.

And the honest note, owed by gate 1: the raw scores out of the [logistic regression](/story/09/04_logistic_regression) are **not calibrated probabilities** in any absolute sense. A score of 0.62 does not mean "62% chance up." Only the **ordering** is trustworthy, which is precisely why the funnel consumes ranks (sort, top-5) and never thresholds on a score's absolute level. Build the book on the part of the signal you can defend.

The objects this stage emits — the trade blotter, the daily equity curve, the summary — are dissected in [blotter / equity / summary](/story/09/34_blotter_equity_summary); the defensive guards (NaN ADV, missing price series, future-bar look-ahead deferral) that keep the loop honest are in the [hardening story](/story/09/39_hardening_story); and whether the whole construction actually paid off is settled in [the verdict](/story/09/55_the_verdict).
