# Regime dependence — how a momentum strategy behaves across 2017–2025

Momentum strategies are not edges so much as bets on a *condition*: that recent winners keep winning long enough to be sold at a profit. When that condition holds, the book prints; when it inverts, momentum unwinds violently. So a momentum-on-extension book — MA250 prefilter, top-5 ranked weekly, 40-day holds — cannot be evaluated as one number across 2017–2025. The honest unit of analysis is the *regime*, because the strategy's structure interacts with each regime differently. This is also why the [yearly walk-forward folds](/story/09/08_walkforward_embargo) we use as test windows are not interchangeable: each calendar-year fold is a sample from a different distribution, and some of those distributions are nearly un-tradeable.

*The per-year figures below are illustrative characterizations of regime behavior, not exact loaded backtest outputs; the structural mechanics are what the pipeline actually enforces.*

## The prefilter makes the eligible pool itself regime-dependent

The first thing to understand is that the size of the tradeable universe is not constant. The [MA250 prefilter](/story/09/07_ma250_prefilter) only admits names trading above their 250-day moving average — a coarse "is this in an uptrend at all" gate applied before ranking. In a broad bull, most of the universe clears it, so the weekly top-5 is drawn from a deep, diverse pool. In a crash, the pool collapses.

That collapse is not cosmetic. With a smaller eligible pool, the top-5 concentrates into whatever handful of names are still extended, and effective diversification falls even though the position count is unchanged. The simulator caps each position at 4% of equity and adds at most five new names a week, but it cannot manufacture breadth the prefilter has removed. When the pool is large the five weekly adds are five independent-ish bets; when the pool is tiny they are five correlated bets on the same surviving theme. Regime changes the *quality* of diversification, not just the count.

## 2017–2019: the comfortable steady bull

In the steady bull, broad participation means a large eligible pool and genuine cross-sectional dispersion among uptrending names. This is the regime the strategy was implicitly designed for: ranking has lots of raw material, the top-5 rotates across sectors, and 40-day holds ride orderly trends. Returns here are unremarkable but real — which is precisely why it is dangerous to extrapolate from it.

## March 2020: the prefilter pool collapses

The COVID crash is the cleanest illustration of pool-dependence. As the index falls through its own 250-day average, names fall below their MA250 *en masse*, and the eligible pool can collapse to under 100 names. Momentum whipsaws: names extended one week are uninvestable the next, ranking churns, and the 40-day hold repeatedly buys into reversals. The book is structurally starved of both breadth and trend persistence at once. This isn't a tuning failure; it's the prefilter doing exactly what it should while the ranking layer has almost nothing left to rank.

## 2020–2021: the strategy's best — and most misleading — regime

The recovery and retail-mania era is where a momentum-on-extension book makes its career. Enormous cross-sectional dispersion, biotech/EV/SPAC/meme winners (the BNTX-style multibaggers), and persistent extension mean the top-5 repeatedly captures names that run for months. As discussed in [chapter 9](/story/09_pipeline_walkthrough), results in this regime are typically driven by *one outlier per cohort* — a single 5–10x trade dominating the others. That is the signature of a fat-tailed momentum payoff, and it flatters every headline metric. It also poisons interpretation: a backtest weighted toward 2020–2021 will show a strong CAGR and Sharpe that are essentially the echo of one extraordinary regime.

## 2022: the rate-shock bear and the momentum crash

When the Fed broke the regime in 2022, the momentum factor crashed hard. High-multiple, high-extension names — exactly what the strategy ranks to the top — led the drawdown. This is structurally the worst case: the prefilter keeps admitting names on the way down through brief bear rallies, ranking buys the most-extended (now most-vulnerable) names, and 40-day holds lock in exposure through the decline. This is plausibly where the [−44% max drawdown](/story/09/28_cagr_drawdown_calmar) lived.

## 2023–2025: narrow mega-cap leadership starves a small/mid-cap book

The most recent regime is subtle. Leadership narrows to a handful of mega-caps while breadth stays poor. The problem is that this book is *structurally small/mid-cap-tilted*: the [ADV20](/story/09/33_adv20_metric) liquidity filter and the [capacity ceiling](/story/09/26_capacity_ceiling) push it toward smaller, more-extended names, and the ADV cap holds each position to a fraction of daily volume. It cannot meaningfully own the mega-caps driving the index, so it lags badly — a beta and sizing problem, not a signal problem. See [SPY beta](/story/09/38_spy_beta) for how thin this book's index exposure actually is.

## Why this breaks the backtest, and what honest reporting looks like

The [splits](/story/09/08_walkforward_embargo) treat each calendar year as a test fold with an expanding, purged-and-embargoed training window. Mechanically clean — but it means folds have *wildly different difficulty*. A 2021 fold is a tailwind; a 2022 fold is a headwind; the 2020 fold may be a [degenerate fold](/story/09/48_degenerate_folds) where the collapsed pool leaves too few trades to mean anything. Averaging across these as if they were i.i.d. is the core error.

This is why a single good regime can make a whole backtest look good, and why [effective sample size](/story/09/24_effective_sample_size) and [the verdict](/story/09/55_the_verdict) deliberately discount it: with one dominant regime and outlier-driven cohorts, you have far fewer *independent* bets than trade-count suggests. Against [simple baselines](/story/09/20_simple_baselines), much of the headline edge is 2020–2021 leaking into the average.

The honest implication: a book that only works in momentum-on regimes is a *regime bet*, not an all-weather edge. The fix is not more tuning but better reporting — **regime-conditional metrics** (per-regime CAGR, drawdown, hit-rate, and pool size) alongside the blended numbers, so a strong aggregate cannot hide a single load-bearing year. That belongs on the [roadmap](/story/09/47_roadmap), the natural complement to the [hardening story](/story/09/39_hardening_story): once the mechanics are leak-free, the next layer of honesty is admitting *when* the edge actually shows up.
