# Effective sample size — why 1,104 trades counts as ~44

## The problem: overlapping trades aren't independent observations

Our full backtest over 8 years produces **1,104 round-trip trades** in the `lgbm_engineered` configuration. The naive instinct is to treat each trade as an independent observation of "did the model's pick work?" That gives us N = 1,104 and tight error bars on everything downstream.

That instinct is wrong, and the reason is structural: **the trades overlap in time**. At any given trading day during the backtest, roughly **25 positions are open simultaneously**. Trade #412 and trade #418 might have been opened three days apart and held for 40 business days each — they share 37 days of market exposure. When the market drops 3% on day 20 of their shared window, *both* trades take a hit. Their P&L is not independent.

A [blotter](/story/09/34_blotter_equity_summary) with 1,104 overlapping trades therefore contains far less independent information than 1,104 non-overlapping trades. We need to quantify "far less."

## The fix: López de Prado's average uniqueness

From *Advances in Financial Machine Learning*, chapter 4. For each trade *i*:

1. Walk the hold window day by day.
2. At each day, count **concurrency** — the number of other trades simultaneously open.
3. Define the trade's **uniqueness** as the mean of `1 / concurrency` across its hold days.
4. The **effective sample size** is the sum of uniqueness values across all trades.

```
uniqueness_i  = mean( 1 / concurrency(t) ) for t in hold_window_i
effective_N   = sum( uniqueness_i for i in trades )
```

A trade that lives alone in the market contributes 1.0 to effective N. A trade that lives alongside 24 other open positions contributes ~1/25 = 0.04. The metric collapses gracefully to N when there's no overlap, and to ~1 when everything overlaps everything.

## The math, in our pipeline

```
# 1,104 trades, ~25 average concurrent positions, 40-day hold
uniqueness_i  ≈ mean(1/25 over 40 days) = 0.04
effective_N   ≈ 1,104 × 0.04 = 44
```

We made 1,104 trades. We effectively made **44 independent bets**.

## SE(Sharpe) widening

Standard errors scale as `1/sqrt(N)`. Going from naive N=1,104 to effective N=44 widens SE(Sharpe) by:

```
sqrt(1104 / 44) = sqrt(25.1) ≈ 5.0x
```

The practical consequence is brutal. A Sharpe of 0.5 that looked like:

```
Naive 95% CI:        [ -0.05,  +1.05 ]   ← "borderline significant edge"
```

becomes:

```
Effective-N 95% CI:  [ -2.50,  +3.50 ]   ← "we have no idea"
```

The "significant edge" verdict doesn't survive contact with effective N.

## Why this matters more than DSR alone

**DSR and effective N correct for different things, and both apply:**

- **DSR** corrects for trying 36 configurations and reporting the best. It deflates the headline Sharpe for *selection across configs*.
- **Effective N** corrects for the fact that *within a single config*, the 1,104 trades supplied less information than 1,104 independent observations. It deflates the *precision of each evaluation*.

A backtest can pass DSR (you actually had an edge versus selection bias) and still fail effective-N significance. Ours does both.

## Concrete numbers from our pipeline

| Metric | Value |
|---|---|
| Total trades (lgbm_engineered) | 1,104 |
| Mean concurrency | ~25 positions |
| Effective N | 44 |
| SE(Sharpe) widening factor | 5.0x |
| Observed Sharpe | 0.510 |
| Naive 95% CI | [+0.41, +0.61] |
| Effective-N 95% CI | [-0.97, +1.99] |
| Effective-N + DSR threshold | **not significant** |

## The implementation

`effective_sample_size(blotter_df, hold_days=40)` in `src/stock_chart/stats.py`:

1. Builds a **date-indexed concurrency series**.
2. For each trade, computes `uniqueness = (1 / concurrency).mean()` over hold window.
3. Returns `sum(uniqueness)`.

Bug #1 regression: the empty-[blotter](/story/09/34_blotter_equity_summary) path used to crash on a `NaT`-only `date_range`. The function now short-circuits to `0.0`.

## Interpretation

> We made 1,104 trades, but we effectively saw **44 independent decisions**.

Most trades were redundant — the model kept reaching for "the same kind of stock at the same kind of moment." The pipeline is not averaging over 1,104 distinct opportunities. It is averaging over 44.

## The tests

- `test_effective_sample_size_no_overlap` — non-overlapping trades return `N == N_trades`.
- `test_effective_sample_size_full_overlap` — 20 simultaneous trades collapse to `N ≈ 1`.
- `test_effective_sample_size_empty_blotter` — Bug #1 regression.

---

> **See also**
> - [/story/06_statistics](/story/06_statistics) — uses effective N as a deflator.
> - [/story/09/10](/story/09/10) — the complementary DSR deflation.
