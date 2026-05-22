# The 5 simple baselines — what they rank by and why each matters

## Motivation: setting the bar to clear

Before claiming that a learned model has discovered something useful, you have to answer a brutal question: can it beat a one-line trading rule? If a thirty-character pandas expression — "sort by 12-month return, take the top five" — produces returns comparable to a multi-million-parameter gradient-boosted ensemble, then the model isn't finding signal.

The five simple baselines in `src/stock_chart/random_baseline.py` exist for exactly this reason. Each ranks the prefiltered universe by a single deterministic momentum statistic and picks the top five every week — same backtest machinery, same costs, same liquidity filters, but with **zero learned parameters**. They are the bar to clear.

## rank_252d_return

Sort candidates by `ret_252d` (year-over-year price return) and buy the top 5. The canonical "buy what's been winning the most over the past year" rule. Since Jegadeesh & Titman's 1993 paper on cross-sectional momentum, this exact strategy has been documented to earn positive risk-adjusted returns across decades and most equity markets. It is arguably the most replicated anomaly in empirical finance, and therefore the baseline to fear most.

## rank_60d_return

Sort by `ret_63d` (roughly a quarterly return). Captures **recent** momentum rather than annual. Empirically, 60d momentum tends to outperform 252d momentum in faster-rotating markets (post-crash recoveries, regime changes) and underperform during steady-grinding bull markets.

## rank_ma250_extension

Sort by `ratio_ma200` (close / 200-day SMA). The most "extended" names. Closely related to `ret_252d` but normalized by a smoothed trailing average — less sensitive to a single bad day a year ago, more sensitive to *sustained* trend strength.

## rank_52w_high_distance

Sort by `pct_from_252d_high` — the stock currently 5% below its 52-week high beats the stock 30% below. The classic Mark Minervini / William O'Neil "stocks at new highs" strategy. A stock pressed against its annual high is one whose overhead supply has been fully absorbed.

## rank_inv_60d_vol

Sort by **negative** `vol_63d` — lower volatility wins. The counterintuitive "boring quality" baseline. Within a universe already prefiltered for momentum, the smoothest names tend to be the highest-quality compounders. Low-volatility-within-momentum is a documented factor tilt.

## Why these five

Each captures a *different* slice of momentum:

- **252d return** — annual price momentum
- **60d return** — quarterly price momentum
- **ma250 extension** — trend strength at the moving-average level
- **52w high distance** — "still near peak" momentum
- **inv 60d vol** — "smooth" momentum (quality tilt)

If the learned model can't beat *any* of these, it has no edge over plain technical analysis.

## The result

From the realistic-cost run, `lgbm_engineered` beats the **average** baseline by 1-2 percentage points of CAGR. Beats the worst (`rank_inv_60d_vol`) by ~4pp. Loses to the best (`rank_252d_return`) by ~3pp in some folds.

## The verdict

The AI model is in the *same league* as the simple baselines — not dramatically better, not dramatically worse. After realistic transaction costs and deflated-Sharpe correction, the gap is statistically indistinguishable from zero. This is the humbling result that drives the falsification chapter.

## Implementation note

Each baseline is just `_score_from_features(feats, baseline_name)` returning a pandas Series. The same simulator wraps each one identically — same universe, same rebalance cadence, same slippage and commission model. Directly comparable on every metric.

---

> **See also:** [/story/05_falsification](/story/05_falsification). The engineered-features track is the only track that meaningfully beats the random and simple baselines after costs.
