# Three validation modes — walk-forward, lockbox, forward paper-trading

A single backtest is a confession, not evidence. The stock-modeling pipeline uses three orthogonal validation modes because each one catches a failure the others can't see. Walk-forward gives you fast iteration. The lockbox keeps you honest after iteration. Forward paper-trading is the only thing that knows what year it is.

## [Walk-forward cross-validation](/story/09/08_walkforward_embargo)

This is the bread-and-butter time-series CV. For each `year` in `2018..2025`, train on everything from 2017 through `year-1` and predict on `year`. Eight folds, each one's training window growing by a year.

Implemented in `src/stock_chart/splits.py` with a López de Prado embargo — 40 trading days × 1.6 calendar-day factor = ~64-day buffer.

**Catches:** per-year regime overfitting. A model that prints money in 2021 but loses badly in 2022 shows up immediately as high cross-fold variance.

**Misses:** selection bias. If you pick the best of 36 configs on the same walk-forward folds you used for development, the winner's apparent performance is inflated.

## Lockbox 2025

**One** specific year — currently 2025, always the latest complete year — is set aside *before any model exploration begins*. While iterating, 2025 doesn't exist. All 36 configs get evaluated on open years (2018-2024) via walk-forward. Exactly ONE winner is declared. *Then and only then* does 2025 get unsealed.

Enforcement lives in `src/stock_chart/lockbox.py`. The `claim_lockbox()` function is one-shot per `(horizon, threshold, model)` tuple. After the first claim, subsequent claims raise `LockboxError` unless `allow_overwrite=True` (research/sandbox modes only).

**Catches:** [multiple-comparison](/story/09/23_multiple_comparison_landscape) selection bias. You only spend the lockbox once.

**Misses:** regime change between the historical lockbox and live trading. 2025 still happened in the past.

## Forward paper-trading

This is the only mode that knows what year it is. Every Friday at 4pm ET, `scripts/forward_pick.py` runs via OpenClaw cron and:

1. Loads the latest market data via `yfinance`
2. Computes features for all currently-eligible tickers
3. Loads the locked-in production model (a `joblib` pickle, frozen weights)
4. Predicts probabilities, ranks, applies the ADV liquidity filter, picks the top 5
5. Writes the picks to `data/forward_picks/{date}.parquet`

The trades are paper, not real. Forty trading days later, `scripts/forward_resolve.py` looks up each pick's actual realized return and appends to `data/forward_returns.parquet`.

**Catches:** regime change, data-pipeline drift, behavioral and execution errors. The model trained on 2017-2024 lived in a specific regime: ZIRP, tech-heavy leadership. If 2026's regime is different, forward paper-trading is the only test that flags it.

**Misses:** edge cases the historical data already covered. Three months is not statistically significant.

## Comparison

| Mode | Catches | Misses | Compute cost | Time to result |
|---|---|---|---|---|
| [Walk-forward CV](/story/09/08_walkforward_embargo) | Per-year regime overfit | Selection bias | ~30 min on 8 folds × 36 configs | Same day |
| Lockbox 2025 | Selection bias | Live regime change | ~1 min | Same day |
| Forward paper-trading | Regime change, pipeline drift | Edge cases already in history | Negligible | ~3-12 months |

## When to use each

- **Hyperparameter tuning / model comparison:** walk-forward only. Fast iteration matters.
- **Final paper / honest report card:** walk-forward as in-sample + lockbox 2025 as out-of-sample.
- **Real-money deployment:** lockbox passed first, then forward paper-trading for at least 3 months, *then* live capital.

The three modes are not redundant — they catch genuinely different failure modes, and the order matters: cheap-and-noisy first, expensive-and-honest last.

> See [/research/03_lockbox_protocol](/research/03_lockbox_protocol) for the full lockbox enforcement spec.
