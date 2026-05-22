# The 40 engineered features — what each one is and BNTX's value

This document is a reference for the 40 columns in `src/stock_chart/features.py::FEATURE_COLS`. Every feature is computable from daily price and volume alone, has no forward-looking leak, and is grouped here into six economically coherent buckets. Concrete values are BNTX as of **2021-07-02**, the day the stock was sitting near $230 after a one-year run-up driven by the Pfizer-partnered COVID-19 vaccine.

---

## 1. Returns at multiple horizons (6 features)

**What it captures.** Pure price momentum at six time scales — from a single day out to a full trading year. Mixing short and long horizons separates noise (1-5 day) from regime (63-252 day), and lets the model see whether recent action contradicts or confirms the longer trend.

```text
ret_Nd = close[t] / close[t-N] - 1     # N in {1, 5, 21, 63, 126, 252}
```

**BNTX on 2021-07-02:**

| Feature | Value | Reading |
|---|---|---|
| `ret_1d` | +0.0000 | Flat session, no intraday signal |
| `ret_5d` | -0.0265 | Mild one-week pullback |
| `ret_21d` | +0.0396 | Up ~4% over the prior month |
| `ret_63d` | +0.6635 | Up ~66% over the prior quarter |
| `ret_126d` | +1.0108 | Doubled over six months |
| `ret_252d` | +1.2294 | Up ~123% year-over-year |

`ret_252d = +1.23` means BNTX more than doubled over the prior year — a textbook stand-out momentum candidate, with the quarterly figure (+66%) showing the move is still accelerating, not fading.

---

## 2. Volatility (3 + 1 = 4 features)

**What it captures.** Realized risk and how it is trending. Three windows (21d, 63d, 252d) describe the term structure of volatility; the ratio `vol_21d_div_252d` collapses two of them into a single "is short-term vol elevated vs. its own annual baseline?" number.

```text
vol_Nd            = std(ret_1d over last N trading days) * sqrt(252)
vol_21d_div_252d  = vol_21d / vol_252d
```

**BNTX on 2021-07-02:**

| Feature | Value | Reading |
|---|---|---|
| `vol_21d` | 0.7089 | ~71% annualized over last month |
| `vol_63d` | 0.7494 | ~75% over last quarter |
| `vol_252d` | 0.7659 | ~77% over last year |
| `vol_21d_div_252d` | 0.9256 | Short-term vol slightly below trailing annual baseline |

BNTX is a high-vol name (a typical large-cap S&P stock sits at 0.20-0.30), but the 0.93 ratio says it has actually *calmed down* a touch versus its own past — useful context the absolute number alone would hide.

---

## 3. Drawdown and distance-from-high (5 features)

**What it captures.** How much pain holders are sitting on, and how long they have been sitting on it. Drawdowns are asymmetric features — they decay slowly, peak at zero, and capture investor regret — so they encode information returns alone cannot.

```text
peak_252d              = rolling_max(close, 252)
max_dd_252d            = min((close - peak_252d) / peak_252d) over last 252 days
current_dd_from_peak   = close[t] / peak_252d[t] - 1
days_since_252d_high   = t - argmax(close over last 252 days)
pct_from_252d_high     = current_dd_from_peak   # same definition, kept for clarity
pct_from_21d_high      = close[t] / max(close over last 21 days) - 1
```

**BNTX on 2021-07-02:**

| Feature | Value | Reading |
|---|---|---|
| `max_dd_252d` | -0.4450 | Suffered a 45% peak-to-trough drop within the year |
| `current_dd_from_peak` | -0.0724 | Today sits 7.2% below the 52-week high |
| `days_since_252d_high` | 17 | 52-week high was set ~3 trading weeks ago |
| `pct_from_252d_high` | -0.0724 | Same value, kept for downstream code |
| `pct_from_21d_high` | -0.0724 | The 21-day high *is* the 52-week high |

This is the classic "uptrend in a shallow consolidation" pattern: a brutal drawdown earlier in the year (-44.5%) is now a distant memory, the stock is only 7% off the year-high, and the month-high equals the year-high (because the same recent peak dominates both windows).

---

## 4. Moving-average ratios (3 features)

**What it captures.** Where the price sits relative to short-, medium-, and long-term mean. Three ratios give a compact picture of the trend stack: above all three is "strong uptrend, all timeframes agree"; mixed values mean a regime change is underway.

```text
ratio_maN = close[t] / SMA(close, N)    # N in {20, 50, 200}
```

**BNTX on 2021-07-02:**

| Feature | Value | Reading |
|---|---|---|
| `ratio_ma20` | 0.9898 | 1% *below* the 20-day average — short-term cooling |
| `ratio_ma50` | 1.0852 | 8.5% above the 50-day — quarterly trend still up |
| `ratio_ma200` | 1.7396 | **74% above the 200-day** — strong annual uptrend |

The ratio cascade (`<1, >1, >>1`) is the fingerprint of a strong long-term uptrend currently digesting recent gains — exactly what `ret_21d` and `pct_from_21d_high` independently confirm.

---

## 5. Volume statistics (3 features)

**What it captures.** Whether the price moves are happening on conviction (heavy volume) or apathy (thin volume), and whether participation is rising or falling.

```text
log_dollar_vol_z252  = z_score( log(close * volume), window=252 )
vol_ratio_20_252     = mean(volume, 20) / mean(volume, 252)
vol_trend_slope_60d  = OLS_slope( log(volume), t ) over last 60 days
```

**BNTX on 2021-07-02:**

| Feature | Value | Reading |
|---|---|---|
| `log_dollar_vol_z252` | -0.3526 | Today's dollar volume ~0.35 std below 1-year mean |
| `vol_ratio_20_252` | 0.7553 | Last month averaged 76% of last year's daily volume |
| `vol_trend_slope_60d` | -0.0155 | Log-volume drifting *down* over last 60 days |

Three independent slices all say the same thing: participation is fading even as the price holds up. That's a common, somewhat ominous signature near the end of a momentum run — the kind of subtle warning a single-window indicator would miss.

---

## 6. Shape descriptors — slope, swing, breadth, distribution (~15 features)

**What it captures.** Everything that is not a return, vol, or volume number: how *straight* the trend is, how *bouncy* it is around that trend, how the daily-return distribution is shaped, and how it co-moves with the market.

```text
# Trend straightness (regressions on log price vs. time)
slope_20, slope_60                 = OLS slope of log(close) over last 20 / 60 days
slope_accel                        = slope_20 - slope_60
trendline_residual_z_60            = z_score of today's residual from the 60d log fit
r2_log_60                          = R^2 of that 60d log fit

# Drawdown / swing counts
dd_count_5pct_120d                 = # of >= 5% peak-to-trough swings in last 120 days
swing_count_60d                    = # of local pivots (5-bar high/low) in last 60 days

# Recent extreme distances (60-day window)
days_since_60d_high                = bars since the last 60-day closing high
pct_from_60d_high                  = close[t] / max(close, 60) - 1
pct_from_60d_low                   = close[t] / min(close, 60) - 1

# Breadth and balance
above_ma200_frac_252d              = fraction of last 252 days that closed > 200d SMA
up_day_frac_63d                    = fraction of last 63 days with positive return
ret_252d_minus_21d                 = ret_252d - ret_21d   # "old momentum" minus "recent"

# Bollinger width
bb_width_pct_20                    = (2 * 20d std of close) / SMA(close, 20)

# Distribution shape (last 252 daily returns)
skew_252d, kurt_252d               = sample skew and excess kurtosis

# Vol-of-vol and return/vol correlation
vol_of_vol_ratio_30_120            = std(realized_vol_21d, 30) / std(realized_vol_21d, 120)
ret_vol_corr_20                    = corr( ret_1d, realized_vol_5d ) over last 20 days

# Market exposure
beta_spy_63d                       = cov(ret_stock, ret_spy, 63) / var(ret_spy, 63)
```

**BNTX on 2021-07-02 — selected values:**

| Feature | Value | Reading |
|---|---|---|
| `skew_252d` | -0.329 | Mildly left-skewed — bigger down days than up days |
| `kurt_252d` | +1.185 | Modestly fat tails, less extreme than a pure meme name |
| `beta_spy_63d` | 0.000 | Effectively uncorrelated with SPY — pure idiosyncratic story |
| `above_ma200_frac_252d` | 0.2103 | Only spent 21% of the year above its 200d SMA |
| `up_day_frac_63d` | 0.5556 | 56% up days over the quarter — slight positive bias |
| `ret_252d_minus_21d` | +1.1899 | Nearly all of the year's gain came *before* the last month |
| `vol_trend_slope_60d` | -0.0155 | Reiterates the volume fade signal |

The two most striking values: **`beta_spy_63d = 0.00`** says the stock's quarter has nothing to do with the broader market — its driver is company-specific (vaccine news, EUA timing, earnings). And **`above_ma200_frac_252d = 0.21`** reveals that the spectacular `ret_252d = +1.23` was earned almost entirely in one explosive leg — the stock spent 79% of the year *below* its 200-day SMA, then sprinted above it. That's a very different shape of "double" than a steady climber would produce, and the shape features are the only ones that surface it.

---

## Why these and not others

> **Design principles for this feature set:**
> - **Interpretable.** Every feature has a one-sentence economic meaning a human trader would recognize. No PCA components, no autoencoder embeddings.
> - **Price-and-volume only.** No fundamentals, no news, no options data — keeps the pipeline runnable on any daily OHLCV history and trivially reproducible.
> - **No forward leak.** Every formula uses only data with timestamp `<= t`. Rolling windows are right-aligned; nothing peeks at tomorrow.
> - **No NaN-prone divisions.** We avoided things like "price / earnings" or "volume / float" where the denominator can be missing, stale, or zero for a long stretch. Where we do divide (e.g. `ratio_ma20`), the denominator is a rolling mean that is always defined once the warm-up window is past.
> - **Multi-horizon by design.** Returns, vols, and shape descriptors all appear at multiple windows (1d–252d) so the model can pick the time scale that matters for a given regime rather than us pre-committing to one.
> - **Deliberately omitted.** RSI/MACD (redundant with momentum + MA ratios), candle patterns (noisy on daily bars), and any cross-sectional or sector-relative features (these stay per-symbol and self-contained so the same vector works in a screener or a single-name model).
