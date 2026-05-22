# The data acquisition layer — yfinance, split-adjustment, and error handling

The pipeline lives or dies on the quality of its source data. For ~6,500 US-listed common stocks, the data acquisition layer in `src/stock_chart/data_acq.py` is the single biggest source of subtle, systemic bugs in any backtest. This document walks through how we pull, clean, and cache daily OHLCV bars — and the trade-offs we accepted along the way.

## The fetch loop

The top of `data_acq.py` is unromantic on purpose:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=8, backend="threading")(
    delayed(fetch_one)(t) for t in universe
)
```

`fetch_one(ticker)` is a thin wrapper around `yf.download(ticker, start, end, auto_adjust=True, progress=False)`. The threading backend (not loky) is deliberate — yfinance is I/O bound, and threads share the HTTP session pool without paying serialization cost.

The **`auto_adjust=True`** flag is load-bearing. With it, yfinance returns split- and dividend-adjusted close prices in the `Close` column and discards the unadjusted ones. That is exactly the series we want for return calculation: `r_t = close_t / close_{t-1} - 1` is meaningful only if the denominator and numerator are on the same adjusted scale. Without `auto_adjust`, a 2-for-1 split shows up as a -50% return on the split day, which would poison every momentum and volatility feature downstream.

## Retry strategy

yfinance's underlying HTTP layer returns `429 Too Many Requests` and `503 Service Unavailable` for somewhere between 1% and 3% of tickers on any given run. These are almost always transient.

```python
for attempt in range(3):
    try:
        df = yf.download(ticker, start, end, auto_adjust=True, progress=False)
        return df
    except Exception as e:
        if attempt == 2:
            log_error(ticker, e)
            return pd.DataFrame()
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
```

Three retries with exponential backoff (1s, 2s, 4s) catches the vast majority of transient failures. We give up after three to avoid runaway loops: a ticker that 429s four times in a row is usually a ticker that genuinely doesn't exist or has been delisted from Yahoo's index, and no amount of retrying will fix it.

## Normalization

`_normalize_one(df, ticker)` is where the raw frame gets beaten into a canonical shape:

```python
def _normalize_one(df, ticker):
    if df is None or df.empty or "Close" not in df.columns:
        # Bug #6 regression: return empty frame with correct schema
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])

    df = df.dropna(how="all")
    df.columns = [c.title() for c in df.columns]
    df = df.dropna(subset=["Close"])
    df.index = df.index.tz_localize(None)
    df = df.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high",
        "Low":"low","Close":"close","Volume":"volume",
    })
    return df[["date","open","high","low","close","volume"]]
```

Notes worth singling out:

- **All-NaN rows** appear on US market holidays when yfinance fills the index with empties. Dropping them keeps the trading calendar honest.
- **Title-casing columns** sounds trivial but isn't — different yfinance versions disagree on capitalization, and the rename map only works if we normalize first.
- **NaN close rows** are the fingerprint of halts and post-delisting padding.
- **tz-naive everywhere** is a discipline choice. Mixing tz-aware and tz-naive frames raises silently in some operations and loudly in others.
- **Bug #6 regression**: a previous version crashed when yfinance returned `None` or a frame missing `Close`. We now return an empty frame with the correct schema.

## The cache

Output goes to `data/adjusted/{TICKER}.parquet`, one row per trading day. Parquet because:

- **Fast load.** zstd-compressed binary; an SPY parquet is ~120KB on disk and loads in single-digit milliseconds.
- **Schema preserved.** `date` survives as `datetime64[ns]`, `close` as `float64`.
- **Columnar.** Queries like "load only the close column for all 6,500 stocks" touch ~1/6th of the bytes a row-store would.

## Split-adjustment after the fact

A stock that 2-for-1 splits on day D will, in our cached parquet, show a $50 close on day D-1 — even though the contemporaneous price was $100. yfinance retroactively folds the split factor into the entire history.

We accept this. The alternative is to pull raw closes and apply our own corporate-actions table, which is several hundred lines of additional code and a fresh source of bugs. The cost is a small look-ahead: our backtest implicitly "knows" the split happened. In practice, splits don't carry much alpha — the return on the split day is by construction zero — so the look-ahead is cosmetic for momentum strategies.

## The delisting hole

Tickers that delist between yfinance's API knowledge cutoff and "today" disappear from `yf.download` — the call returns an empty frame. `fetch_one` passes that through to `_normalize_one`, which returns the empty-schema frame from the Bug #6 fix. The universe builder downstream filters anything with zero rows.

This is the **survivorship problem in miniature**: the stocks most likely to be missing are the ones that failed.

## Per-ticker error logging

Every exception inside `fetch_one` writes a line to `data/acquire_errors.log`:

```
2026-05-22T03:14:22Z  XYZQ  HTTPError  404 Not Found
2026-05-22T03:14:23Z  ABCD  RateLimit  429 Too Many Requests
```

Post-hoc inspection of this file is the fastest way to diagnose a bad run. The two dominant error classes are `404` (ticker delisted before yfinance's earliest data) and `429` (rate limit surviving past three retries).

## Runtime

End-to-end, a full universe refresh takes **25–40 minutes** for ~6,500 tickers at `n_jobs=8` on the Oracle ARM64 4-core box. The bottleneck is yfinance's per-ticker round-trip latency (200–500ms). Bumping `n_jobs` past 8 triggers more 429s than it saves wall time.

---

> **Up next:** the safety net above stops bad rows from entering the cache, but it does not solve survivorship bias. See [/story/02_audit](/story/02_audit) for why that matters and what the flaws critique surfaced.
