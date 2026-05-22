# Building the universe — NASDAQ Trader files, delisted seed list, and how survivorship dies

The first thing the pipeline does on every run is rebuild the tradable universe from scratch. Not load a cached CSV, not query a vendor API — pull two flat files from `nasdaqtrader.com`, parse them, union with a hand-curated delisted seed, and hand the result downstream. The whole thing lives in `src/stock_chart/universe.py` and runs in under five seconds. The interesting part is what it's *trying* to fix: a model trained on whatever trades today has already been told the answer to half its job.

## The NASDAQ Trader files

NASDAQ publishes a symbol directory at `ftp://nasdaqtrader.com/SymbolDirectory/` that gets refreshed daily, late evening Eastern. Two files matter:

- **`nasdaqlisted.txt`** — every common stock and ETF listed on NASDAQ proper
- **`otherlisted.txt`** — everything on NYSE, NYSE American (formerly AMEX), and a handful of regional venues

Both are pipe-delimited with a trailing footer row containing the file creation timestamp. The schema for `nasdaqlisted.txt`:

```
Symbol | Security Name | Market Category | Test Issue | Financial Status | Round Lot Size | ETF | NextShares
```

We filter aggressively:

- `Test Issue = N` — drops symbols NASDAQ uses internally for system testing
- `ETF = N` — drops every ETF and ETN; this pipeline is single-name equities only
- `NextShares = N` — drops the now-defunct NextShares exchange-traded managed funds

After filtering and dropping rows with NaN in `Symbol`, we land on roughly 6,500 active common stocks. That number drifts — IPOs add to it, delistings subtract — but the order of magnitude is stable.

## Cleanup

The text files are reasonably well-formed but not perfectly so:

1. **Uppercase every ticker** — defensively force it so downstream joins don't fail on a stray `brk.b` vs `BRK.B`
2. **Strip whitespace** — both leading/trailing on tickers and names
3. **Deduplicate** — a small number appear in both files due to cross-listings. We `drop_duplicates(subset=['ticker'], keep='first')` with `nasdaqlisted.txt` parsed first, so NASDAQ wins on conflicts. Arbitrary but deterministic.

## The delisted seed

The active universe captures what trades today. It says nothing about what *used to* trade. For a model whose training window runs 2017-2024, that omission is the single largest source of leaked alpha.

We maintain `config/delisted_seed.txt` — about 340 hand-curated tickers of historically interesting delistings: post-merger LMT-style successors, SPAC unwinds, Form 25 voluntary delistings, S-1 withdrawals, biotech zeros, retail bankruptcies:

```
# Retail bankruptcies 2018-2020
TOYS    # Toys R Us
SHLD    # Sears Holdings
JCP     # J.C. Penney
```

One ticker per line. `#` starts a comment. Blank lines ignored. Tickers harvested from SEC EDGAR Form 25 filings (`scripts/scrape_edgar_form25.py`) get appended programmatically.

## Building the union

```python
universe = pd.concat([active_df, delisted_df]).drop_duplicates(
    subset=['ticker'], keep='first'
)
```

Active comes first, so if a ticker appears in both (delisted and later relisted), the active row wins.

## Why not point-in-time?

A "proper" survivorship-bias-free universe would be **point-in-time**: on every historical day, the set of tickers eligible for selection is exactly what was tradable on that day.

True point-in-time requires a database of `(date, ticker, listed?)` tuples for every trading day. We don't maintain that. Instead we approximate:

1. Use the active+delisted union as the candidate pool
2. For each anchor date, filter to tickers whose first data point in the price store is **before** that anchor (catches IPOs that hadn't happened yet)
3. Allow delisted tickers to remain candidates up to the last date they have data

This is not a perfect point-in-time reconstruction. It misses tickers that *did* trade in 2017 but aren't in either our active list or our seed — a known under-coverage.

## Known limitations

- **Seed coverage is partial.** Our ~340 hand-curated tickers cover roughly 28% of the estimated ~1,200 stocks delisted between 2017 and 2024.
- **No corporate-action graph.** A stock that merged into another shows up as either "ends abruptly at last close" or "blends into the successor's price series."
- **Reverse-split-to-zero.** Some failing companies execute a 1:10 reverse split to maintain listing standards, then go to zero. The split bumps the price 10x just before wipeout; we catch most of these but not all.

---

> **Going deeper:** the full survivorship-bias analysis is at [/research/01_survivorship_bias](/research/01_survivorship_bias).
