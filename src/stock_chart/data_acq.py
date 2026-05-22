"""yfinance bulk OHLCV downloader.

Downloads split/dividend-adjusted daily bars (auto_adjust=True) for a
list of tickers. Writes one parquet per ticker to data/adjusted/. The
auto_adjust flag is the structural fix for the original document's
post-hoc 'price-jump filter' workaround for split discontinuities.

Limitations:
- yfinance is rate-limited (informal). We chunk + sleep.
- yfinance does not enumerate delisted tickers; we rely on the static
  seed list. Some delisted tickers will return empty data.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import yfinance as yf


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_one(df: pd.DataFrame) -> pd.DataFrame:
    # Defensive: bail out cleanly on empty or schema-less inputs (yfinance
    # occasionally returns these for delisted/invalid tickers).
    if df is None or len(df) == 0 or len(df.columns) == 0:
        return pd.DataFrame(columns=["date"] + [c.lower() for c in REQUIRED_COLS])
    out = df.dropna(how="all").copy()
    out.columns = [str(c).title() for c in out.columns]
    out = out[[c for c in REQUIRED_COLS if c in out.columns]]
    if "Close" not in out.columns or len(out) == 0:
        return pd.DataFrame(columns=["date"] + [c.lower() for c in REQUIRED_COLS])
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    out = out.reset_index()
    out.columns = [c.lower() for c in out.columns]
    return out


def fetch_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Bulk-download a small batch via yfinance; returns dict ticker -> df."""
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t not in df.columns.get_level_values(0):
                continue
            try:
                sub = df[t]
            except KeyError:
                continue
            sub = _normalize_one(sub)
            if len(sub) > 0:
                out[t] = sub
    else:
        sub = _normalize_one(df)
        if len(sub) > 0 and len(tickers) == 1:
            out[tickers[0]] = sub
    return out


def fetch_all(tickers: list[str], start: str, end: str, out_dir: Path,
              batch_size: int = 50, sleep_between: float = 0.6,
              skip_existing: bool = True) -> dict:
    """Fetch all tickers, write one parquet per ticker. Returns summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in tickers:
        if skip_existing and (out_dir / f"{t}.parquet").exists():
            continue
        todo.append(t)
    written, empty, errors = 0, 0, 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            data = fetch_batch(batch, start, end)
        except Exception:
            errors += len(batch)
            time.sleep(sleep_between * 2)
            continue
        for t in batch:
            if t in data and len(data[t]) > 0:
                data[t].to_parquet(out_dir / f"{t}.parquet", index=False)
                written += 1
            else:
                empty += 1
        time.sleep(sleep_between)
    existing = sum(1 for _ in out_dir.glob("*.parquet"))
    return {
        "requested": len(tickers),
        "fetched_this_run": written,
        "empty_or_missing": empty,
        "errors": errors,
        "total_on_disk": existing,
    }


def load_one(out_dir: Path, ticker: str) -> pd.DataFrame | None:
    p = out_dir / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
