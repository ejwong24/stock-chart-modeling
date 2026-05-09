"""Universe construction.

Builds a working ticker universe from NASDAQ Trader symbol files plus a
static delisted-ticker seed list. Documented limitation: yfinance does
not provide a complete delisted-ticker history. For full survivorship-
bias correction, upgrade the data source to Polygon.io or CRSP.
"""
from __future__ import annotations
from pathlib import Path
from io import StringIO
import urllib.request
import pandas as pd


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "stock-chart/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("latin-1")


def fetch_active_universe() -> pd.DataFrame:
    nas = pd.read_csv(StringIO(_fetch(NASDAQ_LISTED_URL)), sep="|")
    nas = nas[nas["Test Issue"] == "N"]
    nas = nas[~nas["ETF"].astype(str).str.upper().eq("Y")]
    nas_syms = nas["Symbol"].dropna().astype(str)

    oth = pd.read_csv(StringIO(_fetch(OTHER_LISTED_URL)), sep="|")
    oth = oth[oth["Test Issue"] == "N"]
    oth = oth[~oth["ETF"].astype(str).str.upper().eq("Y")]
    oth_syms = oth["ACT Symbol"].dropna().astype(str)

    syms = pd.concat([nas_syms, oth_syms], ignore_index=True)
    syms = syms[~syms.str.contains(r"[\$\.\^]", regex=True)]
    syms = syms.drop_duplicates().sort_values().reset_index(drop=True)
    return pd.DataFrame({"ticker": syms, "active": True})


def load_delisted_seed(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "active"])
    syms = [s.strip().upper() for s in path.read_text().splitlines()
            if s.strip() and not s.strip().startswith("#")]
    return pd.DataFrame({"ticker": syms, "active": False})


def build(active_df: pd.DataFrame, delisted_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([active_df, delisted_df], ignore_index=True)
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    return df.sort_values("ticker").reset_index(drop=True)
