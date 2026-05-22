"""SEC EDGAR Form 25 / Form 25-NSE delisting scraper.

Per RESEARCH P1A1 — incremental ~5,000 unique delisted tickers vs current
28-entry static seed (~150x increase). Combined with Form 8-K Item 3.01
(scrape_edgar_8k_3_01.py) and self-detected terminal patterns
(detect_inferred_delistings.py), captures ~85-90% of historical attrition.

Endpoint: https://efts.sec.gov/LATEST/search-index?forms=25&dateRange=custom
Rate-limited at 10 req/sec; we cap at 8 to be safe.

Usage:
    python scripts/scrape_edgar_form25.py --start 2016-01-01 --end 2026-04-30 \\
        --out data/universe/edgar_form25.csv

Expect ~30-45 minutes for the full 2016-2026 scrape.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import requests
import pandas as pd

UA = {"User-Agent": "OpenClaw Research ejwong@gmail.com"}
SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index"
RATE_LIMIT_SLEEP = 0.13  # ~7.7 req/s, safe under SEC's 10/s cap


def search(forms: list[str], start: str, end: str) -> list[dict]:
    """Iterate the search-index endpoint with pagination across forms."""
    out = []
    for form in forms:
        frm = 0
        while True:
            r = requests.get(SEARCH_BASE, params={
                "forms": form, "dateRange": "custom",
                "startdt": start, "enddt": end, "from": frm,
            }, headers=UA, timeout=30)
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            out.extend(hits)
            print(f"  {form}: {len(hits)} hits at offset {frm}", flush=True)
            frm += 100
            time.sleep(RATE_LIMIT_SLEEP)
            if frm >= 9900:
                # SEC cap is 10000; chunk by date if you hit it
                print(f"  WARNING: pagination cap hit for {form} {start}..{end}; "
                      "split date range to recover full results", flush=True)
                break
    return out


# SPAC display names have tokens like 'CNDA-UN', 'CNDA-WT' (7+ chars), so we
# allow up to 9 chars total per ticker. CIK strings are rejected post-match.
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9}(?:,\s*[A-Z][A-Z0-9.\-]{0,9})*)\)")


def extract_tickers_from_display(display_name: str) -> list[str]:
    """Pull ticker(s) from the EDGAR display_name field.

    EDGAR formats names like 'Concord Acquisition Corp II  (CNDA, CNDA-UN, CNDA-WT)
    (CIK 0001851959)'. We grab the parenthesised group that does NOT start with 'CIK '.
    """
    if not display_name:
        return []
    out = []
    for m in _TICKER_RE.finditer(display_name):
        token = m.group(1)
        if token.startswith("CIK"):
            continue
        for sym in token.split(","):
            sym = sym.strip().upper()
            if sym and not sym.startswith("CIK"):
                out.append(sym)
    return out


def fetch_ticker_master() -> dict[str, str]:
    """SEC's CIK → ticker JSON; used as fallback when display_name has no ticker."""
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=UA, timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json()
        return {str(int(v["cik_str"])).zfill(10): v["ticker"] for v in data.values()}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--out", default="data/universe/edgar_form25.csv")
    ap.add_argument("--limit", type=int, default=0,
                     help="Stop after N filings (debug). 0 = unlimited.")
    args = ap.parse_args()

    print(f"searching EDGAR Form 25 + 25-NSE from {args.start} to {args.end}...")
    hits = search(["25", "25-NSE"], args.start, args.end)
    print(f"  total filings: {len(hits)}")

    print("loading SEC ticker master (CIK -> ticker fallback) ...")
    cik_to_ticker = fetch_ticker_master()
    print(f"  master entries: {len(cik_to_ticker)}")

    rows = []
    for i, h in enumerate(hits):
        if args.limit and i >= args.limit:
            break
        src = h.get("_source", {})
        cik = src.get("ciks", [""])[0]
        display = (src.get("display_names") or [""])[0]
        accession = h["_id"].split(":")[0]
        filing_date = src.get("file_date")

        tickers = extract_tickers_from_display(display)
        if not tickers and cik:
            t = cik_to_ticker.get(str(cik).zfill(10))
            if t: tickers = [t]
        if not tickers:
            continue
        for t in tickers:
            rows.append({
                "ticker": t,
                "issuer": display,
                "cik": cik,
                "filing_date": filing_date,
                "form": src.get("form"),
                "accession": accession,
            })
        if i and i % 200 == 0:
            print(f"  parsed {i}/{len(hits)} filings; {len(rows)} ticker rows so far")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} ticker rows -> {out_path}")
    if len(df) > 0:
        print(f"  unique tickers: {df['ticker'].nunique()}")
        print(f"  date range: {df['filing_date'].min()} to {df['filing_date'].max()}")
        print(df['form'].value_counts().to_string())


if __name__ == "__main__":
    sys.exit(main() or 0)
