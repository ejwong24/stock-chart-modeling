# Problem 1 — Survivorship-bias closure on a hobbyist budget

## Synthesis (top action)

**Build a 4-source free pipeline:**

1. **SEC EDGAR Form 25** (canonical delisting notice, 2005+, ~5k unique tickers)
2. **SEC Form 8-K Item 3.01** (issuer-side advance notice, +1.2k–2k incremental)
3. **Self-detected terminal patterns** in our existing OHLCV (-100% drops, zero-volume tails — caught 32 zombies in our current data already)
4. **Wayback iShares ITOT/IWV/IWM holdings** for point-in-time membership (~99.5% float coverage)

**Estimated post-build delisted-ticker count:** ~7,500–9,000.
**Survivorship-bias residual:** ~85–90% of historical attrition captured. Polygon $30/mo only justified if backtest CAGR moves ≥0.8% absolute when toggling delisted seed inclusion.

**Where free fundamentally fails:** pre-2005 delistings, OTC drop-offs (no formal event), foreign ADR terminations, share-class consolidations, point-in-time fundamentals (just tickers).

---

## A1 — SEC EDGAR Form 25 (HIGH severity gap closed)

EDGAR full-text search at `https://efts.sec.gov/LATEST/search-index?forms=25` returns JSON with accession + display name + file_date. 10 req/sec rate limit with proper User-Agent header. Form 25 + Form 25-NSE together cover essentially all NYSE/NASDAQ/AMEX delistings 2005-2026.

Modern filings (2018+) have machine-readable `primary_doc.xml` with `<issuerTradingSymbol>`, `<exchangeName>`, `<rule>`, `<effectiveDate>`. Pre-2018 needs regex on plain-text. Rule code 12d2-2(a)/(b)/(c) distinguishes voluntary vs exchange-initiated, but does NOT distinguish merger-for-cash from bankruptcy — cross-reference Form 8-K Item 1.01 for that.

**Yield:** ~4,500–6,000 unique delisted tickers vs current 28-entry seed = ~150–200x increase.

```python
import requests, time, json
from xml.etree import ElementTree as ET

UA = {"User-Agent": "OpenClaw Research ejwong@gmail.com"}
BASE = "https://efts.sec.gov/LATEST/search-index"

def search_form25(start, end, forms=("25", "25-NSE")):
    out = []
    for form in forms:
        frm = 0
        while True:
            r = requests.get(BASE, params={
                "forms": form, "dateRange": "custom",
                "startdt": start, "enddt": end, "from": frm
            }, headers=UA, timeout=30)
            hits = r.json()["hits"]["hits"]
            if not hits: break
            out.extend(hits)
            frm += 100
            time.sleep(0.12)
    return out

def parse_filing(accession, cik):
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/primary_doc.xml"
    r = requests.get(url, headers=UA, timeout=30); time.sleep(0.12)
    if r.status_code != 200: return None
    root = ET.fromstring(r.text)
    ns = {"x": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    g = lambda p: (root.find(p, ns).text if root.find(p, ns) is not None else None)
    return {"ticker": g(".//x:issuerTradingSymbol"),
            "issuer": g(".//x:issuerName"),
            "exchange": g(".//x:exchangeName"),
            "rule":   g(".//x:rule"),
            "effective_date": g(".//x:effectiveDate")}
```

## A2 — Wikipedia delisted lists (MED severity)

Wikipedia tilts large-cap; ~85% large, ~50% mid, ~15-20% microcap. Net incremental ~180-260 unique tickers, ~120-160 cleanly resolvable.

URLs to scrape: `List_of_largest_corporate_bankruptcies`, `List_of_corporate_collapses_and_scandals`, year-by-year bankruptcy categories (Category:Companies_that_filed_for_Chapter_11_bankruptcy_in_2020 etc), `List_of_NASDAQ_delistings`, `List_of_S%26P_500_companies` (history section), `List_of_Chinese_companies_delisted_from_U.S._exchanges`, `List_of_SPAC_mergers_and_acquisitions`. Resolve names to tickers via SEC EDGAR `company_tickers.json` with rapidfuzz.

## A3 — NASDAQ Trader Wayback archives (MED severity)

`nasdaqtraded.txt` is the best single file (superset of nasdaqlisted + otherlisted, has Financial Status field with codes D/E/Q/H/J for distress). Wayback CDX returns ~500 weekly snapshots 2016-2026. Build (ticker, first_listed, last_listed, exchange) table. ARCA-only ETFs covered; OTC/pink sheets not. Effort: 6-10 hr scrape + 1 day dev.

## A4 — Wayback iShares ETF holdings (HIGH severity)

iShares URLs `/us/products/<id>/<slug>/1467271812596.ajax?fileType=csv`. CDX queries return 40-120 captures/year per fund post-2018. IWV+IWM+ITOT covers ~99.5% of US float-cap. Bursty captures (clusters at quarter-ends and Russell recon dates). Storage: cache parsed ticker sets (~10 MB total).

```python
def universe(week_yyyymmdd):
    tix = set()
    for fund_path in ["239714/...", "239710/...", "239724/..."]:
        ts = max(captures(fund_path, week_yyyymmdd), default=None)
        if ts: tix |= holdings_at(fund_path, ts)
    return tix
```

## A5 — SPAC mergers (HIGH for this niche)

~700 SPAC mergers 2020-2024, ~25-35% bankrupt/delisted by 2026, another ~20% sub-$1. Backtest universe seeded from current survivors silently drops these failures. SEC EDGAR 8-K Item 2.01 ("completion of business combination") gives old (SPAC) and new (merged) ticker. Known distressed names already in our universe: BBAI, BFRG, BKSY, BLUE, BODY, EOSE, GOEV, HYZN, IRNT, LCID, MNTV, MVST, NKLA, OPAD, PSFE, RIDE, VLTA, WISH, WKHS — ~20 of an estimated 150-200 distressed.

## A6 — Self-detected terminal patterns in existing data (MED severity)

Detector found 32 probable delistings with clear off-exchange signatures in our existing 6,692 parquet files: 18 long zero-volume tails (SVA had 1,798 days of zero-vol forward-pad!), 6 bagel drops, 5 sub-dime distress. None of the 32 overlap with our 67-name static seed — pure incremental. Filtered out 323 noise candidates (SPAC warrants/units with W/U/R suffixes).

## A7 — Sharadar/Quandl/EODHD/Tiingo paid options

Tiingo free tier serves delisted EOD prices ~1962-present, capped at 500 unique symbols/month — workable with monthly universe rotation. Sharadar SF1/SEP gone behind paywall (~$150/mo retail). EODHD All-World $19.99/mo (cheapest paid upgrade, beats Polygon $30/mo, includes delisted + 60 exchanges + fundamentals). Norgate $360/yr Windows-only — Wine on ARM64 is non-starter.

**Recommendation:** Tiingo free for immediate use, EODHD All-World as paid upgrade if Tiingo cap fails.

## A8 — OpenFIGI symbology (LOW-MED)

Free symbol-mapping API (Bloomberg). 25 jobs/req with API key, 250 req/min ≈ 6,250 tickers/min. Does NOT expose clean delist_date. Best use: stable join key (CompositeFIGI) so ticker reassignments don't corrupt time series. Supplementary, not primary.

## A9 — SEC Form 8-K Item 3.01 (MED-HIGH)

Issuer's advance notice of delisting (filed before Form 25). Adds reason metadata (going concern, minimum-bid, going-private merger). ~5,000–6,500 filings 2016-2026, ~60-70% overlap with Form 25 issuers, **~1,200–2,000 incremental** beyond Form 25-only.

```python
REASONS = {"minimum_bid": r"minimum bid|\$1\.00",
           "going_concern": r"going concern|substantial doubt",
           "merger": r"going.private|merger agreement",
           "market_cap": r"minimum (market )?(value|capitalization)",
           "late_filing": r"late filing|delinquent"}
```

## A10 — Reverse-stock-split distress detection (LOW)

`yfinance.Ticker(t).actions` returns split ratios. Reverse split ≤ 0.10 (1:10+ RS) carries ~40-55% 2-year delist rate. Re-pull actions for 6,692 tickers: 35-70 min single-threaded, 8-15 min at 8 workers. Expect ~200-340 hits, of which ~100-180 likely already absent from our daily pulls.

## A11 — Russell 3000 reconstitution (MED)

FTSE Russell publishes annual June recon press releases with adds/deletions. ~200-300 R3000 deletions/yr × 9 years = ~2,200 total, of which ~700-900 are non-M&A distress/delisting. Quarterly IPO additions also exist. Parse with pdfplumber from `https://content.ftserussell.com/sites/default/files/russell-us-indexes-{prelim|final}-{add|delete}-list-{YYYY}.pdf`. ~1 day dev, 1 PDF/year maintenance.
