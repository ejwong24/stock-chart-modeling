#!/usr/bin/env python3
"""Infer probable delistings from already-downloaded yfinance OHLCV.

yfinance does not expose delisting status; instead it forward-pads zero-volume
bars to the most-recent trading day. We exploit five distress signatures
(see project notes) and emit a candidate seed list to augment
config/delisted_seed.txt without paying for Polygon/EODHD.
"""
from __future__ import annotations
import glob, os, sys
import pandas as pd

ADJ_DIR  = "data/adjusted"
OUT_CSV  = "data/universe/delisted_inferred.csv"
ACTIVE_U = "data/universe/working_universe.csv"
TERMINAL = pd.Timestamp("2026-04-20")  # corpus-wide max bar (live tickers reach this)

def classify(df: pd.DataFrame) -> dict | None:
    if len(df) < 20:
        return None
    last = df.iloc[-1]
    last20 = df.tail(20)
    v = df["volume"].to_numpy()
    # trailing zero-volume run length
    zrun = 0
    for x in v[::-1]:
        if x == 0: zrun += 1
        else: break
    last_date  = last["date"]
    last_close = float(last["close"])
    avg_vol20  = float(last20["volume"].mean())
    peak20     = float(last20["close"].max())
    dd20       = (peak20 - last_close) / peak20 if peak20 > 0 else 0.0

    reasons = []
    if last_date < TERMINAL:                       reasons.append("early_terminal")
    if zrun >= 10:                                 reasons.append(f"zerovol{zrun}")
    if last_close <= 0.10:                         reasons.append("subdime_close")
    elif last_close < 0.50 and avg_vol20 < 5_000:  reasons.append("penny_distress")
    if dd20 >= 0.90:                               reasons.append("bagel_drop")
    if last_close == 0.0:                          reasons.append("zero_close")
    if not reasons:
        return None
    return dict(last_date=last_date.date(), last_close=round(last_close,4),
                last_20d_avg_volume=round(avg_vol20,1),
                last_20d_max_drawdown=round(dd20,4),
                trailing_zero_volume_days=zrun, reasons="|".join(reasons))

def main() -> int:
    rows = []
    for f in sorted(glob.glob(f"{ADJ_DIR}/*.parquet")):
        ticker = os.path.basename(f)[:-8]
        # SPACs warrants/units have legitimately tiny float; skip to cut FPs
        if ticker.endswith(("W","U","R")):  continue
        try:
            df = pd.read_parquet(f, columns=["date","close","volume"])
        except Exception:
            continue
        info = classify(df)
        if info is not None:
            rows.append({"ticker": ticker, **info})
    out = pd.DataFrame(rows).sort_values(["last_date","last_close"])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    active = set(pd.read_csv(ACTIVE_U).query("active==True")["ticker"])
    incremental = sum(t not in active for t in out["ticker"])
    print(f"flagged={len(out)}  incremental_vs_active={incremental}  -> {OUT_CSV}")
    print(out["reasons"].value_counts().head(10).to_string())
    return 0

if __name__ == "__main__":
    sys.exit(main())
