"""Settle forward paper-trade picks against realized outcomes.

Companion to scripts/forward_pick.py. forward_pick writes one CSV per Friday to
data/forward_picks/<date>.csv with the week's top-K picks (ticker, anchor_date,
anchor_close, ...). This script closes the loop: for every pick whose forward
window (default 40 trading days) has now resolved, it looks up the realized
return and appends a row to data/forward_realized/realized.csv.

Run daily (system cron). Idempotent: a pick already in realized.csv is skipped,
so re-running only settles newly-resolvable picks. After ~26 weeks the realized
file holds ~130 fully out-of-sample trades evaluated under real execution — the
one result that sidesteps every in-sample/backtest objection.

Usage:
    python scripts/settle_picks.py [--horizon 40]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import config as cfg


REALIZED_COLS = ["pick_generated", "ticker", "horizon_days", "anchor_date",
                 "entry_close", "exit_date", "exit_close", "realized_ret",
                 "score", "track"]


def realized_return(df: pd.DataFrame, anchor_date, horizon: int):
    """Realized return of a pick entered at `anchor_date`, exited `horizon`
    trading days later, using the ticker's adjusted closes.

    Returns (realized_ret, exit_date, entry_close, exit_close) or None if the
    forward window has not fully resolved yet (not enough bars after the anchor)
    or the anchor date isn't in the series.
    """
    if df is None or len(df) == 0:
        return None
    dates = pd.to_datetime(df["date"]).to_numpy()
    anchor = np.datetime64(pd.Timestamp(anchor_date))
    pos = np.flatnonzero(dates == anchor)
    if len(pos) == 0:
        return None
    entry_idx = int(pos[0])
    exit_idx = entry_idx + horizon
    if exit_idx >= len(df):
        return None  # not yet resolved
    closes = df["close"].to_numpy(dtype=np.float64)
    entry_close = float(closes[entry_idx])
    exit_close = float(closes[exit_idx])
    if entry_close <= 0:
        return None  # meaningless entry (see labels.py label-corruption guard)
    realized = exit_close / entry_close - 1.0
    return realized, pd.Timestamp(dates[exit_idx]), entry_close, exit_close


def _load_settled(realized_path: Path) -> set:
    if not realized_path.exists():
        return set()
    df = pd.read_csv(realized_path)
    if df.empty or not {"pick_generated", "ticker", "horizon_days"}.issubset(df.columns):
        return set()
    return {(str(r.pick_generated), str(r.ticker), int(r.horizon_days))
            for r in df.itertuples(index=False)}


def _atomic_append(realized_path: Path, new_rows: pd.DataFrame) -> None:
    """Append rows and rewrite the file atomically (temp + os.replace)."""
    if realized_path.exists():
        existing = pd.read_csv(realized_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    tmp = realized_path.with_name(realized_path.name + ".tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, realized_path)


def settle_all(picks_dir: Path, realized_path: Path, adjusted_dir: Path,
               horizon: int) -> dict:
    """Settle every resolvable, not-yet-settled pick across all pick files.
    Returns a summary dict. Pure function of disk state — safe to re-run."""
    settled = _load_settled(realized_path)
    price_cache: dict[str, pd.DataFrame] = {}
    new_rows = []
    n_picks = n_settled = n_pending = 0

    for pick_file in sorted(picks_dir.glob("*.csv")) if picks_dir.exists() else []:
        picks = pd.read_csv(pick_file)
        for r in picks.itertuples(index=False):
            n_picks += 1
            h = int(getattr(r, "horizon_days", horizon))
            key = (str(getattr(r, "pick_generated", pick_file.stem)),
                   str(r.ticker), h)
            if key in settled:
                continue
            t = str(r.ticker)
            if t not in price_cache:
                p = adjusted_dir / f"{t}.parquet"
                price_cache[t] = (pd.read_parquet(p).sort_values("date").reset_index(drop=True)
                                  if p.exists() else None)
            res = realized_return(price_cache[t], r.anchor_date, h)
            if res is None:
                n_pending += 1
                continue
            realized, exit_date, entry_close, exit_close = res
            new_rows.append({
                "pick_generated": getattr(r, "pick_generated", pick_file.stem),
                "ticker": t, "horizon_days": h, "anchor_date": r.anchor_date,
                "entry_close": entry_close, "exit_date": exit_date.date().isoformat(),
                "exit_close": exit_close, "realized_ret": realized,
                "score": getattr(r, "score", float("nan")),
                "track": getattr(r, "track", ""),
            })
            settled.add(key)
            n_settled += 1

    if new_rows:
        realized_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_append(realized_path, pd.DataFrame(new_rows, columns=REALIZED_COLS))
    return {"picks_seen": n_picks, "newly_settled": n_settled,
            "still_pending": n_pending, "realized_file": str(realized_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--picks-dir", default="data/forward_picks")
    ap.add_argument("--realized", default="data/forward_realized/realized.csv")
    args = ap.parse_args()
    picks_dir = cfg.project_path(args.picks_dir)
    realized_path = cfg.project_path(args.realized)
    adjusted_dir = cfg.project_path("data", "adjusted")
    summary = settle_all(picks_dir, realized_path, adjusted_dir, args.horizon)
    print(f"settled picks: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
