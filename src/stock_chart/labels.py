"""Weekly-anchor label generation.

Anchor rule: per ticker, per ISO calendar week, take the LAST trading day
in that week with a valid close. Skip weeks with fewer than 3 trading days
for the ticker. Skip the first `warmup_days` of each ticker's history and
the last (max_horizon + max_embargo) days where forward labels would not
be observable.

For each anchor we emit:
- binary labels  ret_{H}d_ge_{T}pct = 1 if Close[t+H]/Close[t] - 1 >= T
- continuous     ret_{H}d            = log(Close[t+H]/Close[t])
- excursions     mfe_{H}d, mae_{H}d  (max favorable / adverse)
- prefilter flag prefilter_ma250_15

CRITICAL: every row carries `label_resolution_date` = trading day H steps
ahead of `anchor_date`. The walk-forward splitter uses this column to
purge training rows whose forward window straddles a test fold (López de
Prado purging + embargo). This is the structural fix for the original
document's leakage HIGH-severity flaw.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _weekly_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are the last trading day in their ISO week (>=3 days)."""
    iso = df["date"].dt.isocalendar()
    wk = iso[["year", "week"]].astype(int).agg(lambda r: r.iloc[0] * 100 + r.iloc[1], axis=1)
    df = df.assign(_wk=wk)
    counts = df.groupby("_wk")["date"].transform("count")
    is_last = df["date"] == df.groupby("_wk")["date"].transform("max")
    out = df[(counts >= 3) & is_last].drop(columns="_wk").reset_index(drop=True)
    return out


def _ma_extension(df: pd.DataFrame, window: int = 250) -> pd.Series:
    ma = df["close"].rolling(window, min_periods=window).mean()
    return df["close"] / ma


def label_one(df: pd.DataFrame, ticker: str, horizons: list[int],
              thresholds: list[float], warmup_days: int = 252,
              ma_window: int = 250, ma_extension: float = 1.5) -> pd.DataFrame:
    """Emit per-anchor labels for a single ticker."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < warmup_days + max(horizons) + 2:
        return pd.DataFrame()

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["ratio_ma"] = _ma_extension(df, ma_window)
    df["adv20_usd"] = (df["close"] * df["volume"]).rolling(20, min_periods=20).mean()

    n = len(df)
    H_max = max(horizons)
    eligible_idx = np.arange(warmup_days, n - H_max - 1)
    if len(eligible_idx) == 0:
        return pd.DataFrame()

    eligible = df.iloc[eligible_idx].copy()
    anchors = _weekly_anchors(eligible)
    if len(anchors) == 0:
        return pd.DataFrame()

    closes = df["close"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy()
    idx_map = {d: i for i, d in enumerate(dates)}
    anchor_idx = anchors["date"].map(idx_map).to_numpy(dtype=np.int64)

    out = pd.DataFrame({
        "ticker": ticker,
        "anchor_date": pd.to_datetime(anchors["date"].to_numpy()),
        "anchor_close": closes[anchor_idx],
        "anchor_idx": anchor_idx,
        "ratio_ma250": anchors["ratio_ma"].to_numpy(),
        "adv20_usd": anchors["adv20_usd"].to_numpy(),
    })

    anchor_close_arr = out["anchor_close"].to_numpy()
    for H in horizons:
        end_idx = anchor_idx + H
        end_close = closes[end_idx]
        # Guard against non-positive prices. A non-positive ANCHOR close makes
        # the return meaningless (inf), and since (inf >= T) is True it would
        # silently flip the binary label to a FALSE POSITIVE — corrupting the
        # training target on exactly the delisted / heavily-adjusted names the
        # survivorship-aware universe deliberately includes. A zero END close
        # is instead a legitimate -100% (ret = -1); only its log_ret (-inf)
        # needs cleaning.
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = end_close / anchor_close_arr - 1.0
            log_ret = np.log(end_close / anchor_close_arr)
        ret = np.where(anchor_close_arr > 0, ret, np.nan)
        log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

        # Path stats over [anchor+1 .. anchor+H]
        # MFE (max favorable excursion) is the peak UNREALIZED GAIN — bounded
        # at 0 by convention (if the stock only ever went down, MFE = 0).
        # MAE (max adverse excursion) is the peak UNREALIZED LOSS — bounded
        # at 0 by convention (if the stock only ever went up, MAE = 0).
        mfe = np.empty(len(out))
        mae = np.empty(len(out))
        for i, (a, e) in enumerate(zip(anchor_idx, end_idx)):
            window = closes[a + 1:e + 1]
            if len(window) == 0:
                mfe[i] = np.nan
                mae[i] = np.nan
            else:
                if closes[a] <= 0:
                    mfe[i] = 0.0
                    mae[i] = 0.0
                else:
                    rel = window / closes[a] - 1.0
                    mfe[i] = float(max(0.0, rel.max()))
                    mae[i] = float(min(0.0, rel.min()))

        out[f"fwd_ret_{H}d"] = ret
        out[f"fwd_log_ret_{H}d"] = log_ret
        out[f"fwd_mfe_{H}d"] = mfe
        out[f"fwd_mae_{H}d"] = mae
        out[f"resolution_date_{H}d"] = pd.to_datetime(dates[end_idx])

        for T in thresholds:
            t_str = f"{int(round(T * 100))}pct"
            out[f"ret_{H}d_ge_{t_str}"] = (ret >= T).astype(np.int8)

    out["prefilter_ma250_15"] = (out["ratio_ma250"] > ma_extension).astype(np.int8)

    # Path-dependent labels (per 60-subagent research P5A7) — leverage existing
    # mfe/mae columns. Zero new compute beyond column algebra. Useful for
    # path-aware models without doing the full first-touch barrier walk.
    EPS = 1e-4
    for H in horizons:
        mfe = out[f"fwd_mfe_{H}d"]
        mae = out[f"fwd_mae_{H}d"]
        ret = out[f"fwd_ret_{H}d"]
        abs_mae = mae.abs().clip(lower=EPS)

        # Approximate triple-barrier (no first-touch ordering); a few flagship
        # (T, S) pairs. Biased toward 1 vs true first-touch (caveat documented).
        for T, S in [(0.05, 0.02), (0.05, 0.05), (0.10, 0.05), (0.15, 0.10)]:
            t_pct = int(round(T * 100))
            s_pct = int(round(S * 100))
            out[f"tb_approx_{H}d_T{t_pct}_S{s_pct}"] = (
                (mfe >= T) & (mae > -S)
            ).astype(np.int8)

        # Continuous path-aware labels
        out[f"mfe_mae_ratio_{H}d"] = (mfe / abs_mae).astype(np.float32)
        out[f"sortino_label_{H}d"] = (ret.clip(lower=0) / abs_mae).astype(np.float32)
        out[f"upside_dominance_{H}d"] = (mfe > abs_mae).astype(np.int8)
        out[f"clean_run_{H}d"] = ((mfe >= 0.05) & (mae > -0.02)).astype(np.int8)
    # Drop anchors with a non-positive anchor close: their returns/labels are
    # meaningless (see the np.errstate guard above). A zero END close is kept
    # (it is a real -100% total loss, a valuable survivorship signal).
    out = out[out["anchor_close"] > 0].reset_index(drop=True)
    return out


def build_all(adjusted_dir: Path, tickers: list[str], horizons: list[int],
              thresholds: list[float], warmup_days: int = 252,
              ma_window: int = 250, ma_extension: float = 1.5,
              prefilter_only: bool = True) -> pd.DataFrame:
    rows = []
    for t in tickers:
        p = adjusted_dir / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        sub = label_one(df, t, horizons, thresholds, warmup_days, ma_window, ma_extension)
        if len(sub) == 0:
            continue
        if prefilter_only:
            sub = sub[sub["prefilter_ma250_15"] == 1]
        if len(sub) > 0:
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["anchor_date", "ticker"]).reset_index(drop=True)
    return out
