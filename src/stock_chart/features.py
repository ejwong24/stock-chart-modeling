"""Engineered feature stack — the FALSIFICATION TEST baseline.

The original document never directly answered: 'does the chart image add
anything that simple price/volume features don't already capture?' This
module computes ~30 deterministic features per (ticker, anchor) row from
the trailing 252-day window. A LightGBM trained on these features alone
is the proper baseline that the DINOv2 image stack must beat to justify
its complexity.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "vol_252d",
    "max_dd_252d", "current_dd_from_peak", "days_since_252d_high",
    "pct_from_252d_high", "pct_from_21d_high",
    "ratio_ma20", "ratio_ma50", "ratio_ma200",
    "log_dollar_vol_z252", "vol_ratio_20_252", "vol_trend_slope_60d",
    "skew_252d", "kurt_252d",
    "beta_spy_63d",
    "above_ma200_frac_252d", "up_day_frac_63d",
    "ret_252d_minus_21d", "vol_21d_div_252d",
    # Shape descriptors added per 60-subagent research P2A8
    "slope_20", "slope_60", "slope_accel",
    "trendline_residual_z_60", "r2_log_60",
    "dd_count_5pct_120d", "days_since_60d_high",
    "pct_from_60d_high", "pct_from_60d_low",
    "swing_count_60d", "bb_width_pct_20",
    "vol_of_vol_ratio_30_120", "ret_vol_corr_20",
]


def _safe(s: pd.Series, fill=0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _compute_window_features(closes: np.ndarray, vols: np.ndarray,
                             spy_closes: np.ndarray | None,
                             ti: int) -> dict:
    """Compute features for index ti using window [ti-251, ti]."""
    w = closes[ti - 251:ti + 1]
    v = vols[ti - 251:ti + 1]
    rets = np.diff(np.log(w))

    f = {}
    f["ret_1d"] = float(np.log(w[-1] / w[-2])) if w[-2] > 0 else 0.0
    f["ret_5d"] = float(np.log(w[-1] / w[-6])) if w[-6] > 0 else 0.0
    f["ret_21d"] = float(np.log(w[-1] / w[-22])) if w[-22] > 0 else 0.0
    f["ret_63d"] = float(np.log(w[-1] / w[-64])) if w[-64] > 0 else 0.0
    f["ret_126d"] = float(np.log(w[-1] / w[-127])) if w[-127] > 0 else 0.0
    f["ret_252d"] = float(np.log(w[-1] / w[0])) if w[0] > 0 else 0.0

    f["vol_21d"] = float(rets[-21:].std() * np.sqrt(252))
    f["vol_63d"] = float(rets[-63:].std() * np.sqrt(252))
    f["vol_252d"] = float(rets.std() * np.sqrt(252))

    cumret = w / w[0]
    peak = np.maximum.accumulate(cumret)
    dd = cumret / peak - 1.0
    f["max_dd_252d"] = float(dd.min())
    f["current_dd_from_peak"] = float(dd[-1])

    high_idx = int(np.argmax(w))
    f["days_since_252d_high"] = float(len(w) - 1 - high_idx)
    f["pct_from_252d_high"] = float(w[-1] / w.max() - 1.0)
    f["pct_from_21d_high"] = float(w[-1] / w[-21:].max() - 1.0)

    f["ratio_ma20"] = float(w[-1] / w[-20:].mean()) if w[-20:].mean() > 0 else 1.0
    f["ratio_ma50"] = float(w[-1] / w[-50:].mean()) if w[-50:].mean() > 0 else 1.0
    f["ratio_ma200"] = float(w[-1] / w[-200:].mean()) if w[-200:].mean() > 0 else 1.0

    dv = w * v
    log_dv = np.log1p(dv)
    f["log_dollar_vol_z252"] = float((log_dv[-1] - log_dv.mean()) / (log_dv.std() + 1e-9))
    f["vol_ratio_20_252"] = float(v[-20:].mean() / (v.mean() + 1e-9))
    if len(v) >= 60:
        x = np.arange(60, dtype=np.float64)
        ly = np.log1p(v[-60:].astype(np.float64))
        slope = float(np.polyfit(x, ly, 1)[0])
    else:
        slope = 0.0
    f["vol_trend_slope_60d"] = slope

    if rets.std() > 0:
        z = (rets - rets.mean()) / rets.std()
        f["skew_252d"] = float((z ** 3).mean())
        f["kurt_252d"] = float((z ** 4).mean() - 3.0)
    else:
        f["skew_252d"] = 0.0
        f["kurt_252d"] = 0.0

    # 63-day market beta. NOTE: spy_closes must be aligned to THIS ticker's
    # date index (done in compute_for_anchors), so spy_closes[ti] is SPY's
    # close on the same calendar date as closes[ti]. The slice needs 64 closes
    # to yield 63 log-returns matching my_rets[-63:] — a previous off-by-one
    # (ti-62 -> 63 closes -> 62 returns) made the length guard never pass, so
    # beta was silently 0.0 for every stock.
    if spy_closes is not None and ti >= 63 and len(spy_closes) >= ti + 1:
        spy_w = spy_closes[ti - 63:ti + 1]
        my_rets = rets[-63:]
        if np.all(np.isfinite(spy_w)) and (spy_w > 0).all():
            spy_rets = np.diff(np.log(spy_w))
            if len(spy_rets) == len(my_rets) and spy_rets.std() > 0:
                cm = np.cov(my_rets, spy_rets, ddof=1)
                f["beta_spy_63d"] = float(cm[0, 1] / (cm[1, 1] + 1e-12))
            else:
                f["beta_spy_63d"] = 0.0
        else:
            f["beta_spy_63d"] = 0.0
    else:
        f["beta_spy_63d"] = 0.0

    ma200 = pd.Series(w).rolling(200, min_periods=200).mean().to_numpy()
    above = (w > ma200).astype(np.float64)
    f["above_ma200_frac_252d"] = float(np.nanmean(above))
    up = (rets[-63:] > 0).astype(np.float64)
    f["up_day_frac_63d"] = float(up.mean())

    f["ret_252d_minus_21d"] = f["ret_252d"] - f["ret_21d"]
    f["vol_21d_div_252d"] = f["vol_21d"] / (f["vol_252d"] + 1e-9)

    # Shape descriptors (per 60-subagent research P2A8) — drop-in O(W)
    log_p = np.log(np.clip(w, 1e-9, None))

    # OLS slopes on log-price
    x20 = np.arange(20, dtype=np.float64)
    x60 = np.arange(60, dtype=np.float64)
    s20 = float(np.polyfit(x20, log_p[-20:], 1)[0])
    s60_coef = np.polyfit(x60, log_p[-60:], 1)
    s60 = float(s60_coef[0])
    f["slope_20"] = s20
    f["slope_60"] = s60
    f["slope_accel"] = s20 - s60

    # 60d log-linear fit residual + R²
    fit60 = s60_coef[0] * x60 + s60_coef[1]
    resid60 = log_p[-60:] - fit60
    res_std = float(resid60.std() + 1e-12)
    f["trendline_residual_z_60"] = float(resid60[-1] / res_std)
    ss_tot = float(((log_p[-60:] - log_p[-60:].mean()) ** 2).sum() + 1e-12)
    ss_res = float((resid60 ** 2).sum())
    f["r2_log_60"] = float(1.0 - ss_res / ss_tot)

    # Drawdown count >5% over last 120 days
    w120 = w[-120:]
    peaks = np.maximum.accumulate(w120)
    dd_series = w120 / peaks - 1.0
    in_dd = False
    n_dd = 0
    for dv in dd_series:
        if dv <= -0.05 and not in_dd:
            n_dd += 1
            in_dd = True
        elif dv >= -0.005 and in_dd:
            in_dd = False
    f["dd_count_5pct_120d"] = float(n_dd)

    # Distance from 60d high/low
    w60 = w[-60:]
    high60_idx = int(np.argmax(w60))
    f["days_since_60d_high"] = float(len(w60) - 1 - high60_idx)
    f["pct_from_60d_high"] = float(w[-1] / w60.max() - 1.0)
    f["pct_from_60d_low"] = float(w[-1] / w60.min() - 1.0)

    # Swing count via simple local-max detection (no scipy dep)
    w60d = w60
    n_sw = 0
    for i in range(2, len(w60d) - 2):
        if (w60d[i] > w60d[i-1] and w60d[i] > w60d[i-2] and
            w60d[i] > w60d[i+1] and w60d[i] > w60d[i+2]):
            n_sw += 1
        elif (w60d[i] < w60d[i-1] and w60d[i] < w60d[i-2] and
              w60d[i] < w60d[i+1] and w60d[i] < w60d[i+2]):
            n_sw += 1
    f["swing_count_60d"] = float(n_sw)

    # Bollinger band width as pct of price (20-day)
    ma20 = float(w[-20:].mean())
    std20 = float(w[-20:].std() + 1e-12)
    f["bb_width_pct_20"] = float(4.0 * std20 / max(ma20, 1e-9))

    # Volatility-of-volatility compression vs expansion
    vol_recent = float(rets[-30:].std() + 1e-12)
    vol_prior = float(rets[-120:-30].std() + 1e-12) if len(rets) >= 120 else vol_recent
    f["vol_of_vol_ratio_30_120"] = vol_recent / vol_prior

    # Return-volume correlation (volume-confirmation signal)
    if len(rets) >= 21 and len(v) >= 21:
        ret_recent = rets[-20:]
        vol_recent_arr = v[-20:]
        if ret_recent.std() > 0 and vol_recent_arr.std() > 0:
            f["ret_vol_corr_20"] = float(np.corrcoef(ret_recent, vol_recent_arr)[0, 1])
        else:
            f["ret_vol_corr_20"] = 0.0
    else:
        f["ret_vol_corr_20"] = 0.0

    return f


def compute_for_anchors(adjusted_dir: Path, anchor_df: pd.DataFrame,
                        spy_path: Path | None = None) -> pd.DataFrame:
    """Compute features per (ticker, anchor_date) row in `anchor_df`.

    `anchor_df` must have columns: ticker, anchor_date, anchor_idx.
    Returns the input DF with `FEATURE_COLS` appended.
    """
    spy_by_date = None
    if spy_path is not None and spy_path.exists():
        spy_df = pd.read_parquet(spy_path)
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        spy_df = (spy_df.sort_values("date")
                        .drop_duplicates("date", keep="last")
                        .reset_index(drop=True))
        spy_by_date = pd.Series(spy_df["close"].to_numpy(dtype=np.float64),
                                index=spy_df["date"].to_numpy())

    rows = []
    for ticker, sub in anchor_df.groupby("ticker", sort=False):
        p = adjusted_dir / f"{ticker}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
        closes = df["close"].to_numpy(dtype=np.float64)
        vols = df["volume"].to_numpy(dtype=np.float64)
        # Align SPY to THIS ticker's calendar by DATE (not by row position):
        # spy_closes[ti] must be SPY's close on closes[ti]'s date. Indexing a
        # globally-sorted SPY array by the ticker's positional anchor_idx would
        # silently compare mismatched dates for any ticker whose history starts
        # later than / diverges from SPY's.
        if spy_by_date is not None:
            spy_closes = (spy_by_date.reindex(df["date"].to_numpy())
                                     .ffill().bfill()
                                     .to_numpy(dtype=np.float64))
        else:
            spy_closes = None
        for _, r in sub.iterrows():
            ti = int(r["anchor_idx"])
            if ti < 251 or ti >= len(closes):
                continue
            f = _compute_window_features(closes, vols, spy_closes, ti)
            row = {"ticker": ticker, "anchor_date": r["anchor_date"], **f}
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ticker", "anchor_date"] + FEATURE_COLS)
    out = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        out[c] = _safe(out[c])
    return out
