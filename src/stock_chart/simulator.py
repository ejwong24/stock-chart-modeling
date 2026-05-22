"""Event-driven portfolio simulator with realistic costs and liquidity.

CORRECTED vs original:
- Adds slippage (bps each side, default 10 bps).
- Adds commission per share with min/cap.
- Skips candidates with ADV20 < min_adv_usd.
- Caps position size = min(equity * max_pct, max_adv_pct * adv20_usd).
- Tracks daily equity (mark-to-market via close prices for open positions).

Inputs:
- `scores`: DataFrame with columns
    [ticker, anchor_date, score, anchor_idx, anchor_close, adv20_usd, resolution_date]
  one row per (ticker, anchor) tested in the test fold.
- `price_lookup`: dict ticker -> DataFrame[date, close] (already loaded).

Output:
- trade blotter, daily equity series, summary metrics dict.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    start_equity: float = 100_000.0
    max_position_pct: float = 0.04
    max_new_per_week: int = 5
    max_adv_pct: float = 0.005
    min_adv_usd: float = 5_000_000.0
    slippage_bps_each_side: float = 10.0
    commission_per_share: float = 0.005
    no_duplicate_ticker: bool = True
    # Path-dependent exit (per research P5A1) — opt-in
    trailing_stop_pct: float = 0.0  # 0.0 = disabled; 0.18 = exit at 18% drop from peak
    # Realistic-cost mode (per research P4A1) — opt-in
    use_almgren_chriss_impact: bool = False
    impact_eta: float = 0.142
    impact_beta: float = 0.5
    permanent_frac: float = 0.5
    daily_vol_default: float = 0.03  # used if not provided per-trade
    # Halt-risk simulation (per research P4A6) — opt-in
    halt_risk_enabled: bool = False
    halt_risk_seed: int = 42


@dataclass
class OpenPos:
    ticker: str
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    shares: int
    cost: float
    exit_date: pd.Timestamp
    exit_idx: int
    peak_close: float = 0.0  # for trailing-stop tracking
    adv20_usd: float = 0.0  # for Almgren-Chriss exit cost


def _slippage_mult(bps: float, side: str) -> float:
    s = bps / 1e4
    return (1.0 + s) if side == "buy" else (1.0 - s)


def impact_bps(position_usd: float, adv_usd: float, daily_vol: float = 0.03,
               eta: float = 0.142, beta: float = 0.5, permanent_frac: float = 0.5,
               min_bps: float = 1.0, max_bps: float = 500.0) -> float:
    """One-way Almgren-Chriss impact in bps (per research P4A1).

    Default coefficients from Almgren et al. (2005) US equity fits.
    Round-trip ≈ 2 × this value. Microcap reality is 30-80 bps each side
    versus our previous flat 10 bps assumption.
    """
    if adv_usd <= 0 or position_usd <= 0:
        return float(min_bps)
    participation = position_usd / adv_usd
    temporary = daily_vol * eta * (participation ** beta)  # decimal
    one_way = temporary * (1.0 + permanent_frac / 2.0)
    return float(max(min_bps, min(max_bps, one_way * 1e4)))


def _market_cap_tier(adv_usd: float) -> str:
    """Crude tier proxy from ADV (per research P4A6)."""
    if adv_usd < 5e6: return "micro"
    if adv_usd < 50e6: return "small"
    if adv_usd < 500e6: return "mid"
    return "large"


_HALT_RATES = {"large": 0.001, "mid": 0.003, "small": 0.008, "micro": 0.018}


def _commission(shares: int, per_share: float, trade_value: float) -> float:
    fee = shares * per_share
    return min(fee, 0.005 * trade_value)


def simulate(scores: pd.DataFrame, price_lookup: dict[str, pd.DataFrame],
             cfg: SimConfig) -> dict:
    """Run a single-strategy simulation. Returns blotter, equity, summary."""
    df = scores.copy()
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])
    df = df.sort_values(["anchor_date", "score"], ascending=[True, False]).reset_index(drop=True)

    # Build a master daily date index across all tickers' price_lookup
    all_dates = set()
    for t, sub in price_lookup.items():
        all_dates.update(sub["date"].tolist())
    daily_dates = pd.DatetimeIndex(sorted(all_dates))

    # Per-ticker date->index map for fast price lookup
    ticker_idx = {}
    for t, sub in price_lookup.items():
        m = pd.Series(sub["close"].to_numpy(dtype=np.float64),
                      index=pd.to_datetime(sub["date"]))
        ticker_idx[t] = m

    cash = cfg.start_equity
    open_pos: list[OpenPos] = []
    blotter = []

    anchor_groups = df.groupby("anchor_date", sort=True)

    open_tickers_set = set()
    halt_rng = np.random.default_rng(cfg.halt_risk_seed)

    def _exit_slippage_bps(pos: OpenPos) -> float:
        if cfg.use_almgren_chriss_impact and pos.adv20_usd > 0:
            return impact_bps(pos.cost, pos.adv20_usd, cfg.daily_vol_default,
                               cfg.impact_eta, cfg.impact_beta, cfg.permanent_frac)
        return cfg.slippage_bps_each_side

    def _check_trailing_stop(today: pd.Timestamp):
        """Per-position trailing-stop check; queue early exit at next available."""
        if cfg.trailing_stop_pct <= 0:
            return
        for p in list(open_pos):
            px_series = ticker_idx.get(p.ticker)
            if px_series is None or today not in px_series.index:
                continue
            px = float(px_series.loc[today])
            if px > p.peak_close:
                p.peak_close = px
            dd = px / max(p.peak_close, 1e-9) - 1.0
            if dd <= -cfg.trailing_stop_pct and today < p.exit_date:
                # Exit now; arm exit at today's close.
                p.exit_date = today

    def _close_due(today: pd.Timestamp):
        nonlocal cash
        still_open = []
        for p in open_pos:
            if p.exit_date <= today:
                px_series = ticker_idx.get(p.ticker)
                if px_series is None or p.exit_date not in px_series.index:
                    # find next available trading day after exit_date
                    after = px_series.index[px_series.index >= p.exit_date] if px_series is not None else []
                    if len(after) == 0:
                        # cannot close, abandon at last known price
                        still_open.append(p)
                        continue
                    actual_exit = after[0]
                    exit_close = float(px_series.loc[actual_exit])
                else:
                    actual_exit = p.exit_date
                    exit_close = float(px_series.loc[actual_exit])
                slip_bps_exit = _exit_slippage_bps(p)
                fill = exit_close * _slippage_mult(slip_bps_exit, "sell")
                proceeds = p.shares * fill
                comm = _commission(p.shares, cfg.commission_per_share, proceeds)
                cash += proceeds - comm
                ret_pct = (proceeds - comm - p.cost) / p.cost
                blotter.append({
                    "ticker": p.ticker,
                    "entry_date": p.entry_date, "exit_date": actual_exit,
                    "entry_price": p.entry_price, "exit_price": fill,
                    "shares": p.shares, "cost": p.cost, "proceeds": proceeds,
                    "commission_total": comm,
                    "trade_return_pct": ret_pct,
                    "exit_slippage_bps": slip_bps_exit,
                })
                open_tickers_set.discard(p.ticker)
            else:
                still_open.append(p)
        open_pos[:] = still_open

    daily_equity = []
    for today in daily_dates:
        _check_trailing_stop(today)
        _close_due(today)
        # equity mark
        mtm = cash
        for p in open_pos:
            px_series = ticker_idx.get(p.ticker)
            if px_series is not None and today in px_series.index:
                mtm += p.shares * float(px_series.loc[today])
            else:
                mtm += p.shares * p.entry_price  # fallback
        daily_equity.append({"date": today, "equity": mtm,
                             "cash": cash, "n_positions": len(open_pos)})

        if today not in anchor_groups.groups:
            continue
        candidates = anchor_groups.get_group(today)
        candidates = candidates.sort_values("score", ascending=False)
        new_count = 0
        for _, row in candidates.iterrows():
            if new_count >= cfg.max_new_per_week:
                break
            t = row["ticker"]
            if cfg.no_duplicate_ticker and t in open_tickers_set:
                continue
            # Defensive: skip candidates with NaN/missing critical fields.
            # NaN comparisons in Python return False, so without this any
            # NaN adv would silently pass the ADV filter and crash later.
            raw_adv = row.get("adv20_usd", 0.0)
            adv = float(raw_adv) if raw_adv is not None else 0.0
            if not np.isfinite(adv) or adv < cfg.min_adv_usd:
                continue
            raw_close = row.get("anchor_close")
            if raw_close is None or not np.isfinite(float(raw_close)) or float(raw_close) <= 0:
                continue
            entry_close = float(raw_close)
            equity_now = mtm
            target_usd = min(equity_now * cfg.max_position_pct,
                             cfg.max_adv_pct * adv)
            if target_usd <= 0 or target_usd > cash:
                target_usd = min(target_usd, cash)
            if target_usd < 100:
                continue
            # Halt risk: simulate ~halt-rate-tier probability of fill failure
            if cfg.halt_risk_enabled:
                tier = _market_cap_tier(adv)
                if halt_rng.random() < _HALT_RATES.get(tier, 0.005):
                    continue  # missed entry; no P&L
            slip_bps_entry = (impact_bps(target_usd, adv, cfg.daily_vol_default,
                                          cfg.impact_eta, cfg.impact_beta,
                                          cfg.permanent_frac)
                              if cfg.use_almgren_chriss_impact
                              else cfg.slippage_bps_each_side)
            fill = entry_close * _slippage_mult(slip_bps_entry, "buy")
            if not np.isfinite(fill) or fill <= 0:
                continue  # defensive: nonsensical fill price
            shares = int(target_usd // fill)
            if shares <= 0:
                continue
            cost = shares * fill
            comm = _commission(shares, cfg.commission_per_share, cost)
            total_cost = cost + comm
            if total_cost > cash:
                shares = int((cash - 0.005 * cash) // fill)
                if shares <= 0:
                    continue
                cost = shares * fill
                comm = _commission(shares, cfg.commission_per_share, cost)
                total_cost = cost + comm
            cash -= total_cost
            exit_date = pd.Timestamp(row["resolution_date"])
            open_pos.append(OpenPos(
                ticker=t, entry_date=today, entry_idx=int(row.get("anchor_idx", -1)),
                entry_price=fill, shares=shares, cost=total_cost,
                exit_date=exit_date, exit_idx=-1,
                peak_close=fill, adv20_usd=adv,
            ))
            open_tickers_set.add(t)
            new_count += 1

    # Force close any remaining at last available date
    last = daily_dates[-1]
    forced_close = []
    for p in open_pos:
        px_series = ticker_idx.get(p.ticker)
        if px_series is None:
            continue
        if last in px_series.index:
            close_px = float(px_series.loc[last])
        else:
            close_px = p.entry_price
        fill = close_px * _slippage_mult(cfg.slippage_bps_each_side, "sell")
        proceeds = p.shares * fill
        comm = _commission(p.shares, cfg.commission_per_share, proceeds)
        cash += proceeds - comm
        ret_pct = (proceeds - comm - p.cost) / p.cost
        forced_close.append({
            "ticker": p.ticker, "entry_date": p.entry_date, "exit_date": last,
            "entry_price": p.entry_price, "exit_price": fill,
            "shares": p.shares, "cost": p.cost, "proceeds": proceeds,
            "commission_total": comm, "trade_return_pct": ret_pct,
            "forced_close": True,
        })
    blotter.extend(forced_close)
    open_pos.clear()

    bdf = pd.DataFrame(blotter)
    edf = pd.DataFrame(daily_equity)
    summary = compute_summary(edf, bdf, cfg)
    return {"blotter": bdf, "equity": edf, "summary": summary}


def compute_summary(equity: pd.DataFrame, blotter: pd.DataFrame, cfg: SimConfig) -> dict:
    if len(equity) == 0:
        return {}
    eq = equity["equity"].to_numpy(dtype=np.float64)
    start = float(eq[0])
    end = float(eq[-1])
    days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days
    years = days / 365.25 if days > 0 else 1e-9
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 and years > 0 else 0.0

    daily_ret = pd.Series(eq).pct_change().dropna().to_numpy()
    if len(daily_ret) > 0 and daily_ret.std() > 0:
        sharpe_daily = daily_ret.mean() / daily_ret.std()
        sharpe_ann = sharpe_daily * np.sqrt(252)
    else:
        sharpe_daily = 0.0
        sharpe_ann = 0.0

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    if len(blotter) > 0:
        wins = blotter[blotter["trade_return_pct"] > 0]
        losses = blotter[blotter["trade_return_pct"] <= 0]
        win_rate = len(wins) / len(blotter)
        win_sum = float(wins["trade_return_pct"].sum())
        loss_sum = float(abs(losses["trade_return_pct"].sum()))
        pf = win_sum / loss_sum if loss_sum > 0 else float("inf")
        avg_trade = float(blotter["trade_return_pct"].mean())
        med_trade = float(blotter["trade_return_pct"].median())
    else:
        win_rate = pf = avg_trade = med_trade = 0.0

    avg_pos = float(equity["n_positions"].mean()) if "n_positions" in equity else 0.0
    max_pos = int(equity["n_positions"].max()) if "n_positions" in equity else 0
    avg_cash = float((equity["cash"] / equity["equity"]).mean()) if "cash" in equity else 0.0

    return {
        "start_equity": start, "end_equity": end,
        "total_return": end / start - 1.0,
        "years": years, "cagr": cagr,
        "sharpe_annualized": sharpe_ann, "sharpe_daily": sharpe_daily,
        "max_drawdown": max_dd,
        "n_trades": int(len(blotter)),
        "win_rate": win_rate, "profit_factor": pf,
        "avg_trade_return": avg_trade, "median_trade_return": med_trade,
        "avg_positions": avg_pos, "max_positions": max_pos, "avg_cash_pct": avg_cash,
    }
