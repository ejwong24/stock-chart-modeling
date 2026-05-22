# The trailing stop, the horizon exit, and what happens when both fire

Every position in the simulator has an expiration date stamped on it the moment it opens. That date is `anchor + H` where `H = 40` trading days — the horizon. The trailing stop, when enabled, is the only mechanism that can pull that date earlier. This piece walks through how the two exits coexist in `src/stock_chart/simulator.py`, the edge cases that make the design quietly elegant, and one known limitation around fill prices.

## One exit mechanism, two ways to set the date

When a position opens, `exit_date = resolution_date`. The daily loop then does two things in order: first `_check_trailing_stop(today)` runs, and only after that does `_close_due(today)` close everything whose `exit_date <= today`. The stop check never closes anything directly — it just shortens `exit_date` to `today`. The actual close then happens in the same pass through `_close_due`, at today's close price.

Why bother with the indirection? Because it keeps the closing logic uniform. There is exactly one path that turns a position into a blotter row, and it doesn't care *why* the exit date is what it is. The horizon exit and the stop-out share the same accounting, the same PnL math, the same realized-return computation. The stop is just a `min(exit_date, today)` operation in disguise.

## Edge case: stop fires before the horizon

Most interesting case. Stop fires on day D, where D < horizon_exit_date. The simulator overwrites `exit_date = D`, falls through to `_close_due`, and closes at D's close. The blotter row records `exit_date = D` — not the original horizon date — which is what you want for any downstream analysis that asks "how long was this position held?"

## Edge case: stop fires exactly on the horizon

If the stop happens to fire on the same day the horizon would have exited anyway (D == horizon_exit_date), nothing observable changes. `exit_date` gets set to D, which it already was. `_close_due` closes at D's close. From the blotter's perspective, you cannot tell which mechanism "won" — and that's fine, because the realized return is identical.

## Edge case: trailing_stop_pct = 0.0

The check `today_close / peak_close - 1.0 <= -trailing_stop_pct` becomes `drawdown <= 0.0`. In finite-precision arithmetic with strictly positive peaks, this can only fire if the close equals or exceeds the peak by zero — which means it doesn't fire in practice for any meaningful drawdown. Effectively the stop is disabled, and every position exits at horizon.

## Edge case: trailing_stop_pct = 1.0

A 100% drop from peak means the price has gone to zero. For any traded equity, this is unreachable on the timescale of a 40-day hold. So `pct = 1.0` and `pct = 0.0` produce indistinguishable simulator output — every position exits at horizon. This is locked in by `test_trailing_stop_1pt0_never_triggers`, which is a useful sanity rail: it confirms the stop logic isn't doing something weird at the boundary.

## The fill-price limitation

The original design intent — surfaced in the realistic-cost research notes — was for a stop to fill at the **next day's open**, not today's close. That matches how a retail stop order actually executes: the trigger happens intraday, but the fill is at whatever the next session opens at. The current implementation fills at today's close, which is optimistic. On a quiet day the difference is a few basis points. On a gap-down day it can be material — the close that triggered the stop will systematically be better than the open that would have actually filled. This is documented as a known limitation; we under-count slippage by some amount that scales with overnight gap variance.

## How peak_close tracks

At entry, `peak_close = entry_price`. Each subsequent day, if `close[t] > peak_close`, peak updates. The stop's reference point is therefore not the entry — it's the highest close seen *since* entry. A position can be up 30% from entry, then drop 18% from that peak, and the stop fires even though the position is still profitable from cost basis. This is the whole point of a trailing stop: protect gains, not just floor losses.

## End-of-data force-close

If a position is still open when `daily_dates` runs out — meaning the price data ends before the horizon does — the simulator force-closes at the last available close. This handles tickers with incomplete tails (recent IPOs near the end of the backtest window, delistings, etc.) without leaving zombie positions on the books.

## BNTX 2021-07-02 → 2021-08-30

Concrete example with `trailing_stop_pct = 0.18`. BNTX peaked around $442 on Aug 9 (roughly day 24 of the hold). 18% below that peak is ~$362. The position exited at $340 on Aug 30 — clearly below $362, so a naive look at the endpoints suggests the stop *should* have fired sometime between Aug 9 and Aug 30.

Whether it actually did depends on the daily path. The stop check sees one day at a time: it only fires the *first* day the drawdown-from-peak crosses 18%. If BNTX descended gradually from $442, the stop probably fired well before Aug 30 and the blotter exit date would reflect that earlier day. If it held above $362 most of the way down and only broke below in the final days, the stop fired late. To resolve this, pull the daily closes for BNTX in that window from `reports/full/equity_lgbm_engineered.parquet` and walk the peak/drawdown forward by hand — it's a five-minute exercise that the blotter doesn't directly answer.

---

> **Where did 18% come from?** The default trailing stop was picked through the falsification process in [`/story/05`](/story/05), which is also where you'll find the sensitivity analysis around it.
