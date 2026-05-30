# The simulator's per-day inner loop — exact ordering and why it matters

The event-driven backtester in `src/stock_chart/simulator.py` walks the [trading calendar](/story/09/37_trading_calendar) day by day. Each day it executes the same five-step ritual, in a very specific order. The ordering isn't aesthetic — every adjacent pair of steps has a load-bearing dependency, and swapping any two of them silently corrupts P&L, position counts, or the equity curve. This note pins down why the loop runs the way it does, with BNTX (2021-07-02 entry, 2021-08-30 exit at horizon `H=40`) as a running example.

## The five steps, in order

For each `today` on the [trading calendar](/story/09/37_trading_calendar):

1. **`_check_trailing_stop(today)`** — for every open position, lift `peak_close` if `close[today] > peak_close`; if `(peak_close - close[today]) / peak_close > trailing_stop_pct`, write `exit_date = today`.
2. **`_close_due(today)`** — for every open position whose `exit_date <= today`, fill at `close × (1 - slippage_bps / 1e4)`, deduct commission, credit proceeds to `cash`, append a [blotter](/story/09/34_blotter_equity_summary) row, remove from `open_pos`.
3. **Mark-to-market** — `equity = cash + sum(shares × close[today])` over remaining open positions, falling back to `entry_price` if today's close is missing.
4. **Record `daily_equity`** — append `{date, equity, cash, n_positions}`.
5. **Entry pass (anchor days only)** — iterate the day's candidates by descending score, apply ADV filter + no-duplicate-ticker rule + [position-size](/story/09/32_position_size_formula) cap, fill at `anchor_close × (1 + slippage_bps / 1e4)`, allocate `min(equity × max_position_pct, max_adv_pct × adv20_usd, cash)`, buy integer shares, append to `open_pos`. Cap at `max_new_per_week`.

## Why this exact order

**Stop check before close-due.** The trailing-stop step can *mutate* `exit_date`. If close-due ran first, it would only see the horizon-fixed exit dates set at entry time, and a stop that triggers today would be deferred to tomorrow — one day of unwanted exposure, plus the fill happens at the wrong close. Running stops first means close-due sees a fully updated set of `exit_date` values when it scans `exit_date <= today`.

**Close-due before MTM.** Cash from today's exits has to land in the cash bucket *before* equity is computed; otherwise the closed shares would be double-counted (still in `open_pos` at today's close) or, worse, zero-counted (removed from `open_pos` but proceeds not yet in `cash`). Running close-due first preserves the invariant `equity = cash + Σ(shares × close)` after step 2.

**MTM before entry.** The [position-size formula](/story/09/32_position_size_formula) uses `equity × max_position_pct`. If entries ran before MTM, sizing would use *yesterday's* equity, which is wrong on any day that had exits or material price moves. Putting MTM first means today's buys are sized off today's real equity.

**Entry last — and the deliberate one-day lag.** New positions enter `open_pos` only after `daily_equity` is already recorded. That row reflects the portfolio *before* today's fills, which is exactly what we want: you do not get to mark your own fill on day-0. The buy slippage (`+slippage_bps`) is already a cost; MTM'ing the fresh position at the same `anchor_close` would create a tiny phantom loss on entry day. Skipping it until tomorrow is the cleaner accounting.

## Stop + horizon firing on the same day

Suppose BNTX had dropped 20% from its peak on the same day its horizon expired. The stop check sets `exit_date = today`. Close-due then scans positions with `exit_date <= today` — the horizon-fixed exit (originally also today) satisfies the predicate, but BNTX is only in `open_pos` once, so it gets closed exactly once. There's no double-close because close-due iterates a `list(open_pos)` snapshot and removes from the live dict as it goes. The fill price is today's close either way, so the two paths converge.

In the real BNTX trade, the stop never fired — the position rode the full 40 trading days from 2021-07-02 to 2021-08-30 and exited at the horizon close on Aug 30. The stop branch was checked 40 times and was a no-op each time.

## Same-ticker same-day rotation

The loop deliberately allows the same ticker to exit and re-enter on the same day. Close-due calls `open_tickers_set.discard(ticker)` *before* the entry pass runs its no-duplicate-ticker check, so if BNTX exits on its horizon date and is also a top-scoring anchor candidate that same day, it can be re-bought. This is intentional rotation, not a bug — `test_no_strict_same_ticker_overlap` documents and pins this behavior. If you wanted strict cooldown, you'd hold the discarded ticker in a separate "just-closed-today" set and exclude it from the entry pass; we explicitly do not.

## The terminal force-close

After the final trading day, anything still in `open_pos` is force-closed at the last available close. This guarantees every [blotter](/story/09/34_blotter_equity_summary) row has both an entry and an exit — no dangling positions, no half-completed round trips. Without this step, end-of-backtest equity would have a long tail of unrealized P&L that the trade-level metrics couldn't see.

## Conservation invariants

After every step in every iteration, `cash + Σ(shares × close[today]) == equity` to the cent. At the final timestamp, with all positions force-closed, `Σ(shares × close) == 0`, so `cash == equity`. The integration test `test_cash_equals_equity_at_end_when_no_open_positions` asserts exactly this, and it's the single best smoke test for "did someone reorder the loop without thinking."

> **See:** `tests/test_simulator_invariants.py` — particularly `test_cash_equals_equity_at_end_when_no_open_positions`, `test_no_strict_same_ticker_overlap`, and the trailing-stop tests. If you touch the inner loop, run these first; if they pass, the ordering is still sound.
