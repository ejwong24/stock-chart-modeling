# Look-ahead in the exit fill — the price-gap bug

## The setup: why a master union calendar exists

The simulator's inner loop advances one trading day at a time, marking every open position to market and closing any that have come due. But "one trading day" is not a property of a single ticker — it's a property of the *portfolio*. Different names trade on different days: a halt, a data hole, or a late listing means ticker A might have no bar on a day when ticker B traded normally. If the loop only advanced on dates that *every* ticker shared, it would skip days the portfolio was actually live, and mark-to-market would be wrong.

So the simulator builds a **master calendar = the union of all tickers' trading dates** and iterates that. See [the simulator inner loop](/story/09/06_simulator_loop): the `for today in daily_dates` loop needs a common clock so equity is marked on every day *any* position could move.

The union calendar is correct for marking. The bug was in how the *close* logic interacted with it.

## The bug

When a position's `exit_date` arrives, `_close_due` needs the ticker's close on that date. The complication: `exit_date` lives on the master union calendar, but the *price* lookup is per-ticker. If this ticker has a gap straddling its exit, the date is in the union (because some other ticker traded) but **not in this ticker's own price index**.

The old code handled the missing bar by reaching for the next available one:

```python
after = px_series.index[px_series.index >= p.exit_date]
actual_exit = after[0]          # could be DAYS in the future
exit_close = float(px_series.loc[actual_exit])
cash += proceeds               # ...but booked NOW, on `today`
```

`after[0]` is the next real bar at or after `exit_date`. If the gap is several days wide, that bar is in the **future** relative to the current `today`. The simulator then credited cash *now*, on `today`, computed from a price the loop hadn't reached yet. That is textbook look-ahead: on day D you realize proceeds from day D+k's close.

## A worked example

Take ticker A missing `dates[60..64]` and ticker B complete, with A's `resolution_date = dates[62]`:

```
master union calendar:  ... 59  60  61  62  63  64  65 ...
ticker B (complete):    ... ✓   ✓   ✓   ✓   ✓   ✓   ✓
ticker A (gapped):      ... ✓   ·   ·   ·   ·   ·   ✓
                                          ^exit_date     ^next real bar
```

The loop reaches `today = dates[62]` (it's on the union because B traded). A's exit is due, but `dates[62]` isn't in A's index. The old code computes `after[0] = dates[65]`, reads that close, and books the exit **on `dates[62]` using the `dates[65]` price** — three days of hindsight baked into the fill.

The fix defers instead of reaching forward:

```python
actual_exit = after[0]
if actual_exit > today:
    # real exit bar is in the FUTURE; filling now = look-ahead.
    p.exit_date = actual_exit   # roll forward, keep position OPEN
    still_open.append(p)
    continue
```

A stays open through `dates[62]`, `dates[63]`, `dates[64]` — marked to market each day at its last known price — and only when the loop actually reaches `dates[65]` does `_close_due` find `exit_date == today`, with a real bar present, and fill. No future price is ever touched.

## Small, but exactly the wrong kind of small

This bug is invisible in aggregate. It only fires on tickers whose calendars have gaps — halts, holes, late listings. Most names trade every day the portfolio is live, so most exits hit the clean branch. In a portfolio-level Sharpe or CAGR, the effect washes out.

The problem is *which* names it biases. Gaps cluster in precisely the illiquid and halted tickers — the ones already most fragile to execution assumptions. See [ADV20](/story/09/33_adv20_metric): low-ADV names are likeliest to have thin or interrupted histories, and [frictions/halts](/story/09/25_frictions_beyond_impact) explains why those are the names where realism matters most. A look-ahead edge concentrated in halted small-caps flatters exactly the trades you should trust least.

## The regression test

`tests/test_audit_fixes.py::test_simulator_no_lookahead_exit_across_gap` reproduces the example above: ticker A missing `dates[60:65]`, B complete, A's `resolution_date = dates[62]`. The discriminating assertions are not on `exit_date` alone:

```python
assert int(eq.loc[dates[63], "n_positions"]) == 1   # still OPEN in the gap
assert int(eq.loc[dates[64], "n_positions"]) == 1
assert pd.Timestamp(blot.iloc[0]["exit_date"]) == dates[65]
assert int(eq.loc[dates[66], "n_positions"]) == 0   # closed after real bar
```

The `n_positions == 1` checks *during* the gap are what make this test sharp. The fixed code rolls `exit_date` forward to `dates[65]`, so the [blotter](/story/09/34_blotter_equity_summary)'s `exit_date` ends up at `dates[65]` either way — but the *old* code recorded that same `dates[65]` value while having already closed and booked cash back on `dates[62]`. Only the open-position-count assertions distinguish "held through the gap and exited at the real bar" from "force-closed early with a future price." Checking the exit date alone would have passed both versions — a lesson explored in [testing philosophy](/story/09/45_testing_philosophy).

## The general lesson

Any backtest that mixes a **union calendar** with **per-asset price lookups** carries this hazard: the loop's clock and the data's clock disagree, and the seam between them is where look-ahead hides. The rule is blunt — *never realize a fill at a date the current loop iteration hasn't reached.* If the data forces you forward in time to find a price, the only safe move is to wait, not to peek.

This was one of several seam bugs surfaced in [the hardening story](/story/09/39_hardening_story), and it rhymes with the [trailing stop interactions](/story/09/11_trailing_stop_interactions): both are cases where exit logic must respect the loop's notion of "now" rather than the data's notion of "next available."
