# Phantom positions and accounting integrity in the simulator

A backtester is, at heart, a bookkeeping machine. Every dollar must be somewhere — in cash, or in an open position marked to market. When a simulator silently loses track of a dollar, the equity curve it produces is fiction, and worse, it is *plausible* fiction. This deep-dive walks through a now-fixed bug in `src/stock_chart/simulator.py` that violated this principle without ever throwing an error, and the conservation invariant we now assert to keep it honest.

## The conservation invariant

The simulator must hold one accounting identity at every step of the [simulator loop](/story/09/06_simulator_loop):

```
cash + sum(position mark-to-market) == equity
```

This is just double-entry bookkeeping. Cash leaves the account when a position opens (debited by `cost + commission`) and returns when it closes (credited by `proceeds - commission`). At any instant, total wealth is cash plus the current value of what we hold.

Two corollaries follow at the end of a run, once every position is force-closed:

- **`cash == end_equity`** — with no open positions, marked value is zero, so all wealth is cash.
- **`sum(blotter pnl) ≈ end_equity - start_equity`** — the sum of realized trade P&L reconstructs the total return, up to the commissions netted into each leg.

These are exactly the cross-checks the [blotter/equity/summary](/story/09/34_blotter_equity_summary) reconciliation exists to enforce. If any fails, money was created or destroyed.

## How a phantom position breaks it — silently

The entry path read `anchor_close` and `adv20_usd` straight off the `scores` row. It never asked whether the ticker actually existed in `price_lookup`. Consider a ticker that appears in `scores` but has no price series. Three things went wrong, in sequence:

1. **Entry debits cash.** A position opens, `cash -= total_cost`, and a fill price is booked from the scores row.
2. **It never closes.** In `_close_due`, `ticker_idx.get(ticker)` returns `None`, the next-bar search yields nothing, and the position is pushed back onto `still_open`. It stays perpetually open.
3. **It vanishes at forced-close.** The end-of-sim liquidation begins with `if px_series is None: continue` — so the orphan is skipped, never producing a blotter row.

The net effect: cash is permanently trapped, `n_positions` is stuck above zero to the last day, and the trade is invisible — `n_trades` undercounts it.

What makes this insidious is the masking mechanism. The mark-to-market step has a fallback: for a position it cannot price, it marks at `entry_price`:

```python
mtm += p.shares * p.entry_price  # fallback
```

So the equity curve *looks* conserved. The phantom position's marked value exactly offsets the cash that left to open it, and the identity `cash + MTM == equity` holds — at a stale, fictional valuation. The leak is real (`cash != end_equity` at the end), but the headline equity number never flinches. You would never spot this by eyeballing a chart — the same "plausible wrong number" problem as [the row-alignment landmine](/story/09/41_row_alignment) and [the beta-zero bug](/story/09/42_beta_zero_bug).

## Why it was latent, and why it still mattered

In the real pipeline, `scores` and `price_lookup` are derived from the same source during [data acquisition](/story/09/13_data_acquisition) — every scored ticker has prices, so the divergence never occurred in practice. But "never in practice" is not "never." The moment a ticker is dropped from the price store between scoring and simulation — a delisting reclassification, a partial cache, a re-run against a trimmed price set — the hole opens. A backtester whose correctness depends on two inputs never diverging is a backtester with a latent landmine.

The fix is one guard at entry: skip any candidate whose ticker is absent from `price_lookup`.

```python
# Never open a position we cannot price or close.
if t not in ticker_idx:
    continue
```

If you cannot mark it and cannot close it, you must not open it.

## A consistency fix that rode along

While auditing the close paths, a second discrepancy surfaced. Normal exits cost slippage via `_exit_slippage_bps`, which scales with participation under [Almgren-Chriss](/story/09/09_almgren_chriss) when impact modeling is enabled. The end-of-sim forced-close, however, charged a flat `slippage_bps_each_side`. That meant liquidations were costed differently from ordinary exits — understating frictions for the very positions left holding at the bell. The forced-close now calls the same `_exit_slippage_bps`, matching the [friction model beyond raw impact](/story/09/25_frictions_beyond_impact) used everywhere else.

## The regression test

The discriminators live in the invariant suite. Feed a `scores` row for a ticker absent from `price_lookup` and assert: `n_trades == 0` (no blotter row), `n_positions == 0` at the last bar (no dangling open position), and `end_equity == start_equity` (no cash trapped). Every one fails pre-fix: the old code books a phantom, leaves it open to the final bar, and strands its cost. The companion invariant `test_cash_equals_equity_at_end_when_no_open_positions` asserts the corollary directly.

## The lesson

A backtester must never open a position it cannot mark and cannot close — and it must never trust that two inputs agree. The deeper guard is structural: assert the conservation invariant in tests, so that any future leak surfaces as a failing assertion rather than a quietly wrong equity curve. This is the heart of the [testing philosophy](/story/09/45_testing_philosophy) — invariants over outputs — and a representative chapter in the broader [hardening story](/story/09/39_hardening_story). Equity curves lie politely. Conservation checks do not.
