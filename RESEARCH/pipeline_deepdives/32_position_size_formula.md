# Position size — the three-way `min(equity × pct, max_adv × adv, cash)` formula

## The three caps in one expression

Position sizing in the [simulator](/story/09/06_simulator_loop) lives in five short lines:

```python
target_usd = min(
    equity_now * cfg.max_position_pct,    # cap 1: percent of total equity
    cfg.max_adv_pct * adv20_usd,           # cap 2: percent of stock's daily volume
)
target_usd = min(target_usd, cash)         # cap 3: available cash
if target_usd < 100:
    continue                                # skip tiny positions
shares = int(target_usd // fill_price)     # integer shares only
```

Each `min` argument exists to neutralize a specific failure mode: **concentration risk**, **market impact**, and **insolvency**.

## Cap 1 — `equity_now * max_position_pct` (diversification)

With `max_position_pct = 0.04`, no single new position can exceed **4% of total equity**. The arithmetic:

- 5 fresh picks per week
- ~5-week average hold
- 5 × 5 = **25 simultaneous positions** at steady state
- 25 × 4% = **100% deployed**

That's the design point: fully invested, no leverage, no single name large enough to wreck the book on a -50% surprise day. See [Allocation budget](/story/09/36_allocation_budget) for the full derivation.

## Cap 2 — `max_adv_pct * adv20_usd` (liquidity / impact)

`max_adv_pct = 0.02` says: never let a position be more than **2% of the stock's 20-day average dollar volume**. If you become a meaningful fraction of daily flow, the market starts moving against you.

The 2% number isn't arbitrary. The [Almgren-Chriss impact model](/story/09/09_almgren_chriss) gives temporary impact roughly proportional to `(participation_rate)^0.5`. At 2% participation, modeled impact stays under ~10 bps. See [ADV20 metric](/story/09/33_adv20_metric).

## Cap 3 — `min(target_usd, cash)` (solvency)

You can't spend cash you don't have. This cap is mostly binding in two regimes:

1. **Early weeks**, before the first cohort has rotated out.
2. **Drawdowns**, when many positions are deep underwater and tying up cash.

Without this guard, the simulator silently goes on margin.

## Worked example — BNTX, 2021-07-02

Starting equity = $100k, no prior closed trades.

```
equity_now        = $100,000
max_position_pct  = 0.04
cap1 (equity)     = 0.04 * 100,000        = $4,000
adv20_usd (BNTX)  = $524,000,000
max_adv_pct       = 0.02
cap2 (liquidity)  = 0.02 * 524M           = $10,480,000   # not binding
cash              = $100,000              # untouched
cap3 (cash)       = $100,000              # not binding
target_usd        = min(4000, 10.5M, 100k) = $4,000        # cap 1 wins
fill_price        = 221.07 * (1 + slip)   ~ $221.18
shares            = int(4000 // 221.18)   = 18
actual position   = 18 * 221.18           = $3,981.24
```

Cap 1 wins because BNTX in July 2021 was enormously liquid. For a microcap, cap 2 would have bound first. That's the whole point of carrying both caps: [capacity scales with the smaller stock, not the bigger account](/story/09/26_capacity_ceiling).

## The integer-shares floor

`int(target_usd // fill_price)` rounds **down**. Two reasons:

- Most retail brokerages don't support fractional shares.
- Rounding down is conservative.

For BNTX at $221, that leaves $19 on the table out of $4,000 — about 50 bps.

## The `< $100` floor

`if target_usd < 100: continue` catches degenerate cases. Even at $0.005/share, a round-trip on a $50 position can erode 1-2% just on fees, before slippage. Pure cost, negligible signal.

## Sensitivity — what happens when you turn the dials

**Bumping `max_position_pct` from 4% → 8%** halves the concurrent position count from ~25 to ~12-13. Concentration risk explodes. The model's edge isn't strong enough to underwrite concentrated bets — diversification is doing real load-bearing work.

**Loosening `max_adv_pct` from 2% → 5%** increases capacity at small account sizes, but [Almgren-Chriss](/story/09/09_almgren_chriss) impact roughly doubles. See [Capacity ceiling](/story/09/26_capacity_ceiling).
