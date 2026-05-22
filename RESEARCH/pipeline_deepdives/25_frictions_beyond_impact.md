# Frictions beyond impact — bid-ask spread, halt risk, and tax drag

Almgren-Chriss impact (covered at [/story/09/09](/story/09/09)) is the headline transaction cost in academic literature, but it's not the only friction that erodes a momentum strategy's edge. Our simulator models or accounts for three additional realistic costs: **bid-ask spread**, **halt risk**, and **tax drag**.

## Bid-ask spread

The bid-ask spread is the gap between the highest bid and the lowest ask. A market order pays half the spread on entry and half on exit — you cross it twice per round trip. Even if the mid-price never moves, you lose the full spread to liquidity providers.

Spreads scale inversely with liquidity:

| Liquidity tier | Typical full spread | Per-side cost |
|---|---|---|
| Large-cap (AAPL, MSFT, post-vaccine BNTX) | 1-5 bps | 0.5-2.5 bps |
| Mid-cap | 5-20 bps | 2.5-10 bps |
| Small-cap | 20-50 bps | 10-25 bps |
| Microcap | 50-200 bps | 25-100 bps |

Our simulator's `slippage_bps_each_side` default is **5 bps** when `use_almgren_chriss_impact=False`. Reasonable for "decent liquidity" — fine for mid-cap and larger, optimistic for small caps. When Almgren-Chriss is enabled, the impact term already folds in a spread approximation.

The right long-term model is a separate `spread_bps` parameter scaled by liquidity tier rather than a single flat number. On the future-work list.

## Halt risk

Stocks halt trading for many reasons: LULD circuit breakers, news pending, regulatory inquiry, single-stock volatility pauses. While halted, you cannot enter or exit.

When `halt_risk_enabled=True`, the simulator applies tier-dependent halt rates:

| Tier | ADV range | Halt rate per attempt |
|---|---|---|
| Megacap | > $100M | 0.1% |
| Large-cap | $20-100M | 0.3% |
| Mid-cap | $5-20M | 0.5% |
| Small-cap | $1-5M | 1.5% |
| Microcap | < $1M | 3.0% |

For each fill attempt, the simulator draws `halt_rng.random()`; if it falls below the tier rate, the fill is missed. A failed *entry* means no position opens; a failed *exit* delays one day.

The aggregate effect on `lgbm_image` was small — roughly **0.5pp CAGR drag** — because its picks skew mid-cap and larger. A microcap-heavy strategy at 3% halt rate per attempt would see far worse drag.

## Tax drag

For US taxable accounts, anything held less than one year is a **short-term capital gain**, taxed at ordinary income rates:

| Component | Rate |
|---|---|
| Federal ordinary income (top bracket) | 32% |
| California state | 6% |
| **Effective short-term rate** | **38%** |

Our model holds positions for 40 trading days (~57 calendar days). **Every single gain is short-term.**

The `post_tax_cagr` function in `src/stock_chart/stats.py`:

```python
for each closed trade:
  gain = proceeds - cost
  if gain > 0:
    tax = gain × (federal_rate + state_rate)
    after_tax_proceeds = proceeds - tax
  else:
    # capital loss — offsets $3k/year ordinary income, carries forward
    after_tax_proceeds = proceeds + abs(gain) × loss_offset_rate
```

For the headline run:

| Measure | 10-year CAGR |
|---|---|
| Pre-tax | +4.3% |
| Post-tax (38% effective) | ~+2.6% |

**A 1.7pp drag from taxes alone** — larger than the halt drag and comparable to a bad year's impact cost.

## What's NOT in our simulator

- **Payment for order flow (PFOF) rebates.** Most retail brokers (Robinhood, IBKR Lite) receive PFOF revenue that partially subsidizes spread cost. We don't model this.
- **Borrow cost for shorts.** We only go long.
- **Cash-management drag.** Uninvested cash currently earns ~5% in money market funds; we count it as 0%.
- **Wash-sale rules.** A loss cannot be claimed if you re-buy within 30 days. A momentum strategy that rotates back into recently-sold names defers loss recognition. Ignored.

These omissions roughly cancel for a typical mid-cap momentum book.

---

> **Bottom line.** Stack the four frictions — spread, Almgren-Chriss impact, halts, and tax — and the combined drag typically eats **30-50% of pre-cost CAGR**. Only highly liquid, large-cap strategies preserve their edge after all four are applied.
