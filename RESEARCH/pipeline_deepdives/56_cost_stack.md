# The cost stack end to end — from gross alpha to net-of-everything

A backtest that reports a single number — "lgbm_engineered earned +4.3% CAGR" — has told you almost nothing useful until you know what was *subtracted to get there*, and what still hasn't been. Alpha is gross. The dollars in your account are net-of-everything. Between those two numbers sits a stack of cost layers, each of which scales differently with trade size, liquidity, and the AUM you deploy. This page walks one representative trade and one representative year down that stack, peeling off each layer in the order the simulator and tax model apply them. The headline: every layer is a strict subtraction, none is optional, and the layer that dominates depends entirely on what you're trading.

## Layer 0 — Gross (frictionless) return

Start with the fiction every backtest secretly reports first: you buy at the close, you sell at the resolution close, no fees, no spread, no impact, infinite liquidity. For our representative trade, the model picks a name at $50.00 and it resolves 40 trading days later at $52.15 — a clean **+4.30%** gross. Aggregate that across a year of ~1,100 such trades at ~25 concurrent positions and you get the headline **pre-cost CAGR ≈ +4.3%** for `lgbm_engineered`. This is the number that makes a strategy look fundable. It is also a *ceiling* — it can only go down from here.

## Layer 1 — Commission (per-share)

The simulator charges `min(shares × $0.005, 0.5% × trade_value)` per side. With a 4% position cap on $100k equity, a position is ~$4,000; at $50 that's 80 shares, so $0.40 per side, $0.80 round-trip — about **2 bps**. Trivial at this [position size](/story/09/32_position_size_formula). Commission is the one cost that *shrinks* as a fraction of trade value as you scale up, and the 0.5% cap only binds on absurdly cheap sub-$1 stocks. New trade return: **+4.28%**.

## Layer 2 — Bid-ask spread / base slippage

The default mode applies a flat **10 bps each side** — you buy slightly above the close, sell slightly below — a 20 bps round-trip haircut representing the spread you cross and minor timing slip, before any size-dependent impact. See [frictions beyond impact](/story/09/25_frictions_beyond_impact). Twenty bps off +4.28% leaves **+4.08%**. So far the stack looks gentle — ~22 bps. The reason people get burned is that the *next* layer doesn't show up in the flat-bps assumption at all.

## Layer 3 — Market impact (Almgren-Chriss)

Turn on `use_almgren_chriss_impact` and the flat 10 bps is replaced by a size-aware square-root law: one-way impact = `daily_vol × η × participation^β × (1 + permanent_frac/2)`, with η=0.142, β=0.5, permanent_frac=0.5, daily_vol=3% — clamped to [1, 500] bps. The driver is **participation = position_usd / [ADV20](/story/09/33_adv20_metric)**, and the square root means impact explodes as the name gets thinner. This is where [Almgren-Chriss](/story/09/09_almgren_chriss) earns its keep:

- **Mega-cap** (ADV $2B): a $4k position is participation = 0.000002. Impact ≈ **0.8 bps**, floored to the 1 bps minimum. Round-trip ~2 bps — *cheaper* than the flat assumption.
- **Small-cap** (ADV $8M, at the position cap of 0.5% × ADV = $40k): participation = 0.005. Impact ≈ **37 bps** one-way, ~75 bps round-trip — *each* on entry and exit.

For a portfolio filled mostly in thinner names, the realized blended impact lands near **60–90 bps round-trip**, dwarfing commission and spread combined. New trade return on the representative thin name: ~**+3.55%**.

## Layer 4 — Halt risk (missed fills)

With `halt_risk_enabled`, each candidate draws against a tier probability of a missed fill: large 0.1%, mid 0.3%, small 0.8%, micro 1.8% (`_HALT_RATES`). A missed fill books *no* P&L — but the trades you miss are not random: halts cluster in exactly the volatile thin names where the model's best signals live. Losing ~1% of fills, skewed toward the highest-expectancy thin trades, shaves a few more bps. Call it **+3.50%** net trade, and a portfolio **post-cost CAGR ≈ +2.8%** — a **~35% erosion** of the +4.3% gross, squarely inside the 30–50% band honest cost modeling produces.

## Layer 5 — Short-term capital gains tax

Every hold is ~40 days, so 100% is short-term gain. `post_tax_cagr` applies a blended **38.8%** (federal 32% + state 6% + NIIT 3.8%), taxed per-year on net gains. Tax hits *after* all trading frictions — you're taxed on the post-cost number. See [post-tax / frictions](/story/09/25_frictions_beyond_impact) and [CAGR/drawdown/Calmar](/story/09/28_cagr_drawdown_calmar). Applying ~38.8% lands the honest figure at **post-tax CAGR ≈ +2.6%**.

## The waterfall

| Layer | What's subtracted | Representative trade | Portfolio CAGR |
|---|---|---|---|
| 0. Gross (frictionless) | — | +4.30% | +4.3% |
| 1. Commission | ~2 bps round-trip | +4.28% | +4.28% |
| 2. Spread / base slippage | ~20 bps round-trip | +4.08% | +4.1% |
| 3. Almgren-Chriss impact | ~60–90 bps (thin) / ~2 bps (mega) | +3.55% | +3.0% |
| 4. Halt risk (missed fills) | skewed ~1% of best fills | +3.50% | +2.8% |
| 5. STCG tax (38.8%) | on post-cost gains | — | **+2.6%** |

## Capacity interaction — the same alpha shrinks as AUM grows

The waterfall above is computed at $100k, and it is *not* invariant to AUM. The 4% position cap means a larger account wants larger dollar positions, but the 0.5%-of-ADV cap holds the line — so above a few million in AUM, the strategy can no longer take a full 4% in the thin names and is forced down-list into worse signals, *or* it pushes participation up and pays the square-root impact penalty. Either way the realized edge decays. At ~$50M AUM the same signal may net well under +1%. This is the [capacity ceiling](/story/09/26_capacity_ceiling): impact dominates at the small/micro tier and at high AUM; commission and spread dominate only at the mega-cap, low-AUM corner where there's no edge left to harvest anyway.

## The honest number

A pre-cost backtest is a **ceiling, and usually a generous one** — here it overstated the deployable result by ~40% before a dollar of tax. The number you can actually spend is net-of-everything-*at-your-AUM*: +4.3% gross becomes +2.6% post-tax at $100k, and less as you scale. Report the full stack or report nothing. See [the verdict](/story/09/55_the_verdict), the [hardening story](/story/09/39_hardening_story), and [log vs simple returns](/story/09/35_log_vs_simple_returns) for why the compounding base matters when you chain a year of these together.
