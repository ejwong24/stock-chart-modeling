# The Almgren-Chriss square-root impact model — what it is and what it costs us

## Plain-language motivation

When you buy a stock, you are not a passive observer of the price. Your buying pressure consumes liquidity on the offer side of the book, walks the price up against you, and signals to other participants that someone wants in. The market pushes back. The bigger your trade relative to the stock's normal flow, the harder it pushes.

For decades, traders observed that this push does not scale linearly with size. Doubling your order does not double the impact — it multiplies it by roughly `sqrt(2) ≈ 1.41`. This is the **square-root law of market impact**, formalized by Almgren, Thum, Hauptmann, and Li in 2005 from a dataset of thousands of Citigroup US equity orders, and reconfirmed many times since by Bouchaud, Tóth, BARRA, and others. It is one of the most robust empirical regularities in microstructure.

Our simulator, when run with `use_almgren_chriss_impact=True`, charges every fill a slippage cost computed from this law. The rest of this note unpacks the formula, walks through a worked example on BNTX, and shows where it stops being a rounding error.

## The formula, term by term

```
impact_bps = daily_vol * eta * (position_usd / adv_usd) ** beta
one_way    = impact_bps * (1 + permanent_frac / 2)
slippage_bps = one_way * 1e4
```

Four knobs, each with a real-world meaning:

- **`daily_vol`** — the stock's daily return standard deviation. Default `0.03` (≈48% annualized). High-vol names are punished harder because traders are more aggressive into them and the price moves more per unit of pressure.
- **`eta = 0.142`** — the universal proportionality constant. Almgren et al. (2005) fit this single number across the entire cross-section of US equities and found it remarkably stable. We use their value as-is.
- **`(Q/V)^beta`** — `Q` is the dollar size of your order, `V` is the stock's average daily dollar volume (ADV). `beta = 0.5` is the square-root exponent. Trading 1% of daily volume yields `sqrt(0.01) = 0.1` of the per-unit-volume baseline impact.
- **`permanent_frac = 0.5`** — Almgren-Chriss splits impact into a temporary component (decays after your trade finishes) and a permanent component (the price never comes back). The 50/50 split is the canonical assumption. We add half the permanent piece to the one-way cost.

## Worked example: BNTX, 2021-07-02

Concrete numbers from a real fill in the simulator.

```
anchor_close   = $221.07
position_usd   = $4,000        # ~18 shares
adv20_usd      = $524,000,000  # BNTX is heavily traded post-vaccine
daily_vol      = 0.03          # default

Q / V          = 4000 / 524_000_000 = 7.6e-6
(Q/V)^0.5      = 0.00276
impact         = 0.03 * 0.142 * 0.00276 = 1.18e-5
one_way        = 1.18e-5 * (1 + 0.5/2) = 1.47e-5
slippage_bps   = 0.147 bps
```

About **0.15 bps one-way, ~0.3 bps round-trip**. For a high-volume mid-cap like BNTX, Almgren-Chriss is essentially free. You are a rounding error on the tape and the model says so.

## Where the model bites: small-caps

Hold the trade size at $4,000 but drop ADV to $2M — a typical small-cap.

```
Q / V          = 4000 / 2_000_000 = 2e-3
(Q/V)^0.5      = 0.0447
impact         = 0.03 * 0.142 * 0.0447 = 1.90e-4
one_way        = ~19 bps
round_trip     = ~28 bps (with permanent half)
```

That is already a meaningful chunk of the 25% return threshold we use to label "winners." Two trades and you have spent more than 50 bps on slippage alone, before commissions or spreads.

## Microcaps: where it actually hurts

Push ADV down to $200k — a thinly-traded microcap, the kind of name that frequently surfaces in momentum-prefiltered universes.

```
Q / V          = 4000 / 200_000 = 0.02   # 2% of daily volume
(Q/V)^0.5      = 0.141
impact         = 0.03 * 0.142 * 0.141 = 6.0e-4
one_way        = ~60 bps
round_trip     = ~90 bps
```

Nearly **1% round-trip on a single $4k trade**. And that is assuming our default volatility of 3% daily — bump it to 5% (not unusual for microcaps) and the cost jumps another 67%.

## Why this matters for the strategy

The simulator's highest-ranked picks tend to skew small and micro. Two reinforcing reasons:

1. The universe is pre-filtered for momentum, and microcaps move more (higher vol, more dispersion, more 25%+ winners per unit time).
2. Several of our models — `lgbm_image` in particular — learn features that fire harder on noisy, low-liquidity names.

Without Almgren-Chriss, the simulator hands these picks a free pass: it fills $4k at the anchor close as if you were the only participant in the market. With Almgren-Chriss, the cost of touching an illiquid name is priced in, and many of those picks stop being profitable.

## The aggregate damage

In the realistic-cost run (commissions + spread + Almgren-Chriss combined), `lgbm_image` lost **~10 percentage points of CAGR** versus its frictionless backtest. Almost all of that haircut came from the impact term — its picks skewed toward names where `Q/V` was no longer a rounding error.

The cheaper-liquidity models lost far less. That delta — the gap between strategies that pick liquid names and strategies that pick illiquid ones — is exactly what Almgren-Chriss is for.

---

> See [`/story/04_costs`](/story/04_costs) for the full per-model breakdown, including the side-by-side CAGR table that quantifies the damage.
