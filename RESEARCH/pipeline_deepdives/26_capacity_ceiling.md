# Capacity ceiling — how much money this strategy could actually manage

## The capacity question

Every alpha source has a finite size. A strategy that prints 4% CAGR on a $50k account does not necessarily print 4% on $50M, because at $50M the strategy is no longer a passive observer of prices — it *is* the price-moving order flow. The mechanism is market impact: pushing size into a finite order book moves the print against you, and the move scales with how much of the day's volume you consume.

The standard reference is the [Almgren-Chriss](/story/09/09_almgren_chriss) impact model (see [/story/09/09](/story/09/09)):

```
impact ≈ σ_daily × η × sqrt(Q / V)
```

where `σ_daily` is the stock's daily vol, `η ≈ 0.142` is the calibration constant, `Q` is your traded shares, and `V` is the stock's average daily volume. The key fact: `Q` grows linearly with AUM, but impact grows as `sqrt(AUM)`. That sublinear growth is forgiving at small size and brutal at large size.

## The math, worked out for our strategy

- **Position sizing**: 5 picks per week, 4% per position = 20% deployed per cohort.
- **Cohort overlap**: 5 cohorts overlap, putting 100% of AUM at work at steady state.
- **Universe ADV**: median pick ADV ≈ **$50M**.
- **Pre-cost alpha**: 4% CAGR headline.
- **Slippage budget**: we can afford 50 bps round-trip per trade. Per-trade impact budget = **25 bps**.

## Solving for capacity

```
0.0025 = 0.03 × 0.142 × sqrt(Q / V)
sqrt(Q / V) = 0.587
Q / V       = 0.344
```

We can trade **34.4% of ADV per position**. With ADV = $50M, that is **Q ≈ $17.2M per position**. At 25 simultaneous positions × $17.2M ≈ **$430M deployed AUM**.

That headline is misleading. [Almgren-Chriss](/story/09/09_almgren_chriss) prices a *single-day* trade; 34% of ADV in one session is outside the model's calibrated regime. Real execution has to spread across multiple days.

## The realistic capacity ceiling, in three scenarios

- **One-day execution (aggressive)**: cap participation at ~8% ADV per day → ~$4M per position → **$50M AUM ceiling**.
- **Multi-day TWAP, 3 days**: impact drops by `sqrt(3) ≈ 1.7×` → **~$200M AUM ceiling**.
- **Multi-day TWAP + universe broadening** (drop the [MA250 prefilter](/story/09/07_ma250_prefilter), allow large caps): **$500M+ AUM ceiling**, at the cost of changing what the strategy is.

## The cost of multi-day execution

Splitting an entry across 3 days reduces impact, but it introduces opportunity cost. For a holding period of ~40 trading days, a 3-day execution window costs roughly `3/40 = 7.5%` of the per-trade alpha. Tolerable — the impact reduction (1.7×) more than pays for the decay (0.93×).

Beyond 5 days the trade-off inverts.

## Why capacity matters for any reader

If our report card says "4% CAGR pre-cost" and stops there, a retail reader and a $50M family office walk away with the same number — and one of them is wrong. Capacity-honest reporting is part of the honest-report-card requirement.

## The current honest report for `lgbm_engineered`

| AUM | Regime | Expected impact | Status |
|---|---|---|---|
| **$50k retail** | Single-day fills | < 5 bps per trade | Feasible — impact well inside budget |
| **$500k aggressive retail** | Limit-on-close | ~15 bps per trade | Feasible — most of the edge survives |
| **$5M small-fund** | 3-day TWAP required | ~30 bps per trade | Workable — alpha shrinks ~10% |
| **$50M+** | Multi-day + universe broadening | > 50 bps per trade | Edge fully eroded |

## The sweet spot

The strategy lives between **$50k and $5M**. Below $50k, fixed costs dominate. Above $5M, impact takes over. In the middle, the model's edge survives intact.

That window is narrow, but it is real, and it is honest.

---

> **See also**: [/story/09/09](/story/09/09) and [/research/04_costs_capacity](/research/04_costs_capacity).
