# Average Daily Volume — the ADV20 metric, why 20 days, edge cases

The ADV20 (20-day average daily dollar volume) is one of the most load-bearing numbers in our pipeline. It sneaks into nearly every downstream stage: it caps how much we can buy via the [position-size formula](/story/09/32_position_size_formula), it gates which stocks the simulator considers via the MIN_ADV filter, it parameterizes the [Almgren-Chriss impact model](/story/09/09_almgren_chriss), and it sets the ceiling in [capacity analysis](/story/09/26_capacity_ceiling).

## Definition

ADV20 is the mean of `close × volume` over the past 20 trading days. We multiply *shares* by *closing price* to get **dollar volume**, because dollars — not share counts — determine whether our trade is a rounding error or a market-moving event.

## Why 20 days specifically

Twenty trading days ≈ one calendar month is the practitioner default:

- **Long enough to smooth single-day spikes.** A major earnings day can be 5x normal volume.
- **Short enough to reflect current liquidity.** A stock that surged 6 months ago but has since gone quiet shouldn't be sized off its old volume.

Shorter windows (5–10 days) are too jumpy; longer (60–90) are too lagged.

## The formula

```
ADV20[t] = mean( close[t-20..t-1] × volume[t-20..t-1] )
```

Strictly past 20 days. No peeking at today. Causal.

## Concrete number: BNTX on 2021-07-02

- ADV20 = **$524M**
- Higher than typical mid-cap because of post-vaccine retail interest
- Our weekly $4k × 5 picks = $20k total is **0.004%** of BNTX's daily flow — a rounding error

## Distribution across our universe

| Tier      | ADV20 range      | Examples |
|-----------|------------------|----------|
| Megacap   | $5B–$50B         | AAPL, MSFT, NVDA |
| Large-cap | $500M–$5B        | most S&P 500 |
| Mid-cap   | $50M–$500M       | BNTX sits here |
| Small-cap | $5M–$50M         | many Russell 2000 |
| Microcap  | < $5M            | **our cutoff** |

## The $5M MIN_ADV filter

The [simulator](/story/09/06_simulator_loop) drops candidates with `adv20_usd < $5M`:

- At $5M ADV, our $4k position is 0.08% of daily flow → impact stays under 1 bp
- Below $5M, the impact model says we start materially moving the market
- The model's edge concentrates in mid+ caps anyway

Crucially, the [random baseline](/story/09/20_simple_baselines) draws from the **same prefiltered universe**, so it inherits the same liquidity floor.

## Edge cases

- **IPO with < 20 days of history.** ADV20 is `NaN`. The pipeline drops these at the anchor date. See [universe construction](/story/09/14_universe_construction).
- **Trading halts inside the window.** A halt day's volume might be 0. The 20-day mean smooths a single halt out.
- **Recent reverse split.** Share volume drops 10x, price rises 10x. Since we use $-volume, the split is largely invisible. Edge protected.
- **Forward split.** Symmetric.
- **Delistings mid-window.** The stock exits before the anchor.

## The (Q/V) ratio in [Almgren-Chriss](/story/09/09_almgren_chriss)

The [Almgren-Chriss impact model](/story/09/09_almgren_chriss) uses ADV20 as `V`:

- `Q` = `position_usd` (our trade size)
- `V` = `adv20_usd` (stock's typical daily flow)
- Impact scales as `sqrt(Q/V)`

Worked examples at our $4k [position size](/story/09/32_position_size_formula):

| Stock type | ADV20 | Q/V | sqrt(Q/V) ≈ impact |
|---|---|---|---|
| BNTX (mid-cap) | $524M | 7.6e-6 | ~0.3 bps |
| Small-cap floor | $5M | 8.0e-4 | ~2.8 bps |
| Hypothetical microcap | $500K | 8.0e-3 | ~9 bps |

The microcap row is exactly why the $5M floor exists — impact grows as the *square root* of `Q/V`, so dropping from $5M to $500K ADV roughly **triples** the cost. Filter first, model second.
