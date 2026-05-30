# The allocation budget — 5 picks × 4% × 5 cohorts = 100% deployed

## The arithmetic that defines the strategy

```
max_new_per_week  = 5      # new positions per anchor date
max_position_pct  = 0.04   # 4% of equity at entry
hold_horizon      = 40     # trading days
cohorts_in_flight = 40 / 5 = 8 weeks  →  8 cohorts overlap

peak_deployed = 5 × 8 × 0.04 = 1.60   →  160% of equity at peak overlap
```

In practice we never see 160%. Same-week duplicates, ADV filtering, and the [capacity ceiling](/story/09/26_capacity_ceiling) trim the universe so the average concurrent book is closer to **25 names**, landing the steady-state deployment at **~100% of equity** — fully invested, never leveraged.

## Why we don't leverage

The simulator's third capital constraint — `cash >= position_usd` at entry — is a hard clamp. The 160% number is a *headroom* calculation, not a target. See [Position size formula](/story/09/32_position_size_formula).

## The per-week cash flow

On $100k equity:

```
new positions:    5 × $4,000 = $20,000 deployed
exiting cohort:   5 × $4,000 = $20,000 returned (roughly)
net change:                  ≈ $0
```

About $20k changes hands every week. For a median pick with `ADV20 ≈ $50M`, the per-name participation is `Q/V = $4k / $50M = 8e-5`, giving roughly **0.06 bps one-way impact**. See [ADV20 metric](/story/09/33_adv20_metric) and [Almgren-Chriss](/story/09/09_almgren_chriss).

## Why 5 picks (not 3, not 10)

```
3:  one bad week dominates the cohort — cohort variance too high
5:  picks 1–5 are the high-confidence band
10: picks 6–10 dilute the signal (rank-10 confidence ≪ rank-1)
```

Empirically the per-rank edge drops sharply past rank 5.

## Why 4% per position

```
1% × 25 names = 25% deployed   →  too much idle cash
10% × 5 names = 50% concentrated  →  single-name blow-up risk
4% × 25 names = 100%  →  fully deployed, diversified
```

A Kelly-style argument says "concentrate when edge is large." Our edge is small and noisy, so diversification dominates. See [Position size formula](/story/09/32_position_size_formula).

## Why 40-day hold

```
5 days:    high turnover, costs eat the alpha
40 days:   matches empirical momentum decay (~60d half-life)
100 days:  signal has decayed; capital trapped in dead positions
```

Forty trading days = eight weeks is the practitioner default. See [Weekly anchors](/story/09/18_weekly_anchors).

## Compounding mechanic

[Position size](/story/09/32_position_size_formula) is a *percentage*, not a dollar amount. As equity grows the absolute size grows linearly:

```
equity $100k → 4% = $4,000 per name
equity $200k → 4% = $8,000 per name
equity $1M   → 4% = $40,000 per name
```

The strategy is a "growth-of-one-dollar" engine. The same math triggers the [capacity ceiling](/story/09/26_capacity_ceiling) eventually — at some equity level the 4% slot starts to push the [Almgren-Chriss](/story/09/09_almgren_chriss) impact past the per-trade alpha.

## Drawdown behavior

In a drawdown, equity shrinks and so do new positions. We do **not** double down:

```
equity $100k → $80k drawdown → 4% × $80k = $3,200 per name
```

There's no martingale, no inverse-Kelly bet-up. The strategy stays calm and small in the storm.

## Sensitivity

Bumping `max_position_pct` from 4% → 6%:

```
6% × 25 = 150% theoretical → ~110% after cash clamp
```

Returns rise marginally, concentration risk rises noticeably, capacity drops. The cumulative default of `5 × 4% × 40d` is well-tuned — not magical, but coherent with the rest of the stack.
