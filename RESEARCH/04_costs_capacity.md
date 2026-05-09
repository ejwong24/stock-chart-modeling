# Problem 4 — Realistic transaction costs + capacity analysis

## Synthesis (top action)

**Cost stack at $100k AUM (microcap-tilted, 40-day holds, 6.25 turns/yr):**

| Line item | Round-trip bps |
|---|---|
| Commission (IBKR Pro proxy) | 0–2 |
| Bid-ask spread (half × 2) | 60–120 |
| Almgren-Chriss temporary impact | 15–80 (scales with √(Q/ADV)) |
| Permanent impact (~½ temporary) | 10–40 |
| Slippage vs decision price | 20–60 |
| Halt / locked-limit tail risk | 5–15 |
| Round-lot drag (only at <$25k AUM) | 0–3 |
| **Total round-trip per trade** | **~110–320 bps** |
| **Annualized drag (× 6.25 turns/yr)** | **~7–20%** |
| Plus tax drag (40-day → 100% STCG @ ~38%) | **~7%** |

**Honest CAGR conversion:**
| Pre-cost CAGR (backtest) | Post-cost @ $100k | @ $1M | @ $10M |
|---|---|---|---|
| 25% | ~14% | ~12% | ~3% |
| 6% (current sim) | ~-2% to -6% (negative!) | ~-4% | strategy dead |

**Tradable-AUM ceiling:** ~$3-5M taxable, ~$10M tax-deferred (IRA). Above this, costs explode AND capacity disappears simultaneously.

**Single most-impactful change:** Replace fixed 10-bps slippage with tier-based or Almgren-Chriss square-root impact model. Microcap reality is 60-200 bps round-trip.

---

## A1 — Almgren-Chriss square-root impact (HIGH severity)

Temp impact = σ × η × (Q/V)^β with β ≈ 0.5, η ≈ 0.142 (Almgren et al. 2005). Permanent ≈ ½ temporary. Round-trip ≈ 3 × one-leg temporary.

Worked: AAPL ADV $30B, σ=2%/d, Q=$10M = 0.033% ADV → impact 0.20 bps. Microcap $5M ADV, σ=4%/d, Q=$20k = 0.4% ADV → impact 25 bps each side, 50 bps round-trip.

**Drop-in for simulator.py:**
```python
def impact_bps(position_usd, adv_usd, daily_vol, eta=0.142, beta=0.5,
               permanent_frac=0.5, min_bps=1.0, max_bps=200.0):
    if adv_usd <= 0 or position_usd <= 0: return min_bps
    temporary = daily_vol * eta * (position_usd / adv_usd) ** beta
    one_way = temporary * (1.0 + permanent_frac / 2.0)
    return float(max(min_bps, min(max_bps, one_way * 1e4)))
```

**Honest CAGR drag vs current 10-bp model:**
- Large-cap-only: +30-80 bps/yr **better** (we were overcharging)
- Mid-cap blend: -50 to -150 bps/yr worse
- Microcap-tilted: -250 to -500 bps/yr worse

## A2 — Microcap real-world impact data (HIGH severity)

Tiered cost table (round-trip bps, includes spread + impact + PFOF):

| Cap tier | Quoted spread | Effective spread | Impact @0.1% ADV | PFOF | **Total RT** |
|---|---|---|---|---|---|
| Mega (>$200B) | 1-2 | 0.7-1.5 | 1-2 | 0-1 | **3-7** |
| Large ($10-200B) | 3-8 | 2-6 | 3-6 | 1-2 | **8-22** |
| Mid ($2-10B) | 10-25 | 7-18 | 8-15 | 2-4 | **25-55** |
| Small ($300M-$2B) | 30-80 | 25-75 | 20-40 | 5-10 | **80-200** |
| Micro (<$300M) | 100-400 | 110-450 | 40-80 | 8-15 | **250-800** |

For our universe (60% small + 25% micro + 15% mid), volume-weighted ~190 bps round-trip. At 6.3 turns/yr → **~12% annual drag**. A 6% gross becomes **-6% net**. Strategy needs ≥15% gross OR universe needs hard ADV floor of $5M+ and market-cap floor of $2B.

## A3 — Bid-ask spread modeling (MED-HIGH severity)

yfinance has no bid/ask. Roll (1984) implied spread = `2 * sqrt(-cov(p_t-p_{t-1}, p_{t+1}-p_t))` from daily closes (60d window per ticker, median-clip when cov goes positive).

```python
def estimate_spread_bps(close_price, adv_usd, market_cap_proxy):
    tick_floor = 0.5 * 10000.0 / max(close_price, 1.0)
    adv_term = max(0.0, 8.0 * np.log10(1e9 / max(adv_usd, 1e5)))
    size_term = 6.0 * max(0.0, np.log10(1e10 / max(market_cap_proxy, 1e7)))
    penny_penalty = 25.0 if close_price < 5.0 else (10.0 if close_price < 10.0 else 0.0)
    return float(np.clip(tick_floor + adv_term + size_term + penny_penalty, 1.0, 250.0))
```

Multiply by 1.8× if executing at market open/close (auction prints 2-3× wider).

## A4 — Commission across brokers (HIGH — already roughly correct)

| Broker | Effective bps RT @ $30 stock |
|---|---|
| IBKR Pro tiered | ~1.7 |
| IBKR Lite (PFOF) | ~1-2 |
| Schwab | ~1-3 |
| Fidelity | ~0.5-1.5 |
| Robinhood (worst PFOF) | ~3-8 |

Our $0.005/share + 0.5% cap is conservative IBKR Pro proxy. **Recommendation:** keep as default, raise cap to 1.0% to match IBKR's actual max. For PFOF retail scenario, set commission to $0 and bake +1.5 bps per fill into slippage instead. Don't double-count.

## A5 — Capacity at AUM tiers (Tradable ceiling: $1-5M)

Median candidate ADV $1.4M, 0.5%-of-ADV cap pins max position at $7,000. With 25 slots → **~$175k AUM floor of comfortable zone.**

```python
def capacity_curve(universe_adv_dist, weekly_picks=5, max_pct_adv=0.005,
                   expected_cagr_at_zero_aum=0.20):
    cagrs = []
    for aum in np.logspace(5, 8.5, 50):  # $100k to $316M
        pos_size = 0.04 * aum
        min_adv_required = pos_size / max_pct_adv
        tradeable_frac = (universe_adv_dist >= min_adv_required).mean()
        alpha_retention = tradeable_frac ** 0.7  # convex decay
        pick_drag = min(1.0, tradeable_frac * len(universe_adv_dist) / weekly_picks)
        cagrs.append((aum, tradeable_frac, expected_cagr_at_zero_aum * alpha_retention * pick_drag))
    return cagrs
```

| AUM | Tradeable universe | Expected CAGR |
|---|---|---|
| $100k | ~95% | ~19% |
| $500k | ~70% | ~16% |
| $2M | ~45% | ~11% |
| $10M | ~10% | ~5% |

## A6 — Halt / locked-down risk (MED severity, large left tail)

Per-trade fill-failure rate by cap:
- Large: ~0.1% | Mid: ~0.3% | Small: ~0.8% | Micro: ~1.5-2.0%

When halt straddles exit, resumption prints often 30-70% lower than halt price. Fat-tailed.

```python
def apply_halt_risk(trade, mcap_tier, rng):
    halt_rates = {"large":.001, "mid":.003, "small":.008, "micro":.018}
    if rng.random() >= halt_rates[mcap_tier]: return trade
    roll = rng.random()
    if roll < 0.70: trade["status"] = "missed_entry"; trade["pnl"] = 0
    elif roll < 0.95:  # halt-resumption gap
        slippage = rng.uniform(0.30, 0.70)
        trade["exit_price"] *= (1 - slippage)
        trade["status"] = "halt_resumption_loss"
    else: trade["status"] = "t1_suspended"; trade["pnl"] = -trade["entry_price"]*trade["shares"]
    return trade
```

**Watch for double-counting** with delisted-seed fix from problem 1. Gate behind single `terminal_loss_source` flag.

## A7 — Borrow availability (LOW — don't add shorting)

60% of small caps and 30% of microcaps are HTB. HTB borrow runs 10-40% annualized; crowded microcap shorts 75-150%+ (GME 2021 hit 100%+). Locate failures intraday at materially higher rates for microcap. Forced buy-ins after squeezes.

If we added 30% short sleeve: estimated **240-450 bps annual drag from borrow alone**, plus 1-3% from failed locates. **Stay long-only.** Address drawdowns through position sizing, sector caps, regime-triggered cash overlay.

## A8 — Live-trading $5k validation (HIGH — concrete next step)

Open IBKR Pro (NOT Lite, want SMART routing transparency). Deposit $5k. ~200 fills over 12 weeks at retail sizing. Standard error on mean shortfall ±10-15 bps.

**Expected result:** 30-80 bps round-trip. If 2-3× worse (150-250 bps), recalibrate sim's `slippage_bps_each_side` to empirical p50 and rerun. Add liquidity filter (min ADV, max spread%) to picker. If filtered version still doesn't clear hurdle, **kill the strategy before scaling capital.**

## A9 — Tax drag (HIGH — just compute post-tax)

40-day holds → 100% short-term cap gains → blended fed+state ~38-42% for $200k CA earner.

```python
def post_tax_cagr(blotter, federal=0.32, state=0.06, niit=0.038, start=100_000):
    rate = federal + state + niit  # ~40.6% blended STCG
    blotter['year'] = pd.to_datetime(blotter['exit_date']).dt.year
    equity = start; tax_paid = 0
    for year, g in blotter.groupby('year'):
        gross = g['pnl_dollars'].sum()
        tax = max(gross, 0) * rate
        equity += gross - tax; tax_paid += tax
    years = blotter['year'].nunique()
    return ((equity / start) ** (1/years) - 1, tax_paid)
```

A 6% pre-tax sim → 3.5-3.7% post-tax. **Below T-bills.** A 25% pre-tax → 15% post-tax. **IRA mitigates fully** ($7k/yr cap).

## A10 — Round-lot constraints (LOW-MED)

CAGR drag by AUM:
- $5k: 200-350 bps drag (target $200, 3 shares at $60 = 10% rounding error compounded)
- $25k: 50-90 bps drag (acceptable)
- $100k: 15-25 bps drag (negligible)

**Recommendation:** Add `min_position_usd = $100` floor in simulator.py. Document recommended minimum AUM = $25k in README. Surface "effective deployed capital" as backtest output.

## A11 — Fill probability modeling (MED-HIGH severity)

Replace 100% close-fill assumption with:
- MOO large: 0.99 fill | mid: 0.98 | micro: 0.95
- Marketable-limit (last_close × 1.005): 0.88 across — cheaper but adverse-selection-biased
- Move exits from same-day close to **next-day open**

```python
def slippage_bps(adv_usd, position_usd, order_type):
    return half_spread_bps(adv_usd) + temp_impact_bps(position_usd / adv_usd)
```

**Post-realism CAGR drag: 70-130 bps/yr**, concentrated in microcap. Sharpe drag ~0.05-0.10. Run with new module flagged off vs on, report delta as the "execution alpha" the old model claimed for free.
