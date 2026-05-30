# Market beta — what beta_spy_63d measures and why BNTX's was 0.00

> **⚠️ Correction (post-audit).** The "BNTX `beta_spy_63d` = 0.00" figure below was a
> **bug artifact, not a measurement.** An off-by-one made the beta length-guard never
> pass, so the feature returned `0.0` for *every* stock on *every* anchor; SPY was also
> aligned by row position instead of by date. The economic narrative ("BNTX traded on
> vaccine catalysts → β≈0") was pattern-matching on a constant. Both bugs are now fixed
> (64-close slice + date alignment). See **[why beta_spy_63d was silently zero](/story/09/42_beta_zero_bug)**
> for the post-mortem. The CAPM explanation below is still correct as *theory*; treat the
> specific 0.00 value as illustrative only.

## The CAPM definition

In the Capital Asset Pricing Model, beta is the slope coefficient of a stock's returns regressed against the market's returns:

```
β = cov(r_stock, r_market) / var(r_market)
```

In our [engineered features](/story/09/03_engineered_features), the market is approximated by SPY (the S&P 500 ETF). A beta of 1 means the stock moves 1-for-1 with the market on average. A beta of 0 means the stock's daily returns are uncorrelated with SPY's. That doesn't mean the stock has no risk — it means the risk lives somewhere other than broad market exposure.

## Reading the beta scale

- **β > 1** — aggressive / cyclical. Tech, semis, high-beta growth.
- **β ≈ 1** — market-tracking. Large blend ETFs.
- **0 < β < 1** — defensive. Utilities, staples, regulated telecom.
- **β ≈ 0** — uncorrelated. Single-stock catalyst-driven names: biotech in EUA windows, M&A targets, distressed names.
- **β < 0** — contrarian. Gold miners during equity crashes, inverse ETFs.

## Why a 63-day window

We compute beta over a rolling 63-day window — roughly one trading quarter. Longer windows (252-day) smooth out short-term regime shifts; shorter (20-day) are noisier. 63 is the practitioner default for "current beta."

## The computation

```python
def beta_63d(stock_returns, spy_returns):
    cov_matrix = np.cov(stock_returns[-63:], spy_returns[-63:])
    return cov_matrix[0, 1] / cov_matrix[1, 1]
```

## The BNTX 0.00 story

For BNTX on 2021-07-02, the rolling 63-day window covers April through June 2021. In that quarter, BNTX wasn't trading on macro news. It was trading on:

- Strong overseas vaccine demand
- EUA and full-approval milestones
- Delta variant headlines

None of these catalysts had anything to do with SPY's broader behavior — which during the same window was grinding higher on reopening trades and Fed expectations. BNTX's daily moves were essentially orthogonal to SPY's. The regression slope came out at **0.00**.

## Beta = 0 does not mean "no risk"

This is the trap. BNTX had `vol_252d = 76%` — extremely volatile. It just wasn't volatile **in a market-correlated way**. CAPM's beta measures the systematic (market) risk component only. The idiosyncratic risk — single-stock, single-catalyst, news-driven — is invisible to beta.

## Implications for portfolio construction

In a market-hedged portfolio, beta-0 names are the cleanest source of "pure alpha." Their returns aren't driven by the market. The model implicitly favors low-beta-but-high-momentum names — biotech in catalyst windows, EVs in commodity runs, energy in supply-shock regimes. See the [capacity ceiling](/story/09/26_capacity_ceiling) for why this preference doesn't scale linearly.

## Known limitations

CAPM beta assumes returns are jointly normal (they're not), the market portfolio is observable (SPY excludes small-caps, international, bonds), and beta is constant within the 63-day window (definitely not for catalyst-driven names). None of these hold perfectly. Beta is a useful **feature** for the model, not an absolute measure of risk.

## Interaction with the baselines

None of the five [simple baselines](/story/09/20_simple_baselines) explicitly uses beta. Neither does the [MA250 prefilter](/story/09/07_ma250_prefilter). So the model can hypothetically gain edge by preferring low-beta high-momentum names. Empirically this works in some folds, doesn't in others.

## The full BNTX 2021-07-02 anchor view

- `ret_252d = +123%` — huge annual gain
- `vol_252d = 76%` — high realized volatility
- `beta_spy_63d = 0.00` — zero market correlation in the last quarter

The combination of high momentum, high vol, and zero beta is a classic pure-alpha signature — exactly what a momentum-on-extension strategy hunts for.
