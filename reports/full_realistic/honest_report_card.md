# Honest Report Card — full_realistic

**BLUF — Would I trade this with my own money?** TODO

Reasoning (2 sentences max):
> TODO

## Pre-registration

- Hypothesis hash (SHA-256 of prereg doc): `(no prereg)`
- Pre-registration date: auto-generated (no prereg)
- N_trials_in_registry (this family): **1**
- Multiple-comparison correction applied: Deflated Sharpe (N_trials=1)

## Headline metrics (OOS only unless flagged)

| Metric | Point | 95% CI (block bootstrap) | Notes |
|---|---|---|---|
| CAGR | +9.79% | [-7.7%, +33.5%] | OOS, daily-return resample, block=horizon |
| Sharpe (annualized) | +0.36 | [-0.27, +0.60] | raw, before deflation |
| Deflated Sharpe Ratio | +0.356 | p=0.108 | adjusts for N_trials=1 |
| Reality Check (Hansen SPA) | p=(run reality_check_spa to fill) | bootstrap 5000 reps | recommended over White's RC |
| Max drawdown | -58.09% | [-78.2%, -35.0%] | non-smooth, slower CI convergence |

## Baseline gap (the only comparison that counts)

- Best simple baseline: **rank_inv_60d_vol (+7.26%)**
- Model CAGR − baseline CAGR: +2.53%, 95% CI: (needs paired bootstrap; see stats.py)
- Gap statistically distinguishable from 0 after N_trials adjustment? **N (gap ≤ 5pp)**

## Post-cost CAGR at 3 AUM tiers

| AUM | Slippage model | Post-cost CAGR | Notes |
|---|---|---|---|
| $100k | TODO | TODO | retail account |
| $1M | TODO | TODO | meaningful capacity |
| $10M | TODO | TODO | near capacity ceiling |

## Hidden risks (mandatory; "none" not allowed)

- **Regime dependence:** TODO
- **Capacity ceiling:** TODO
- **Slippage assumption fragility:** TODO
- **Data-snooping residual:** TODO
- **Survivorship/look-ahead audit:** TODO

## Effective sample size

- Naive trade count: ?
- López de Prado average uniqueness (effective N): 84
- SE(Sharpe) widening factor: 0.11x
