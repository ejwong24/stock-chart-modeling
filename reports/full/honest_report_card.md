# Honest Report Card — full

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
| CAGR | +17.27% | [-11.0%, +59.3%] | OOS, daily-return resample, block=horizon |
| Sharpe (annualized) | +0.51 | [-0.33, +0.90] | raw, before deflation |
| Deflated Sharpe Ratio | +0.510 | p=0.226 | adjusts for N_trials=1 |
| Reality Check (Hansen SPA) | p=(run reality_check_spa to fill) | bootstrap 5000 reps | recommended over White's RC |
| Max drawdown | -77.28% | [-87.9%, -40.8%] | non-smooth, slower CI convergence |

## Baseline gap (the only comparison that counts)

- Best simple baseline: **rank_inv_60d_vol (+12.01%)**
- Model CAGR − baseline CAGR: +5.26%, 95% CI: (needs paired bootstrap; see stats.py)
- Gap statistically distinguishable from 0 after N_trials adjustment? **Y**

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
- López de Prado average uniqueness (effective N): 45
- SE(Sharpe) widening factor: 0.15x
