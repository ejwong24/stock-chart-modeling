# RESEARCH — Multi-subagent deep-dive index

This directory contains the synthesized output of 60 parallel subagents that
audited 5 hard open problems for this project. Each problem was hit by 12
subagents from different angles, then synthesized into a recommended path.

## The 5 problems

| # | Problem | Headline finding |
|---|---------|------------------|
| 01 | [Survivorship-bias closure on a hobbyist budget](01_survivorship_bias.md) | SEC EDGAR Form 25 + Form 8-K Item 3.01 + self-detected terminal patterns + Wayback iShares = ~7,500–9,000 delisted tickers (≈85–90% of historical attrition) without paying for Polygon |
| 02 | [Encoder bake-off — what beats DINOv2 here](02_encoder_bakeoff.md) | Top picks: PatchTST (1D transformer on raw returns) + engineered-features LightGBM + small custom 1D CNN. Drop chart-rendering entirely. Lock 2025 OOS before bake-off. |
| 03 | [Lockbox protocol for a credible OOS headline](03_lockbox_protocol.md) | 4-piece minimum: 2025 lockbox + git pre-registration + block-bootstrap CIs + White's Reality Check / SPA. Replaces "100th percentile vs 100 random seeds" forever. |
| 04 | [Realistic transaction costs + capacity](04_costs_capacity.md) | Pre-cost CAGR ~25% → post-realistic-cost CAGR ~14% at $100k AUM, ~12% at $1M, ~3% at $10M. Tradable-AUM ceiling: ~$3–5M taxable, ~$10M tax-deferred. |
| 05 | [Path-dependent exits + relabeling](05_path_dependent_exits.md) | First-week: vol-scaled trailing stop in simulator (4 hrs, no relabeling). Six-month gold standard: triple-barrier labels + multi-class head. Estimated Sharpe lift: +0.25–0.40. |

## How to read each file

Each problem file has:
- **Synthesis** — the 12th subagent's recommended-path summary
- **Top recommended action** — the single most-impactful first step
- **Subagent details** — the 11 specialized critiques (one per angle)

## Total token spend

~60 subagents × ~15k tokens each ≈ 900k tokens of research compressed into
this directory. Pulled from /tmp/research/ and staged here for browsing
through the web UI at /research.
