# The config surface — every knob and its default

This is the complete tunable inventory for the corrected pipeline. Defaults below are read directly from `config/default.yaml`, `config/realistic.yaml`, and the `SimConfig` dataclass in `src/stock_chart/simulator.py`. Where the narrative disagreed with the repo, the repo value is used and flagged.

Two things to know up front. First, there are **two profiles**: `default.yaml` (clean baseline) and `realistic.yaml` (adds path-dependent exits, Almgren-Chriss impact, and halt risk — see [the hardening story](/story/09/39_hardening_story)). Second, the bare `SimConfig` *dataclass* defaults are deliberately conservative — the expensive realism knobs default to **off**, and only `realistic.yaml` turns them on. So "the default" can mean two things; both are listed.

## Prefilter

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `ma_window` | 250 | Lookback for the moving average trend gate | [MA250 prefilter](/story/09/07_ma250_prefilter) |
| `ma_extension` | 1.5 | Max price/MA ratio before a chart is excluded | [MA250 prefilter](/story/09/07_ma250_prefilter) |
| `enabled` | true | Master switch for the prefilter stage | — |

**Sensitivity:** `ma_extension` is the high-leverage knob — it sets how much of the universe survives to labeling, shaping base rates. `ma_window` is structurally load-bearing but rarely retuned.

## Labels

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `horizons` | [5, 10, 15, 20, 30, 40] | Forward trading-day windows | [label grid](/story/09/19_label_grid) |
| `thresholds` | [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] | Return cutoffs defining a positive label | [label grid](/story/09/19_label_grid) |
| `warmup_days` | 252 | Min history before a chart is eligible | [label grid](/story/09/19_label_grid) |
| `anchor_weekly` | true | Anchor weekly rather than daily | [weekly anchors](/story/09/18_weekly_anchors) |

**Sensitivity:** the `horizons × thresholds` cross-product (36 cells) defines the entire prediction problem. `anchor_weekly` matters a lot for sample independence (daily anchoring leaks).

## Render

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `lookback_days` | 252 | Trading days drawn per chart | [fixed log-y axis](/story/09/29_fixed_log_y_axis) |
| `image_size` | 224 | Output side length (px), matches DINOv2 input | [fixed log-y axis](/story/09/29_fixed_log_y_axis) |
| `log_y_min` | 0.1 | Lower bound of the fixed log-price y-axis | [fixed log-y axis](/story/09/29_fixed_log_y_axis) |
| `log_y_max` | 11.0 | Upper bound of the fixed log-price y-axis | [fixed log-y axis](/story/09/29_fixed_log_y_axis) |
| `line_width` | 2 | Plotted line thickness (px) | — |

**Sensitivity:** `log_y_min/max` are critical — a *fixed* axis is what makes charts comparable. `image_size` is fixed by the backbone; `line_width` is cosmetic.

## Embed

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `model` | dinov2_vits14 | Vision backbone | [DINOv2](/story/09/01_dinov2_architecture) |
| `batch_size` | 16 | Images per forward pass | [DINOv2](/story/09/01_dinov2_architecture) |
| `emb_dim` | 384 | Embedding dim (fixed by ViT-S/14) | [DINOv2](/story/09/01_dinov2_architecture) |

**Sensitivity:** all cosmetic/throughput-only.

## Models

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `pca_dim` | 64 | Dims kept after PCA before the heads | [logistic regression](/story/09/04_logistic_regression) |
| `logreg.C` | 1.0 | Inverse L2 regularization | [logistic regression](/story/09/04_logistic_regression) |
| `logreg.class_weight` | balanced | Reweights for label imbalance | [logistic regression](/story/09/04_logistic_regression) |
| `logreg.max_iter` | 1000 | Solver iteration cap | [logistic regression](/story/09/04_logistic_regression) |
| `lgbm.n_estimators` | 400 | Boosting rounds | [LightGBM](/story/09/31_lightgbm_internals) |
| `lgbm.learning_rate` | 0.05 | Shrinkage per round | [LightGBM](/story/09/31_lightgbm_internals) |
| `lgbm.num_leaves` | 63 | Tree complexity | [LightGBM](/story/09/31_lightgbm_internals) |
| `lgbm.min_child_samples` | 200 | Min samples per leaf (overfit brake) | [LightGBM](/story/09/31_lightgbm_internals) |
| `lgbm.reg_alpha` | 0.1 | L1 penalty | [LightGBM](/story/09/31_lightgbm_internals) |
| `lgbm.reg_lambda` | 0.1 | L2 penalty | [LightGBM](/story/09/31_lightgbm_internals) |

**Sensitivity:** `min_child_samples` (200 is aggressive) and `num_leaves` are the dominant overfit/underfit dials; `class_weight: balanced` materially shifts the logreg boundary given rare positives. `pca_dim` caps how much embedding signal reaches the heads — see [image-track post-mortem](/story/09/59_image_track_postmortem).

## Splits

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `test_years` | [2019…2024] | Walk-forward test folds | [walk-forward](/story/09/08_walkforward_embargo) |
| `test_year_lockbox` | 2025 | Held-out lockbox year (touched once) | [walk-forward](/story/09/08_walkforward_embargo) |
| `embargo_horizon_multiplier` | 1 | Embargo gap = multiplier × max horizon | [walk-forward](/story/09/08_walkforward_embargo) |
| `min_train_examples` | 1000 | Min rows to train a fold | — |

**Sensitivity:** `embargo_horizon_multiplier` is integrity-critical — too low and label windows leak across the boundary.

## Simulator (`SimConfig`)

| Knob | Default (dataclass / realistic.yaml) | Controls | Deep-dive |
|---|---|---|---|
| `start_equity` | 100,000 | Starting capital | [allocation budget](/story/09/36_allocation_budget) |
| `max_position_pct` | 0.04 | Equity cap per position | [position size](/story/09/32_position_size_formula) |
| `max_new_per_week` | 5 | New-entry budget per week | [allocation budget](/story/09/36_allocation_budget) |
| `max_adv_pct` | **0.005** | [ADV20](/story/09/33_adv20_metric) cap per position | [position size](/story/09/32_position_size_formula) |
| `min_adv_usd` | 5,000,000 | Liquidity floor to be tradable | [position size](/story/09/32_position_size_formula) |
| `slippage_bps_each_side` | 10 | Flat slippage (fallback if AC off) | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `commission_per_share` | 0.005 | Per-share fee (capped at 0.5% of value) | — |
| `trailing_stop_pct` | **0.0 / 0.18** | Exit at drop from peak (0 = disabled) | [trailing stop](/story/09/11_trailing_stop_interactions) |
| `use_almgren_chriss_impact` | **false / true** | Flat slippage → square-root impact | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `impact_eta` | 0.142 | AC temporary-impact coefficient | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `impact_beta` | 0.5 | AC participation exponent (square-root) | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `permanent_frac` | 0.5 | Permanent share of impact | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `daily_vol_default` | 0.03 | Vol used when none supplied per-trade | [Almgren-Chriss](/story/09/09_almgren_chriss) |
| `halt_risk_enabled` | **false / true** | Simulate fill-failure on halts | [frictions](/story/09/25_frictions_beyond_impact) |
| `halt_risk_seed` | 42 | RNG seed for halt draws | [reproducibility seeds](/story/09/12_reproducibility_seeds) |

> **Correction:** the narrative listed `max_adv_pct = 0.02`. The repo uses **`max_adv_pct = 0.005`** (0.5% of ADV) in both YAMLs and the dataclass; `min_adv_usd` is confirmed at $5M. The **dataclass** defaults `trailing_stop_pct`, `use_almgren_chriss_impact`, and `halt_risk_enabled` to off — only `realistic.yaml` enables them.

**Sensitivity:** `max_adv_pct` and `min_adv_usd` are the most outcome-defining knobs — they decide whether microcap fantasy fills are allowed (see [the verdict](/story/09/55_the_verdict) and [cost stack](/story/09/56_cost_stack)). `use_almgren_chriss_impact` flips cost realism on; `max_position_pct` × `max_new_per_week` jointly set the deployment schedule.

## Seeds

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `numpy` | 42 | NumPy RNG | [reproducibility seeds](/story/09/12_reproducibility_seeds) |
| `torch` | 42 | DINOv2 / Torch RNG | [reproducibility seeds](/story/09/12_reproducibility_seeds) |
| `sklearn_random_state` | 42 | PCA / logreg determinism | [reproducibility seeds](/story/09/12_reproducibility_seeds) |
| `lightgbm_random_state` | 42 | LightGBM determinism | [reproducibility seeds](/story/09/12_reproducibility_seeds) |
| `random_baseline_base_seed` | 1000 | Base seed for the random-portfolio null | [reproducibility seeds](/story/09/12_reproducibility_seeds) |

**Sensitivity:** all cosmetic for *results*, critical for *reproducibility*. Changing a seed shouldn't move conclusions; if it does, that's a finding, not a tuning win.

## Stats

| Knob | Default | Controls | Deep-dive |
|---|---|---|---|
| `n_resamples` | 10,000 | Block-bootstrap resamples | [block bootstrap](/story/09/22_block_bootstrap_params) |
| `block_size` | 40 (= horizon) | Bootstrap block length | [block bootstrap](/story/09/22_block_bootstrap_params) |

**Sensitivity:** `block_size` is the one that matters — too small and overlapping label windows inflate significance. `n_resamples` is precision-only past ~10k. The trial count feeding the [Deflated Sharpe](/story/09/10_deflated_sharpe) is load-bearing for [the verdict](/story/09/55_the_verdict).
