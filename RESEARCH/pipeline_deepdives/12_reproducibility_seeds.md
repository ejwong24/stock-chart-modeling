# Reproducibility — every random seed and where it lives

## Why seeds matter

A stock-modeling pipeline that cannot reproduce its own output is not a pipeline — it is a story you tell once. Reproducibility serves three concrete needs:

1. **Debugging.** When a backtest produces a suspicious [Sharpe ratio](/story/09/27_sharpe_ratio) or an outlier trade, we need to rerun the exact same scenario, drop into the offending step, and inspect intermediate state. A pipeline that produces different blotters on every run gives us nothing to diff against.
2. **Auditing.** When the model picks a name and we act on it, we need to be able to reconstruct, months later, why that pick was made. That means the same config, the same data snapshot, and the same seeds must yield the same decision.
3. **Comparability.** When we change a feature or a hyperparameter, we want the diff in output to be attributable to the change, not to RNG drift. Two researchers running the same code on the same data with the same seeds should get **bit-identical** outputs.

## The failure mode

One unseeded RNG anywhere in the chain breaks the whole guarantee. The common offenders in Python ML code are:

- `sklearn.model_selection.train_test_split(...)` without `random_state`
- sklearn estimators (`LogisticRegression`, `PCA`, `RandomForestClassifier`, ...) without `random_state`
- `numpy.random.rand / randn / choice / shuffle` — these use the **global** numpy RNG, which is seeded from `/dev/urandom` at import time
- `random.random()` — same problem, global module-level state
- [LightGBM](/story/09/31_lightgbm_internals) / XGBoost without `random_state` or `seed`

Any one of those, anywhere in the call graph, and the determinism story collapses.

## Our practice

We thread a `seed` int through every random-bearing function and use it explicitly:

- Every sklearn estimator gets `random_state=seed`.
- Every `train_test_split` call gets `random_state=seed`.
- Every numpy random draw uses `np.random.default_rng(seed)` — a local `Generator` instance, never the global `np.random.*` module.
- The config file (`config/default.yaml`) is the single source of truth for the seed values.

## The full RNG inventory

| Location | Function / Site | Seed source | Default |
|---|---|---|---|
| `config/default.yaml` | `seeds.numpy` | config | 42 |
| `config/default.yaml` | `seeds.torch` | config | 42 |
| `config/default.yaml` | `seeds.sklearn_random_state` | config | 42 |
| `config/default.yaml` | `seeds.lightgbm_random_state` | config | 42 |
| `config/default.yaml` | `seeds.random_baseline_base_seed` | config | 1000 |
| `src/stock_chart/models.py` | `fit_lr_baseline` — PCA `random_state` | `seed` param | 42 |
| `src/stock_chart/models.py` | `fit_lr_baseline` — [LogisticRegression](/story/09/04_logistic_regression) `random_state` | `seed` param | 42 |
| `src/stock_chart/models.py` | `fit_lgbm` — [LightGBM](/story/09/31_lightgbm_internals) `random_state` | `seed` param | 42 |
| `src/stock_chart/models.py` | `fit_lgbm` — `train_test_split` `random_state` | `seed` param | 42 |
| `src/stock_chart/simulator.py` | `halt_risk_seed` (halt-risk RNG when enabled) | `seed` param | 42 |
| `src/stock_chart/random_baseline.py` | `run_random_seeds` — `np.random.default_rng(base_seed + i)` per seed | `base_seed` param | 1000 |
| `src/stock_chart/stats.py` | `stationary_block_bootstrap` | `seed` param | 42 |
| `src/stock_chart/stats.py` | `reality_check_spa` | `seed` param | 42 |
| `scripts/forward_pick.py` | Model fit | reuses config seed | 42 |
| `src/stock_chart/render.py` | Renderer | — | deterministic, no RNG |
| `src/stock_chart/embed_dinov2.py` | [DINOv2](/story/09/01_dinov2_architecture) inference | — | deterministic (eval mode, no dropout) |
| `src/stock_chart/labels.py` | Label generation | — | deterministic |
| `src/stock_chart/features.py` | Feature engineering | — | deterministic |
| `src/stock_chart/splits.py` | Time-series splits | — | deterministic (no shuffling) |

## The full chain for a typical run

`config/default.yaml` → `run_pipeline.py` loads `seeds.*` → seeds are passed explicitly to `fit_lgbm` / `fit_lr_baseline` → those pass them down to sklearn estimators and `train_test_split` → the simulator receives `halt_risk_seed` if halt risk is enabled → stats use the `seed` param for bootstrap and SPA reality checks. There is no point in the chain where a function reaches for a global RNG.

## [DINOv2](/story/09/01_dinov2_architecture) — deterministic, with a caveat

DINOv2 inference is deterministic in our setup because we run it under `torch.set_grad_enabled(False)`, in `.eval()` mode, with no dropout active. There is no torch seed needed during inference because no random operations occur.

**Caveat:** torch reductions can produce subtly different floats on CPU vs GPU due to nondeterministic reduction order. We sidestep this entirely by running CPU-only on ARM64, which **is** deterministic.

## What is NOT in the seed chain

Some inputs are inherently nondeterministic and are documented as not reproducibility-bearing:

- **`yfinance` HTTP downloads.** Server responses can vary (corrections, restatements, transient errors). We snapshot to disk so downstream runs are deterministic against the snapshot.
- **Git SHA.** Captured in audit logs as metadata, not as an RNG input.
- **Wall-clock timestamps** in audit logs. Cosmetic only.

## Verification tests

Three test files enforce the determinism contract:

- `tests/test_pipeline_integration.py::test_full_pipeline_determinism_same_seed` — runs the full chain twice with the same seeds and asserts bit-identical [blotter](/story/09/34_blotter_equity_summary) and equity DataFrames.
- `tests/test_reproducibility_audit.py` — covers every RNG primitive individually (models, baselines, stats, simulator).
- `tests/test_simulator_invariants.py::test_deterministic_with_same_inputs` — confirms the simulator is a pure function of its inputs.

## The reality check

The reproducibility audit ran **17 hardening passes**. One of them — `test_full_pipeline_determinism_same_seed` — confirms that running the full chain end-to-end twice, with the same seeds and the same data snapshot, produces bit-identical [blotter](/story/09/34_blotter_equity_summary) and equity outputs. That is the load-bearing guarantee.

---

> **Adding a new RNG safely.** If you introduce any new randomness, follow three rules:
> 1. **Thread a `seed` parameter through the function signature** — never reach for an ambient default.
> 2. **Use `np.random.default_rng(seed)`** for numpy work; never call `np.random.rand` / `np.random.choice` / `np.random.shuffle` directly. The global numpy RNG is forbidden in this codebase.
> 3. **Pass `random_state=seed`** to every sklearn estimator and every `train_test_split`. Add the new RNG to the inventory table above and to `tests/test_reproducibility_audit.py` in the same PR.
