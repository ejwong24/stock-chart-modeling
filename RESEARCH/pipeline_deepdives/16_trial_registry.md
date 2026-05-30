# The trial registry — every config tried, counted once, audited forever

The single most damning critique of the original stock-selection document was four words long: *best of seventy-two*. The author had quietly screened seventy-two configurations and reported only the winner's [Sharpe ratio](/story/09/27_sharpe_ratio). No deflation, no honest accounting. The trial registry exists so that mistake cannot be repeated by accident.

## Why the registry exists

The [deflated Sharpe ratio](/story/09/10_deflated_sharpe) (Bailey & López de Prado, 2014) requires one input that almost nobody honestly reports: `N_trials`, the number of distinct configurations actually evaluated before the headline number was chosen. Without it, DSR cannot correct for selection bias. The trial registry at `configs/trial_registry.jsonl` is the source of truth for that N.

## The JSONL format

Each evaluated configuration produces exactly one row:

```json
{"ts": "2025-09-12T03:14:15Z", "git_sha": "abc1234", "model": "lgbm_engineered",
 "label": "ret_40d_ge_25pct", "horizon_d": 40, "threshold_q": 0.25,
 "universe": "us_common", "headline_sharpe": 0.510, "n_obs": 2014}
```

Three roles:

- **Cosmetic / provenance**: `ts`, `git_sha`. Never used for dedup.
- **The dedup tuple**: `model`, `label`, `horizon_d`, `threshold_q`, `universe`. These five fields identify a trial.
- **Results**: `headline_sharpe`, `n_obs`. Recorded for later analysis.

## The dedup rule

`trial_count(p)` opens the JSONL, parses each line, and builds a Python `set` of `(model, label, horizon_d, threshold_q, universe)` 5-tuples. Re-running the same configuration one hundred times produces one hundred rows but still counts as **one** trial.

## The append-only invariant

The file is never edited and never truncated. The design choice is deliberate: every configuration you have ever evaluated should count toward your selection bias, because you **did** try it.

## Concurrent-write semantics

Pipeline runs are parallelized with `joblib` and can append from multiple processes simultaneously. We rely on POSIX `open(path, "a")` being atomic for writes smaller than `PIPE_BUF` — about 4 KB on Linux. A single JSONL row is comfortably under that. `tests/test_lockbox.py` includes a four-thread concurrent-write test that confirms every row arrives whole and parseable.

## The audit log

A second file at `configs/audit.jsonl` plays a complementary role. It records every significant event — [data acquisition](/story/09/13_data_acquisition), model fit, fold evaluation — with `ts` and `git_sha`. The audit log answers *what did the pipeline do, and when?* The trial registry answers *how many configurations have ever been evaluated?*

## The lockbox claim

Adjacent to the registry, `claim_lockbox(p, horizon, threshold, model)` is a one-shot mechanism. The first call writes a claim file. The second call raises `LockboxError` unless explicitly overridden. This is the enforcement primitive behind "the 2025 fold is locked."

## Verification tests

- `test_trial_count_then_count`: two distinct configs logged, `trial_count` returns `2`.
- `test_trial_count_skips_blank_lines_and_bad_json`: malformed lines are silently skipped, not crashed on.
- `test_audit_concurrent_threads`: eight threads writing simultaneously, every resulting JSON line valid.

## Adding a new model/label combo

```python
from stock_chart.lockbox import log_trial

log_trial(
    trial_path,
    config={"model": "lgbm_engineered", "label": label_col,
            "horizon_d": h, "threshold_q": t, "universe": "us_common"},
    headline_sharpe=fold_sharpe,
    n_obs=n_test,
)
```

Call it **once per `(label, threshold, model)` tuple**, not once per fold.

> For the broader story see [/research/03_lockbox_protocol](/research/03_lockbox_protocol).
