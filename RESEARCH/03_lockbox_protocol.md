# Problem 3 — Lockbox protocol for a credible OOS headline

## Synthesis (top action)

**Minimum-viable 4-piece protocol (in priority order):**

1. **Frozen 2025 lockbox + git pre-registration** (8 dev-hours). Tag the commit, write a single locked Python script that takes hardcoded `(config_id, seed_set)` and runs against 2025 data the search never touched, commit the SHA-256 of the chosen config to git BEFORE fetching 2025 prices. **Highest gap-closed-per-hour of all 11 options.** Converts the entire claim from "data-mined backtest" to "out-of-sample test with registered hypothesis."

2. **Block bootstrap CIs on the lockbox result** (6 hrs). Stationary block bootstrap on daily equity returns, expected block length = horizon = 40d. 10,000 resamples for CAGR/Sharpe, 50,000 for max-DD. Without this the lockbox is just a point estimate.

3. **Combinatorial purged k-fold CV on the original 2018-2024 data** (12 hrs). Retroactively rescue effective-N from the existing 72-config search. 66 splits at N=12, k=2, 5-day embargo. Watch for non-stationarity bias — use as diagnostic, not headline.

4. **Forward paper-trading log (Alpaca, daily commit to public-readable git)** (4 hrs setup, then passive). Three months of forward results survives all methodological objections.

**Acceptance criteria for "real edge":**
- Lockbox 2025 excess-return CI strictly above zero AND
- CPCV median Sharpe rank in top quartile of 72 configs (not top-1 lucky) AND
- Forward paper after 60 days shows excess-return sign matching backtest within 1.5σ

Any 2 of 3 = "promising, keep paper-trading." All 3 = "deploy small real capital."

**Post-protocol headline format:**
> "Pre-registered 2025 OOS: X% excess vs SPY [95% CI: a%, b%]; CPCV rank N/72; forward paper +P% over Q days."

**Estimated post-protocol effective N_trials:** ~14 (1 lockbox + 12 CPCV folds + 1 forward).

---

## A1 — 2025 lockbox concretely (HIGH tractability, ~6 hrs)

```yaml
# config/default.yaml
splits:
  test_year_lockbox: 2025
  lockbox_audit_path: reports/lockbox.audit.jsonl
  lockbox_registry_path: reports/lockbox.registry.json
```

```python
# splits.py
def yearly_walk_forward(test_years, *, unlock_2025=False, lockbox=2025):
    years = [y for y in test_years if y < lockbox]
    if unlock_2025:
        years.append(lockbox)
        _audit("split_unlocked", {"year": lockbox})
    for y in years: yield _make_fold(y)
```

Pre-push hook blocks committing 2025-dated reports unless `lockbox-broken.json` exists. One-shot registry refuses second `claim()` per (H, T, model) tuple. **First 2025 number is the headline — good or bad, no iterating.**

## A2 — Pre-registration via git commit (HIGH tractability)

`docs/preregistration_<id>.md` with frozen sections: hypothesis, model spec (incl. seed), features (with feature_set_hash), label, dataset, splits, metric, pass threshold, alternatives considered. Commit to protected `main` branch, get SHA, then `oos_evaluate.py --prereg <sha>` validates that running config matches declared hash.

What can change after prereg: nothing. Seed change = new prereg. CI scans `docs/preregistration_*.md` touching overlapping OOS dates and tags as a family; family size = effective trial count for Bonferroni.

**How many preregs can a strategy survive?** N=1 unless you Bonferroni-correct.

## A3 — Combinatorial purged k-fold CV (MED tractability)

López de Prado CPCV: N=12 monthly groups, k=2 test, ~5d embargo. (12 choose 2) = 66 splits. ~5-8 min for LightGBM, ~30 min for DINOv2-LR. **Causality issue:** half of 66 folds train on data chronologically AFTER test → mean upward-biased for non-stationary series.

```python
def cpcv_splits(dates, n_groups=12, k_test=2, embargo_days=5):
    groups = np.array_split(np.argsort(dates), n_groups)
    for test_ids in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[i] for i in test_ids])
        lo, hi = dates.iloc[test_idx].agg(['min', 'max'])
        embargo = pd.Timedelta(days=embargo_days)
        purge_mask = (dates >= lo - embargo) & (dates <= hi + embargo)
        train_idx = np.setdiff1d(np.where(~purge_mask)[0], test_idx)
        yield train_idx, test_idx
```

**Use case: hyperparameter search inside training window. Keep yearly walk-forward as headline OOS metric.** If CPCV mean and walk-forward mean disagree substantially, that's a non-stationarity signal worth flagging.

## A4 — Continuous OOS rolling window (HIGH tractability)

Step = window = 4 weeks (non-overlapping). Train length = 3 years (configurable). Embargo = 1 week. ~120 windows over 2017-2026 = ~10 min for LightGBM, ~6 hr for DINOv2-LR (cache features per ticker-date so retrain only touches LR head, drops to ~30 min).

**Recommendation: complement, don't replace.** Yearly walk-forward = headline (literature comparability). Rolling-window = diagnostic that reveals "when did the edge die?"

## A5 — Effective sample size (HIGH severity gap closed)

López de Prado's `average uniqueness`. For each trade i, uniqueness_i = mean(1/concurrency over its hold days). Effective N = sum of uniqueness across trades.

For our 1,104 trades with ~25 concurrent positions: uniqueness_i ≈ 1/25 ≈ 0.04. **N_effective ≈ 44**, not 1,104.

```python
def effective_sample_size(blotter_df, hold_days=40):
    df = blotter_df.copy()
    if 'exit_date' not in df: df['exit_date'] = df['entry_date'] + pd.Timedelta(days=hold_days)
    days = pd.date_range(df.entry_date.min(), df.exit_date.max(), freq='B')
    concurrency = pd.Series(0, index=days, dtype=float)
    for _, r in df.iterrows():
        concurrency.loc[r.entry_date:r.exit_date] += 1.0
    uniqueness = np.empty(len(df))
    for i, r in enumerate(df.itertuples()):
        uniqueness[i] = (1.0 / concurrency.loc[r.entry_date:r.exit_date]).mean()
    return float(uniqueness.sum())
```

**Rule of thumb:** for 40d horizon + weekly 5-name entries, divide naive trade count by ~8 (block size) for upper bound, by ~25 (avg concurrency) for López de Prado lower bound. Realistic N_eff = 40-140. SE(Sharpe) widens by √(1104/44) ≈ **5×**. A naive-N "significant" Sharpe of 1.2 (t=4) collapses to t=0.8 — not significant.

## A6 — Block bootstrap CIs (HIGH severity)

Stationary bootstrap (Politis-Romano) with random block lengths Geometric(p=1/40), expected block length = 40d = holding horizon. Resample daily log-returns of EQUITY CURVE, never trade returns. 10k resamples for CAGR/Sharpe; 50k for max-DD (non-smooth statistic, slower convergence).

For model_CAGR − random_baseline_CAGR: bootstrap JOINT path (same date indices on both curves) so CI captures correlated luck. Hot-deck variant: subsample 80% of trade entry dates without replacement, then block-bootstrap.

**Reporting:** "CAGR = 61.1% (95% CI: 32% to 95%, n=10k stationary block, E[L]=40d)". If `Edge over random CI` crosses zero, headline is luck.

## A7 — Permutation test (HIGH severity)

Within each anchor week, shuffle forward labels across tickers. Refit, compute CAGR. 1000 reps → null distribution. p = (1 + #{null ≥ actual}) / (1 + N).

**Cheap routine variant:** shuffle predicted scores within week, no refit needed. Sub-second per permutation. Test "are scores informative" rather than "does fitting find signal."

**Headline variant:** refit each permutation. ~1.4 hr for LightGBM, ~8 hr for DINOv2-LR. Once per model version.

## A8 — White's Reality Check / Hansen's SPA (HIGH severity)

The canonical formal test for "best of N strategies." Use Hansen's SPA over White's RC because RC is conservative (a single terrible strategy inflates the bootstrap variance, pulling null right). With 72 configs, several are losers — SPA is correct.

```python
from arch.bootstrap import SPA
spa = SPA(benchmark_loss, losses, block_size=10, reps=10000,
          studentize=True, bootstrap='stationary', seed=42)
spa.compute()
print(f"SPA p (consistent): {spa.pvalues['consistent']:.4f}")
```

Report `pvalues['consistent']` as the headline (Hansen's recommended variant). Naïve t-test on best CAGR likely gives p < 0.001; SPA on the same series often returns p in [0.05, 0.30]. **This replaces the "100th percentile vs random seeds" framing permanently.**

## A9 — Trial registry (HIGH tractability)

`configs/trial_registry.jsonl`, append-only, one JSON object per line with `(ts, trial_id, sha_config, sha_data, sha_code, model, label, horizon_d, threshold_q, universe, headline_sharpe, prereg_id)`. Pre-run hook calls `log_trial()` BEFORE training; missing prereg_id auto-increments and warns.

CI rejects PRs that shorten the file or rewrite past trial_ids. Reporter computes `N_eff = len(set(unique_keys))` and prints both raw and deflated Sharpe in every report header.

**At 5 horizons × 5 thresholds × 3 models × 2 universes = 150 trials, deflated Sharpe hurdle rises ~3.5×.** Kills the "let me try one more horizon" temptation — that's the discipline a stock-picker needs.

## A10 — Honest report card template (HIGH tractability)

Mandatory fields in `reports/<tag>/honest_report_card.md`:
- BLUF (Y/N) — "Would I trade this with my own money?" + 2-sentence reasoning
- Pre-registration: hypothesis SHA, date, N_trials_in_registry
- Headline metrics with 95% CIs (CAGR, Sharpe, Deflated Sharpe, Reality Check p, max DD)
- Baseline gap (model − best simple baseline) with CI distinguishable from 0?
- Post-cost CAGR at 3 AUM tiers
- Hidden risks (mandatory; "none" not allowed): regime dependence, capacity ceiling, slippage fragility, snooping residual, survivorship audit

**Forbidden claims** (linter-rejected): "best of K" without N_trials_in_registry, "outperformed random" without DSR/Reality Check, "Sharpe > X" without CI. The original document's "100th percentile, 61.1% CAGR" claim FAILS the linter.

## A11 — Forward paper-trading (HIGH tractability)

GitHub Actions cron Friday 17:00 PT or system crontab on Oracle box. `forward_pick.py` freezes model trained ≤ today, generates Monday's top-5, writes `data/forward_picks/<date>.csv`. `settle_picks.py` looks up realized 40d returns 8 weeks later.

26 weeks × 5 picks = 130 fully out-of-sample trades. Standard error large but adequate for sign-direction. **Honest expectations:** if backtest shows 25% CAGR, expect forward 8-15% (typical 30-70% degradation). Forward ≈ backtest is a red flag for leakage. Forward < 0 means signal is dead.

Don't size up real capital until ≥12 months forward AND survived a regime change.
