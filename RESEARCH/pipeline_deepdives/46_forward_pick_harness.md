# The forward paper-trading harness — generating live picks without look-ahead

Every backtest is a story about the past told with the past's own data. The forward paper-trading harness is the one leg of validation that refuses that comfort: it commits a model to a prediction *before* the outcome exists. `scripts/forward_pick.py` is the script that does it — and until recently, it didn't. It ran every Friday, printed reassuring log lines, and wrote nothing. The bug and its fix turn out to be a clean lesson in why labeling and picking are not the same operation.

## The asymmetry between labeling and picking

A supervised label needs its answer. For the 40-day-forward target, anchor `t` is only labelable once `Close[t+40]` exists — so `label_one` trims the trailing tail of every ticker (`eligible_idx = arange(warmup, n - H_max - 1)`). The newest *labelable* anchor is always ~40 trading days in the past, because anything fresher has no resolved forward window. That is correct, and necessary, for training. See [the 36-label grid](/story/09/19_label_grid) and [weekly anchors](/story/09/18_weekly_anchors).

Picking wants the exact opposite. The bar we would actually buy is the *freshest* anchor — the one whose forward window has explicitly **not** resolved yet, because that's the future we're betting on. The original harness reused `labels.build_all` to source its candidate anchors, then filtered for `resolution_date >= today`. Since every label's resolution date is by construction in the past, that filter returned zero rows. Forever. The category error was treating the labeling anchor set as the universe of *pickable* anchors. They are disjoint at exactly the bar that matters.

The fix is a dedicated `latest_candidates()` that builds tradable anchors directly, independent of label trimming:

```python
elig = np.flatnonzero(df["date"].to_numpy() <= np.datetime64(today))
ti = int(elig[-1])                       # latest bar at/before today
if ti < warmup - 1 or ti < ma_window - 1:
    continue                             # not enough history for features/MA
ma = closes[ti - ma_window + 1: ti + 1].mean()
if prefilter and (closes[ti] / ma) <= ma_ext:
    continue                             # MA250 prefilter, same as the sim
```

Crucially `anchor_idx = ti` is the position in the **full sorted parquet**, matching how `features.compute_for_anchors` indexes the price series — so the same bar that's selected is the bar that's featurized (the alignment discipline of [the row-alignment landmine](/story/09/41_row_alignment)).

## Three validation modes, and the missing leg

The project runs three [validation modes](/story/09/21_validation_modes). Walk-forward (purged, embargoed) and lockbox both score on *historical* data — held-out, but already lived. Only forward paper-trading is genuinely out-of-sample in the temporal sense: it is exposed to regime changes that didn't exist when the model was frozen. A backtest can't surprise you with a new macro regime; a live week can. When the harness silently emitted nothing, this entire leg was absent — the project's strongest anti-overfitting argument was structurally unavailable, and the failure was invisible because "no picks" and "ran fine" looked identical in the logs.

## No-look-ahead discipline in live scoring

The harness enforces three temporal guarantees. First, training uses only resolved rows: `train = merged[merged[res_col] < today]`. A row whose 40-day window straddles `today` is excluded, because its label would peek past the freeze point. Second, scoring targets the single latest *unresolved* anchor per ticker — the live bar. Third, features for both training and scoring use only data at or before each anchor; the [engineered features](/story/09/03_engineered_features) are causal by construction, and anchor selection respects the [trading calendar](/story/09/37_trading_calendar) (latest *trading* bar at/before today, not a naive calendar date). Train on the resolved past, score the unresolved present, never let the present's features reach back into data that doesn't exist yet.

## The freshness guard

There is a quiet failure mode that mimics success: stale data. If the price store stops updating, `latest_candidates` still happily finds "the latest bar" — it's just weeks old, and the model scores ancient prices as if they were live. Without a guard, stale data is indistinguishable from a healthy week of picks. So:

```python
newest_bar = pd.to_datetime(cand["last_bar_date"]).max()
stale_days = (today - newest_bar).days
if stale_days > 7:
    print(f"  WARNING: newest data bar is {newest_bar.date()} "
          f"({stale_days} days stale vs {today.date()})")
```

Seven days covers a long weekend or a holiday-shortened week; beyond that, the [data acquisition](/story/09/13_data_acquisition) pipeline has stalled and the operator needs to know before trusting the picks.

## What's still missing (honestly)

The picks are generated; they are not yet *graded*. The docstring promises a companion `settle_picks.py` that looks up each pick's realized return 40 trading days later and writes `data/forward_realized/`. That script does not exist. The loop is open: we emit five tickers a week and never circle back to score them against what actually happened. The "after 26 weeks you have 130 fully out-of-sample trades" claim in the docstring is, today, aspirational — the harness produces the inputs to that result but not the result itself. This is the #1 item on the [feature gaps roadmap](/story/09/47_roadmap): a forward test you don't settle is just a forward *log*.

## The regression test

The fix is pinned by `test_forward_pick_latest_candidates_emits_picks`. It builds two synthetic 300-bar series — `UP` (exponential trend, close ≫ 1.5× MA250) and `FLAT` (constant) — and asserts that `latest_candidates`:

- returns the **latest** bar (`anchor_idx == n - 1`, `anchor_close == up[-1]`),
- respects warmup and the [MA250 prefilter](/story/09/07_ma250_prefilter) (only `UP` passes; `FLAT`'s ratio of 1.0 is filtered),
- and yields both tickers once the prefilter is disabled.

The buggy version returned zero rows under any configuration. The test now fails loudly if anyone reattaches picking to the labeling anchor set. Liquidity (the ADV gate and [ADV20](/story/09/33_adv20_metric)) and [position size](/story/09/32_position_size_formula) mirror the simulator so paper picks reflect tradable reality — part of the broader [hardening story](/story/09/39_hardening_story).
