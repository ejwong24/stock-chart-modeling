# The data-flow contract — anchor, feature, label, split alignment end to end

Every silent ML failure in this project lived in the same place: not in a model, not in a metric, but in a *seam* — the join between two stages that each looked correct in isolation. The [row-alignment landmine](/story/09/41_row_alignment) was one such failure. This page is its inverse: a positive statement of the contract every stage must honor, traced through a single `(ticker, anchor)` row from raw parquet to the simulator's exit.

It reduces to one idea: **a single row of data must mean the same thing at every stage, and the system must enforce that structurally rather than rely on it happening to be true.**

## The row's journey

```
adjusted/{ticker}.parquet  (sorted by date, deduped on date)
        │  POSITIONAL index: anchor_idx = row position in THIS ordering
        ▼
labels.label_one ─────────► fwd_ret_{H}d, ret_{H}d_ge_T, resolution_date_{H}d
        │  closes[anchor_idx+H]                            (key: ticker, anchor_date)
        ▼
features.compute_for_anchors ─► 40 features from closes[:idx+1]  +  SPY by DATE
        │                                                  (key: ticker, anchor_date)
        ▼
merge(features, labels) on (ticker, anchor_date)  ──► merged   [ROW IDENTITY KEY]
        ▼
render + DINOv2 embed ─► embs  (1:1 with anchor_kept via keep_idx)
        ▼
volume features + vmask ─► embs[vmask], merged_kept = anchor_kept.iloc[vmask]
        │                   ONE canonical frame, ONE mask, applied together
        ▼
splits.yearly_walk_forward(merged_kept) ─► {year: (tr_df, te_df)}
        │  purge/embargo driven by resolution_date_{H}d
        ▼
fold loop: tr_idx / te_idx = df.index.values  (POSITIONS into merged_kept)
        │  embs[tr_idx], vol[tr_idx], X_eng[tr_idx], y[tr_idx] all == row i
        ▼
scores parquet (carries resolution_date) ─► simulator: resolution_date = exit timing
```

## Stage by stage, with the invariant at each seam

**1. The anchor is a position, not a name.** In `labels.label_one`, the per-ticker frame is `sort_values("date").reset_index(drop=True)`, and `anchor_idx` is the *integer position* of the anchor's date inside that exact ordering (`anchors["date"].map(idx_map)`). It is meaningless except against that one canonical parquet ordering. Every later stage — features, volume, render — re-reads the same parquet, re-applies the same sort, and indexes by that integer. **Invariant: every stage uses the same parquet ordering.** This is why [atomic writes and date-dedup](/story/09/50_atomic_writes_integrity) matter: if a non-atomic write or a duplicate date ever changes the row order, every `anchor_idx` ever computed silently points at the wrong day. The ordering is the foundation the contract stands on.

**2. Labels resolve forward from the anchor position.** Given `anchor_idx`, the label is `closes[anchor_idx + H] / closes[anchor_idx] - 1`, binarized at threshold `T`, plus `resolution_date_{H}d = dates[anchor_idx + H]` (see [the label grid](/story/09/19_label_grid)). The forward look is the *only* place the future is touched, and it is deliberate. Non-positive anchor closes are dropped so `inf >= T` can never forge a false positive. **Invariant: the label for row `i` describes the future of row `i`'s anchor, and records when that future becomes known.**

**3. Features look strictly backward.** `features.compute_for_anchors` computes all 40 `FEATURE_COLS` from `closes[ti-251 : ti+1]` — the window *ending at and including* the anchor, never past it. No look-ahead is possible because no index greater than `ti` is ever read. The one cross-series input, SPY, is aligned **by date, not by position**: `spy_by_date.reindex(df["date"])` so `spy_closes[ti]` is SPY's close on the *same calendar date* as the ticker's `closes[ti]` (the subtlety behind [SPY beta](/story/09/38_spy_beta) and the rest of the [engineered features](/story/09/03_engineered_features)). Indexing a globally-sorted SPY array by this ticker's positional `anchor_idx` would compare mismatched dates for any name whose history diverges from SPY's. **Invariant: features see only `[:idx+1]`; cross-series joins are by date.**

**4. The merge defines row identity.** `feats.merge(labels_df, on=["ticker", "anchor_date"], how="inner")` is where features and labels become one row, joined on a *semantic* key, not a positional one. After this point, "row" means a `(ticker, anchor_date)` pair with its features and label welded together. **Invariant: `(ticker, anchor_date)` is the row identity key; everything downstream is keyed off `merged`.**

**5. One canonical frame, one mask, for all parallel arrays.** This is the seam where the [row-alignment landmine](/story/09/41_row_alignment) detonated. Rendering can drop anchors (insufficient lookback, missing window), so `_render_and_embed` returns `embs` *and* `anchor_kept = merged.iloc[keep_idx]` — `embs` is 1:1 with `anchor_kept` **by construction**, because the same `keep_idx` list built both. Volume features are then computed on `anchor_kept` (not `merged`), producing `vmask`, and the mask is applied to *both* arrays in the same breath: `embs = embs[vmask]`, `merged_kept = anchor_kept.iloc[vmask]`. The defeated bug used `merged.iloc[:len(embs)]` — a positional *prefix* — paired with a mask computed over the *full* `merged`; the moment any anchor was dropped, image features silently married the wrong labels. **Invariant: parallel arrays (`embs`, `vol_feats`, and the metadata frame) derive from one canonical frame and one mask, applied together — never reconstructed independently.**

**6. The split is driven by resolution date.** `splits.yearly_walk_forward` defines test folds as calendar years and builds the training pool from anchors *strictly before* the test year. Then it **purges**: any training row whose `resolution_date_{H}d >= test_start - embargo` is dropped, because its forward label would resolve into (or just before) the test window — exactly the year-boundary leakage the original document committed. The embargo is `H * embargo_horizon_multiplier` trading days (see [walk-forward + embargo](/story/09/08_walkforward_embargo)). The purge keys entirely off `resolution_date`, which is why that column has to be a real [trading-calendar](/story/09/37_trading_calendar) date and not a naive calendar offset. **Invariant: no training label resolves inside the test fold, enforced by `resolution_date`.**

**7. Fold indices are positions into the parallel arrays.** Inside the fold loop, `tr_idx = tr_df.index.values` and `te_idx = te_df.index.values`. Because `merged_kept` was built with `reset_index(drop=True)`, those index values are *positions* — and `embs[tr_idx]`, `vol_feats[tr_idx]`, `X_eng[tr_idx]`, and `y_all[tr_idx]` all select the same physical row `i`. This holds only because step 5 guaranteed row `i` of every array is the same `(ticker, anchor_date)`. **Invariant: row `i` of `embs`, `vol`, `X_eng`, and `y` all correspond to row `i` of `merged_kept`.** The split returns *views* of that frame; it never re-orders or re-keys it.

**8. Scores carry their own clock into the simulator.** Each test-score frame copies `resolution_date_{H}d` and renames it `resolution_date`. The [simulator loop](/story/09/06_simulator_loop) uses that column — not a re-derived offset — to decide when each position exits. The "when is this known" date from step 2 flows untouched all the way to the trade exit.

## The three load-bearing invariants

1. **One parquet ordering everywhere.** `anchor_idx` is a position; every stage must re-derive it from the identical sorted, deduped frame. Guaranteed upstream by atomic writes and date-dedup.
2. **One canonical frame and one mask for parallel arrays.** `embs`, `vol_feats`, and the metadata frame are built from a single `keep_idx`/`vmask` and filtered together — never reconstructed by independent prefixes or separately-computed masks.
3. **`resolution_date` is the single source of truth for "when is this known."** It drives the purge/embargo in splitting and the exit timing in simulation; nothing re-derives it.

## Why this is the whole game

Most silent ML bugs do not live inside models — they live at the seams between stages, where two arrays that are *supposed* to line up only line up *by coincidence*. A prefix slice, a positional index into the wrong frame, a date offset that ignores holidays: each is locally plausible, globally catastrophic, and none of them throws. The fix is to make alignment **structural rather than coincidental** — one ordering, one frame, one mask, one resolution date — so that "row `i` means the same thing everywhere" is a property of how the data is *constructed*, not a fact you must remember to check. That is why the [testing philosophy](/story/09/45_testing_philosophy) asserts these invariants directly (`assert_no_leakage`), why the [hardening story](/story/09/39_hardening_story) was largely a campaign against these seams, and why [the verdict](/story/09/55_the_verdict) only counts once the contract holds end to end.
