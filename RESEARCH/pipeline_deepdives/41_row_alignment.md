# The row-alignment landmine — keeping arrays and DataFrames in lockstep

## The shape of the bug

The `stock_chart_modeling` pipeline carries several **parallel** data structures that only mean anything when they stay row-aligned. After the render step and the [DINOv2](/story/09/01_dinov2_architecture) encoder produce embeddings, `scripts/run_pipeline.py` holds:

- `embs` — a numpy array of DINOv2 embeddings (later [PCA](/story/09/02_pca_math)-reduced),
- `vol_feats` — numpy volume features,
- `X_eng` — [engineered features](/story/09/03_engineered_features),
- `y_all` — labels,
- and the metadata DataFrame (ticker, date, fold assignment).

The contract is brutally simple and entirely implicit: **row `i` of every array must be the same observation as row `i` of the frame.** The [walk-forward](/story/09/08_walkforward_embargo) fold loop leans on this directly — it slices the numpy arrays with `tr_df.index.values`, treating the frame's index as *positions* into the arrays. If the arrays and the frame ever describe different rows, nothing throws. You just train image features against the wrong labels and staple predictions onto the wrong `(ticker, date)`.

## The old code

```python
vol_feats, keep_mask = _volume_features_for_anchors(merged, price_lookup, ...)
embs = embs[keep_mask[:len(embs)]] if keep_mask.sum() != len(embs) else embs
merged_kept = merged.iloc[:len(embs)].reset_index(drop=True)   # <-- positional PREFIX
```

There are two independent errors here.

**Error 1 — a mask from frame A applied to array B.** `keep_mask` is computed over `merged` inside `_volume_features_for_anchors`. But `embs` was not built from `merged` — it was built from the render step's `keep_idx`, a *different* subset of anchors. Indexing `embs[keep_mask[...]]` lines up two boolean vectors that index two different universes. The lengths can coincide; the *meaning* does not.

**Error 2 — a positional prefix standing in for a boolean subset.** `merged.iloc[:len(embs)]` takes the first `K` rows of `merged` by *position*. But the rows that survived `keep_mask` are not necessarily the first `K` — a dropped row in the middle shifts everything after it. "The first `K` rows" and "the `K` rows where the mask is True" are the same set only when every drop happens to fall at the tail, which is to say, almost never on purpose. This line is **wrong by construction** even if `keep_mask` were applied to the right array.

## Why it was latent — and why latent is not safe

In the full production run, the render filter and the volume filter happen to apply *identical* guards: same DataFrame, same time index `ti`, same lookback window. They drop exactly the same rows — which today is zero rows. So `keep_mask` is all-`True`, `keep_mask.sum() == len(embs)`, the ternary skips the bad slice entirely, and `merged.iloc[:len(embs)]` equals `merged` *by accident*. Every test passes. Every score looks plausible.

That coincidence is load-bearing, and it is one diverging filter away from collapse. The instant the two filters disagree — a single NaN in a volume series, one render failure, one anchor with `ti < 251` that the lookback rejects on one path but not the other — `keep_mask` develops a `False`, the positional prefix grabs the wrong rows, and the pipeline silently trains on misaligned labels. No exception. No NaN in the output. Just a quietly corrupted result that looks exactly like a correct one.

This is the part worth internalizing: **latent ≠ safe.** A bug that only manifests under a condition you haven't hit yet is not a bug you've avoided; it's a bug you've armed. And `merged.iloc[:len(embs)]` is wrong-by-construction regardless of whether the filters ever diverge — it just hasn't been *given* a row to misplace.

## The fix — one canonical frame

The cure is to stop maintaining two parallel notions of "which rows survived" and derive everything from a **single canonical frame**. The render step already returns `anchor_kept`, which is 1:1 with `embs` by construction — same rows, same order, because `embs` was produced *from* it. Make that frame the source of truth:

```python
# anchor_kept is 1:1 with embs by construction (same render keep_idx)
vol_feats, vmask = _volume_features_for_anchors(anchor_kept, price_lookup, ...)
embs        = embs[vmask]
merged_kept = anchor_kept.iloc[vmask].reset_index(drop=True)
# vol_feats is already the vmask subset
```

Now there is exactly **one** mask (`vmask`), computed over exactly **one** frame (`anchor_kept`), applied to **everything together**. `embs[vmask]`, `anchor_kept.iloc[vmask]`, and the already-subset `vol_feats` are guaranteed to be the same rows in the same order — not by accident, but because they all flow from the same boolean over the same universe.

## The general anti-pattern

Two mistakes recur whenever code juggles arrays beside DataFrames:

1. **Boolean mask from frame A applied to array B.** A mask is only meaningful against the exact collection it was computed over. Reusing it on a differently-derived array is a category error that the type system can't catch — both are just `bool[n]`.
2. **Positional prefix as a stand-in for a boolean subset.** `df.iloc[:k]` is not "the rows that passed"; it's "the first `k` rows." These coincide only when filtering is a no-op or strictly tail-trimming.

The fragile core that makes both possible is the `reset_index(drop=True)` + `.index.values` idiom — collapsing a DataFrame's index to bare positions and then using those positions to slice numpy arrays. Positional alignment has no self-check. Contrast this with a join on a real key, which would simply produce NaNs or fail to match. The same theme recurs in [duplicate-date dedup](/story/09/50_atomic_writes_integrity) and the [SPY date-alignment](/story/09/42_beta_zero_bug) bug.

**The principle: one canonical frame, one mask, applied to everything at once.**

## Why this class of bug is the worst kind in ML

A crash gets fixed before lunch. A bug that throws an exception is a *gift* — it tells you where and when. This one does none of that. It produces no error, no NaN, no warning; it corrupts the exact thing you're measuring — the relationship between features and labels — and leaves every downstream metric looking healthy. You can't unit-test your way out of it after the fact, because the corrupted run and the correct run are indistinguishable from their outputs alone. The only defenses are structural: align by construction, assert lengths and keys at the seams, and never let a coincidence carry your correctness. See [the hardening story](/story/09/39_hardening_story) for the broader pass, [testing philosophy](/story/09/45_testing_philosophy) for the assertion-at-the-seams discipline (and the honest note that `run_pipeline.main` still lacks an end-to-end test), and [reproducibility seeds](/story/09/12_reproducibility_seeds) for why a silently-misaligned run is doubly insidious when it's also perfectly repeatable.
