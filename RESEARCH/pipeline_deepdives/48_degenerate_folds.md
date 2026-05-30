# Degenerate folds — PCA components, tiny samples, and singleton classes

Walk-forward evaluation has a built-in asymmetry: the earliest test years are starved of training data. Two bugs in `src/stock_chart/models.py` lived in exactly that starved corner — both invisible until a fold got small enough or skewed enough to violate an assumption that holds comfortably everywhere else. Both are now fixed.

## Why early folds are tiny

In [walk-forward + embargo](/story/09/08_walkforward_embargo), each fold trains on everything before a cutoff and tests on the year after. The first test years see almost no prior history — the training window is short by construction. Then the embargo carves out the rows nearest the boundary to prevent label leakage, shrinking the usable training set further. A late-decade fold might train on 5,000 rows; the opening fold can train on 30. The pipeline's `run_pipeline` guard only requires ≥10 samples per class (≥20 total), so a 20–63-row fold sails past the gate and into the model code, where dimensionality assumptions quietly break.

## The PCA constraint

PCA cannot manufacture more components than the data supports. From [PCA](/story/09/02_pca_math): the principal components are eigenvectors of the covariance matrix, and a rank-`r` data matrix yields at most `r` non-trivial directions. Concretely, `n_components <= min(n_samples, n_features)`.

The old code only respected half of that bound:

```python
# before — clamps to features, ignores samples
PCA(n_components=min(pca_dim, n_features))
```

With `pca_dim=64` and 384-dim image embeddings, `min(64, 384) = 64` — fine, as long as you have at least 64 rows. A fold with 20–63 training rows passed the per-class guard, hit PCA, and died:

```
ValueError: n_components=64 must be between 0 and min(n_samples, n_features)=30 with svd_solver='full'
```

The fix adds the missing term:

```python
PCA(n_components=min(pca_dim, Xi.shape[1], Xi.shape[0]))
```

Now `pca_dim` is a *ceiling*, not a demand. A 30-row fold gets `PCA(30)`; a wide-but-tall fold gets `PCA(64)`. The model degrades to fewer components instead of crashing — the right behavior for a fold that has little signal to extract anyway.

## The singleton-class trap

The second bug lived in the calibration split. Every fold holds out a 10% slice for [isotonic calibration](/story/09/30_isotonic_calibration), and the split was stratified to preserve class balance in both halves — sensible, because positives are rare (~8% base rate), and an unstratified 10% slice can easily land zero positives.

But stratification has its own degenerate case. `train_test_split(stratify=y)` raises when any class has exactly one member:

```
ValueError: The least populated class in y has only 1 member, which is too few.
```

At an 8% positive rate, a tiny or skewed fold can contain a single positive label. The [logistic regression](/story/09/04_logistic_regression) and [LightGBM](/story/09/31_lightgbm_internals) paths both feed this split, so both inherited the crash. The fix degrades gracefully — stratify only when it's legal:

```python
counts = np.bincount(y) if len(y) else np.array([0])
strat = y if counts.min() >= 2 else None
return train_test_split(X, y, test_size=0.10, random_state=seed, stratify=strat)
```

## Why the tests missed both

The existing model tests used `n=200–500` with roughly balanced classes — comfortably above the sample bound and never near a singleton. The one PCA-clamp test varied only the *feature* count (`200×10` with `pca_dim=64`), exercising `min(pca_dim, n_features)` but never `min(…, n_samples)`. As [testing philosophy](/story/09/45_testing_philosophy) argues, a test that only moves one axis can't catch a bug on the axis it holds fixed.

## The regression tests

Two new tests pin the degenerate edges: a 30-row fold with `pca_dim=64` trains and returns finite predictions (locking the sample term into the `min(...)`); and a label vector with a single positive trains without raising (locking in the `stratify=None` fallback).

## The lesson

Both bugs share a shape: a default tuned for the typical fold — 64 components, stratified calibration — silently assumes the data is large enough and balanced enough to honor it. On the degenerate edges of walk-forward, neither assumption holds. The discipline is twofold: clamp every dimensionality and sample assumption to the *actual* data (`min(want, n_features, n_samples)`), and degrade gracefully rather than crash. This is the same posture as the rest of [the hardening story](/story/09/39_hardening_story) and [numerical stability](/story/09/44_numerical_stability): the edges are where defaults go to die, so meet the data where it actually is.
