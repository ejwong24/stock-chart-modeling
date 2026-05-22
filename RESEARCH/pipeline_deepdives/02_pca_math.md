# What PCA is doing, and why per-fold matters

## The plain-language definition

Principal Component Analysis finds new axes for your data such that each axis captures as much variance as possible while staying perpendicular to the ones before it. Imagine a cloud of points in 384-dimensional space. The **first principal component (PC1)** is the single direction along which that cloud is most stretched out. The **second principal component (PC2)** is the direction of biggest remaining spread, subject to being orthogonal to PC1. PC3 is the biggest spread orthogonal to both, and so on.

When we keep the top 64 of these 384 directions, we are choosing the 64 axes that, together, "explain" the most variance in the embedding cloud. Projecting each 384-dim DINOv2 embedding onto those 64 axes gives us a 64-dim vector that preserves 91.5% of the variance of the original. The other 320 directions get thrown out as low-variance noise.

That is the entire elevator pitch. PCA is a coordinate rotation followed by a truncation, chosen to keep the directions of biggest spread.

## The math, in one formula

Given a centered data matrix `X` of shape `(N, 384)`, compute the covariance matrix `C = (1/N) * X^T @ X`. Then eigendecompose:

```
C = V @ diag(lambda) @ V^T
```

The columns of `V` are eigenvectors (the principal directions), and `lambda` is the vector of eigenvalues (the variance captured along each direction). Sort the eigenvectors by descending eigenvalue and stack the top 64 columns into a matrix `W` of shape `(384, 64)`. The compressed embedding is `Z = X @ W`.

In practice we use SVD on `X` directly rather than forming `C` explicitly — it is numerically better behaved — but the result is mathematically the same. The 64 PCs we keep are exactly the 64 eigenvectors of the covariance matrix with the largest eigenvalues.

## Why variance is not the same as predictive signal

Here is the uncomfortable part. PCA optimizes for variance in `X`. It never looks at the label `y`. That means it is entirely possible for PCA to discard the features that actually matter for prediction.

Concrete toy example. Suppose dims 0 through 99 of our 384-dim embedding are scaled-up noise channels — they have huge variance but are uncorrelated with next-week return. Dim 100 has a tiny variance — maybe a standard deviation of 0.01 — but its sign perfectly predicts whether the stock goes up or down next week. The remaining 283 dims are middling.

PCA-64 will happily keep linear combinations of dims 0–99 because they dominate the covariance matrix. Dim 100, with its tiny variance, gets absorbed into one of the discarded 320 components. We just threw away the only feature that mattered, and our reconstructed 64-dim vector still "explains 91.5% of variance" in the data we don't care about.

This is not hypothetical. Image embeddings are full of high-variance directions encoding lighting, color, gross composition — none of which has any business predicting stock returns. The directions that *might* matter (subtle chart patterns, candle structure) could easily live in low-variance corners of the embedding space.

## Why we fit PCA on each training fold

If you fit PCA once on the full dataset — train + validation + test — you have committed a leak. The eigenvectors of the combined covariance matrix encode information about how the test-period embeddings are distributed. Your "compressed" features at training time are now defined in a coordinate system that quietly knows about the future.

It is not a labels-direct leak (PCA never sees `y`). It is a covariance-structure leak: the test-period embeddings shifted the principal axes, and your training projections were computed against those shifted axes. In a walk-forward setting where you are pretending to not have seen 2024 yet, that is cheating.

The fix is mechanical:

```
# inside fold k
pca = PCA(n_components=64).fit(X_train_fold_k)
Z_train = pca.transform(X_train_fold_k)
Z_test  = pca.transform(X_test_fold_k)   # transform only, no refit
```

Fit on train, transform-only on test. Refit per fold. We pay 24 PCA fits across the walk-forward but we keep the test boundary honest.

## What "the basis is not stable across folds" means

Because we refit PCA in every fold, the 64 axes are different in every fold. The PC1 of the fold ending 2020-12-31 is some specific 384-dim direction in embedding space. The PC1 of the fold ending 2021-12-31 is a *different* direction, computed from a different (overlapping but not identical) training window.

So when the downstream model in fold A learns "PC5 is important", and the downstream model in fold B also learns "PC5 is important", those are not the same PC5. They are unrelated coefficients in unrelated coordinate systems that happen to share an index. Component-1 of one fold is not component-1 of another.

The practical implication is that we cannot interpret components. We cannot say "PC5 represents trending stocks" or "PC12 captures consolidation patterns" because PC5 means something different in every fold. Any interpretive story we tell about a specific PC is at best valid within one fold and at worst pure pareidolia.

This is fine for prediction — the model relearns the coefficients in each fold — but it kills any hope of a stable feature narrative.

## Alternatives that could plausibly do better

PCA is the default, not the optimum. Several methods could in principle extract more signal from the 384-dim DINOv2 embedding:

- **Partial Least Squares (PLS)** — supervised. Picks directions that maximize covariance with the label, not just variance in `X`. Would not have discarded our hypothetical dim 100.
- **Linear Discriminant Analysis (LDA)** — supervised. Picks directions that best separate the classes (up vs down). Limited to `n_classes - 1` components, so not a drop-in for 64 dims, but powerful for binary classification.
- **Independent Component Analysis (ICA)** — picks statistically independent directions rather than orthogonal-by-variance. Useful when the underlying signals are non-Gaussian.
- **Sparse PCA** — adds an L1 penalty so each component uses only a handful of the 384 dims. Loses some variance but each PC becomes interpretable as "these 8 input dims matter together".
- **Kernel PCA** — applies PCA in a feature space induced by a kernel, capturing nonlinear structure that linear PCA cannot.
- **Random Projection** — Johnson-Lindenstrauss lemma says a random `(384, 64)` matrix preserves pairwise distances almost as well as PCA, with no fitting at all. Often shockingly competitive.
- **Autoencoder** — learn a nonlinear `384 -> 64 -> 384` bottleneck end-to-end. Can capture structure linear methods miss, at the cost of training another network.

## Why we still chose PCA

Three reasons. **Simplicity:** one line, no hyperparameters worth tuning beyond `n_components`. **Reproducibility:** PCA is deterministic given the training fold, so fold-to-fold variation is purely a function of the data, not of random init. **Parity with the source document:** the original procedure we are replicating used PCA, and matching it lets us isolate the effect of the changes we *did* make.

The deeper reason is that the falsification test ended up showing the image stack did not earn its keep in the final ensemble. If DINOv2 + PCA-64 isn't pulling its weight, swapping PCA for PLS would shuffle deck chairs on a ship that already isn't sailing. We picked the boring choice and the boring choice did not turn out to be the bottleneck.

## Open question

> Would supervised PLS change the falsification verdict? PCA is label-blind; if the predictive signal in DINOv2 embeddings lives in low-variance directions, PCA discarded it and we never had a fair test of the image stack. Rerunning the walk-forward with PLS-64 (or even just LDA + 63 PCA components) is the cleanest way to find out whether the image stack is genuinely uninformative or merely poorly compressed.
