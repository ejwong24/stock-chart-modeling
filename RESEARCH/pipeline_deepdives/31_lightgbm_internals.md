# LightGBM internals — how gradient boosting actually finds tree splits

The `lgbm_engineered` model — the one that survived our [falsification gauntlet](/story/05_falsification) — is a LightGBM classifier. Understanding why it wins requires understanding what gradient boosting actually *does* under the hood: not just "an ensemble of trees," but a very specific iterative procedure that fits each tree to the *gradient* of a loss function.

## Gradient boosting in one paragraph

Gradient Boosted Decision Trees (GBDT) are a **sequential ensemble**. The procedure: start with a constant prediction `F_0(x) = mean(y)`. Compute residuals `r_i = y_i - F_0(x_i)`. Fit a small tree `h_1(x)` to predict those residuals. Update the prediction: `F_1(x) = F_0(x) + lr * h_1(x)`. Recompute residuals against `F_1`. Fit `h_2` to *those* residuals. Repeat `n_estimators` times.

```
F_M(x) = F_0(x) + lr * (h_1(x) + h_2(x) + ... + h_M(x))
```

Each tree is small. Each tree corrects the *mistakes* of the running ensemble.

## Why GBM beats single trees

A single deep decision tree fits anything but overfits everything. A GBM dodges this two ways: **each tree is shallow** (we cap at `num_leaves=63`, roughly depth 6) so no individual tree memorizes; and **each subsequent tree corrects the residuals** of all previous trees, so the ensemble can learn arbitrary nonlinearities and interactions. The `learning_rate << 1` shrinkage prevents any one tree from dominating.

This is exactly the structure that beats [logistic regression](/story/09/04_logistic_regression) on tabular data with interactions.

## The 'gradient' in gradient boosting

What we're fitting each tree to is the **negative gradient of the loss function with respect to the current prediction**. For squared error loss `L = (y - F(x))^2`, the negative gradient is `y - F(x)` — the literal residual.

For our binary log-loss objective (`objective="binary"`), the gradient works out to:

```
-dL/dF = y - sigmoid(F(x))    # observed label minus current predicted probability
```

So when LightGBM is "fitting a tree to the residual," it's really fitting a tree to *how wrong the predicted probability is, in the direction that would most reduce log-loss*. This is **gradient descent in function space**.

## How LightGBM finds splits — the histogram-based approach

Classic GBDT implementations sort feature values and try every candidate threshold. That's `O(features × n_samples × log n_samples)` per split.

LightGBM instead **bucketizes each feature into ~255 bins** at fit time (a quantile-based histogram). Finding the best split then reduces to scanning bin boundaries: `O(features × bins)`. On our ~50k-row folds with 40 [engineered features](/story/09/03_engineered_features), this is roughly two orders of magnitude faster per tree than exact-split GBDT — without measurable accuracy loss.

The histogram trick also lets LightGBM exploit a **subtraction identity**: once you've computed the histogram for a node, the histogram for one child is `parent_hist - other_child_hist` — free.

## Leaf-wise growth

Most boosters grow trees **level-wise**: expand all leaves at depth `d` before considering depth `d+1`. LightGBM grows **leaf-wise**: at each step, split the *single leaf* with the highest gain, regardless of depth. The resulting trees are unbalanced but each split reduces loss by the maximum amount available at that moment. Empirically this fits faster but overfits more aggressively — which is why `num_leaves` and `min_child_samples` matter so much.

## Our hyperparameters explained

```python
LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)
```

- **`n_estimators=400`** — 400 sequential boosting rounds. With `lr=0.05`, that's `400 × 0.05 = 20` "effective trees" of full influence.
- **`learning_rate=0.05`** — the shrinkage parameter. Smaller = more trees but more robust. Typical sweet spot: 0.01-0.1.
- **`num_leaves=63`** — max leaves per tree. Roughly depth 6 since `2^6 = 64`.
- **`min_child_samples=200`** — refuse to split a node with fewer than 200 samples. Critical for our ~50k-row [walk-forward folds](/story/09/08_walkforward_embargo).
- **`reg_alpha=0.1`** — L1 penalty on leaf weights. Encourages sparsity.
- **`reg_lambda=0.1`** — L2 penalty on leaf weights. Shrinks leaves toward zero.
- **`objective="binary"`** — predicts `P(y=1)` as `sigmoid(F(x))`.

## The feature_importance output

LightGBM ranks features by **total gain** — the sum of split-improvement scores across every tree. For `lgbm_engineered`, the top features consistently are `ret_252d`, `pct_from_252d_high`, and `vol_252d`. See [The 40 engineered features](/story/09/03_engineered_features).

## Why we use it

1. **Handles tabular nonlinearities and interactions.** The [40 engineered features](/story/09/03_engineered_features) have known multiplicative relationships (e.g., high momentum AND low vol = high-quality momentum) that [linear models](/story/09/04_logistic_regression) can't express without manual cross-feature engineering.
2. **Robust to feature scale.** No `StandardScaler` needed.
3. **Fast inference.** 400 trees × ≤63 leaves is tiny by deep-learning standards.

## Why not XGBoost

XGBoost is essentially identical in accuracy on tabular data of our size. LightGBM is faster on CPU (histogram + leaf-wise), and the original document used LightGBM. We faithfully reproduce.

## The calibration

Raw `sigmoid(F(x))` outputs are not well-calibrated probabilities. We fix this exactly as we do for LR: hold out a 10% slice, fit `IsotonicRegression`, apply it to test-fold predictions. See [Isotonic calibration](/story/09/30_isotonic_calibration).

## The bit-equality

With `random_state=42` set, two runs of `lgbm_engineered` on identical input produce **byte-identical predictions**. See [Reproducibility seeds](/story/09/12_reproducibility_seeds).
