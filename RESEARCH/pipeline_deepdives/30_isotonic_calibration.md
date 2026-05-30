# Isotonic calibration — recovering probabilities from class-weighted models

## The problem in plain terms

A classifier is **calibrated** if its stated probabilities match empirical frequencies. If you take every prediction the model emitted as `P(y=1|x) = 0.7` and look at what actually happened, 70% of those samples should be positive. A model can be highly accurate and badly miscalibrated, or poorly accurate and well calibrated — these are orthogonal properties.

In our pipeline we use a `LogisticRegression` with `class_weight='balanced'` (see [LR internals](/story/09/04_logistic_regression) for why we don't drop the weighting). That single argument breaks calibration in a specific way.

## Why class weighting destroys calibration

Our positive rate is roughly **8%** in the training window. `class_weight='balanced'` multiplies the positive-class loss by `n_neg / n_pos ≈ 6.25`, so the optimizer behaves as though it were training on a synthetic 50/50 dataset.

The consequence: the model is well calibrated **to a 50/50 world that does not exist**. A raw output of 0.7 doesn't mean "70% chance this stock goes up." It means "in the balanced-prior world the optimizer was hallucinating, this is well above the midpoint of evidence."

## What isotonic regression does

Isotonic regression is a **non-parametric monotone mapping** fit on `(raw_prob, true_label)` pairs. Unlike Platt scaling or beta calibration, isotonic makes no functional assumption — it returns a step function that interpolates the empirical positive rate among samples with similar raw probability.

The optimization problem:

```
For ordered raw_probs r_1 <= r_2 <= ... <= r_n with labels y_1, ..., y_n:
isotonic finds the unique sequence p_1 <= p_2 <= ... <= p_n minimizing
  sum( (y_i - p_i)^2 )
subject to p_i in [0, 1].
The pool-adjacent-violators algorithm solves this in O(n) time.
```

The pool-adjacent-violators algorithm (PAVA) walks through the sorted predictions and whenever it finds a violation `p_i > p_{i+1}`, it pools the two values into their average. Linear time, no hyperparameters.

## The 10% held-out slice

**The isotonic must be fit on data the LR never saw.** Otherwise the calibration overfits to training residuals.

We carve off 10% with `train_test_split(..., random_state=seed)` (see [Reproducibility seeds](/story/09/12_reproducibility_seeds)) **before** the LR fit. The LR sees 90% of the fold; the isotonic gets the remaining 10%. Split stratified on the label so the calibration slice retains the ~8% positive rate.

10% is not a lot — typically 200–500 samples per fold. This is one of the limitations we'll come back to.

## Per-fold isotonic

A fresh `IsotonicRegression` is fit inside every [walk-forward fold](/story/09/08_walkforward_embargo). The relationship between raw score and true positive rate drifts across regimes — calibrating fold 11 with a mapping learned in fold 7 would smuggle stale market behavior into a new period.

The consequence: **calibrated scores are not directly comparable across folds.** BNTX scoring 0.720 in fold 7 does not mean the same thing as a different stock scoring 0.720 in fold 11 — they passed through different step functions.

## What isotonic preserves

**Rank order.** Because isotonic is monotone by construction, the ordering produced by the LR is preserved exactly. Picking the top-5 stocks by calibrated probability is identical to picking the top-5 by raw probability. Isotonic only touches the absolute numbers, not the sort.

## What isotonic doesn't fix

The pull toward the balanced-50/50 prior introduced by `class_weight='balanced'` is too strong for isotonic to fully undo with only 10% of the fold. The honest calibration would require:

- Platt scaling on a substantially larger holdout (we don't have the data)
- Refitting the LR without `class_weight`, which on our 8% positive rate collapses to "always predict 0" (see [LR internals](/story/09/04_logistic_regression))
- A two-stage model with explicit prior correction

So our isotonic does a partial job.

## Why we keep it anyway

1. **Partial calibration is better than none.**
2. **Per-fold consistency** makes year-over-year reporting less misleading.
3. **Faithful reproduction.** The original document used isotonic at this step.

## Concrete example: BNTX

Raw LR output: ~0.85. Isotonic mapping: 0.85 → 0.720. The practical reading: BNTX is in the **top 30% of confidence** for fold 7, not the top 15% the raw 0.85 would have suggested.
