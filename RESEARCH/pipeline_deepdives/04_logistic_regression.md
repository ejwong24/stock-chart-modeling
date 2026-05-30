# Logistic regression internals — turning 128 numbers into 1 probability

The classifier at the heart of this pipeline is the simplest model that still does something useful with 128 PCA features. No trees, no neural nets, no kernel tricks — just a single dot product squashed through a sigmoid. This document walks through what actually happens when those 128 numbers (64 image PCA + 64 volume PCA, both pre-scaled) get turned into the single probability we use for ranking, and why the number you see on the leaderboard isn't quite what it looks like.

We'll use BNTX, our #10 pick with a calibrated score of **0.720**, as the worked example throughout.

## The sigmoid: from a real number to a probability

Logistic regression starts with a linear combination of the inputs:

```
z = w · x + b
   = w_1*x_1 + w_2*x_2 + ... + w_128*x_128 + b
```

For BNTX, each of the 128 PCA components gets multiplied by a learned weight, summed, and offset by a bias `b`. The output `z` is an unbounded real number — it could be -4.2, it could be +1.8. To turn it into a probability we squash it through the logistic (sigmoid) function:

```
P(y=1 | x) = sigmoid(z) = 1 / (1 + exp(-z))
```

Three useful facts about this curve:

- `sigmoid(0) = 0.5`
- `sigmoid(+inf) -> 1`, `sigmoid(-inf) -> 0`
- It is **strictly monotone**: bigger `z` always means bigger probability. This will matter later.

So `z` is the "raw evidence score" and the sigmoid is just a presentation layer.

## The loss the model is actually minimizing

During training, sklearn finds `w` and `b` by minimizing the average cross-entropy loss over the training fold, plus an L2 penalty:

```
L(w, b) = (1/N) * sum_i [ -y_i * log(p_i) - (1 - y_i) * log(1 - p_i) ]
        + (1 / (2*C)) * ||w||^2
```

where `p_i = sigmoid(w · x_i + b)`. The first term pushes predictions toward the right labels; the second term keeps the weights small.

## L2 regularization and what `C=1.0` means

In sklearn, `C` is the **inverse** regularization strength. Small `C` means heavy regularization (weights pulled hard toward zero); large `C` means almost none. `C=1.0` is the default and a moderate choice — strong enough to prevent any single PCA component from dominating, gentle enough that real signal survives.

With 128 inputs and only a few thousand training anchors per fold, some regularization is non-negotiable. Without it, lbfgs would happily find a weight vector that overfits to noise in the high-order PCA components (which by construction explain less variance).

## Why `lbfgs`

`lbfgs` is a quasi-Newton method: it approximates the Hessian using only recent gradient history, which makes it cheap in memory but still much faster than plain gradient descent for smooth convex problems. L2-regularized logistic regression is exactly that — smooth and convex. For our problem size (a few thousand rows, 128 features, L2 penalty), lbfgs converges in a handful of iterations on easy folds and dozens on hard ones.

We set `max_iter=1000` to be safe. Most folds finish in well under 100 iterations; the limit is there so that the occasional hard fold (e.g. a period with very few positives) doesn't silently produce a non-converged classifier.

## `class_weight='balanced'` and the calibration cost

Our labels are heavily imbalanced — only about 5-15% of anchors have `ret_40d_ge_25pct = 1`. Take a representative 8% positive rate. Sklearn's `class_weight='balanced'` computes:

```
weight(c) = n_samples / (n_classes * bincount(y)[c])

# n_classes = 2, positive rate = 0.08
weight(1) = 1 / (2 * 0.08) ≈ 6.25
weight(0) = 1 / (2 * 0.92) ≈ 0.54
```

So every positive example contributes roughly **6.25× more loss** than every negative. This stops the model from collapsing to the degenerate "always predict 0" solution that would otherwise get 92% accuracy and tell us nothing.

But there is a price. The classifier no longer minimizes loss against the **true** distribution — it minimizes loss against a re-weighted distribution where positives and negatives are effectively equally common. The optimal `b` (bias) shifts accordingly. The model now believes a typical anchor has a 50% chance of being positive, because that's what its training distribution looks like.

## What the output number actually means

Concretely, `P = 0.5` from this model does **not** mean "50% chance this stock goes up 25% in 40 days." It means "this anchor is on the boundary in a world where positives and negatives are equally common."

Two consequences:

1. **Ranking is fine.** Because the sigmoid is monotone in `z`, sorting by `P` is identical to sorting by `z`. The order is information-preserving.
2. **Absolute thresholds are not fine.** If a future user reads our leaderboard and decides "I'll only buy when `P > 0.6`," they will get either nothing (because almost no anchor crosses 0.6 once calibration is reapplied) or everything (because in the balanced world a lot do).

For BNTX at **0.720**: the right reading is "this anchor is meaningfully above the model's balanced-world midpoint, and ranked #10 out of all anchors in its fold." The wrong reading is "the model is 72% confident BNTX will hit the +25%/40d threshold."

## The partial fix: per-fold [isotonic calibration](/story/09/30_isotonic_calibration)

To claw back some interpretability, the pipeline fits an **[isotonic regression](/story/09/30_isotonic_calibration)** on a held-out 10% slice of each training fold, mapping raw probabilities to empirical positive rates. Isotonic is a non-parametric monotone mapping, so it preserves rank order while pulling probabilities back toward the true base rate.

Caveat: this calibration is fit **per fold**. The mapping that turned BNTX's raw `z` into 0.720 was learned from one specific slice of one specific fold. A score of 0.720 in fold 7 and 0.720 in fold 12 are not strictly comparable — they're outputs of two different isotonic functions on two different underlying classifiers.

## Reading the leaderboard

Putting it together: when DBI shows a score of 1.0 and BNTX shows 0.720, the right interpretation is:

- DBI is the strongest pick in its fold; BNTX is the 10th-strongest in (likely) another fold.
- The **ratio 0.72 / 1.0** says BNTX's evidence is meaningfully below DBI's on a relative scale, but it does **not** mean "DBI is 100% certain, BNTX is 72% certain." Neither score is a true probability.
- For trading decisions you should treat these as ordinal ranks, not Bayesian probabilities.

---

> **Note — why we kept `class_weight='balanced'` anyway.** This rebuild deliberately preserves the original document's choice. Dropping it would let the classifier minimize loss by predicting 0 for everything (correct ~92% of the time, useful 0% of the time). The calibration distortion is a known, documented cost — and isotonic calibration plus rank-based interpretation is the workaround. Anyone changing this knob needs to also change how downstream consumers read the scores.
