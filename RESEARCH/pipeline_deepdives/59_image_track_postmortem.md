# The DINOv2 image-track post-mortem — why it lost and what would make it fair

The image track is the most expensive thing in this pipeline and it lost. This is the autopsy — not a defense, not a quiet deletion, but an honest accounting of *why* a frozen 22M-parameter vision transformer got beaten by 40 numbers, and what experiments would be needed before we could call that verdict final rather than provisional.

## The pipeline we are burying

Let's be precise about what the image track actually computes, because the cost is in the details. Each example starts as a **252-day window of closing prices** (one trading year). That window is rendered by `render.py` into a single 224×224 RGB image: white background, one black polyline at `line_width=2`, no axes, no ticks, no grid, no labels. The y-axis is the [fixed log-y axis](/story/09/29_fixed_log_y_axis) spanning 0.1× to 11× of the anchor close — the correction that stops a +5% chart and a +500% chart from collapsing to the same silhouette.

That image goes into a **frozen [DINOv2 ViT-S/14](/story/09/01_dinov2_architecture)** running on CPU in `eval` mode, no grad, ImageNet normalization. Out comes a **384-dim** CLS embedding. We then run [PCA-64](/story/09/02_pca_math) on the image embeddings, run a separate PCA-64 on the [volume features](/story/09/05_volume_processing), `np.concatenate` the two into a **128-dim** vector, and hand that to either logistic regression (`fit_lr_baseline`) or LightGBM (`fit_lgbm_image`). Every scaler and PCA is re-fit on the train fold only, so this is leakage-safe — that part is not the problem.

The [falsification test](/story/05_falsification) ran this side-by-side against `lgbm_engineered`, which skips images entirely and feeds [40 hand-built features](/story/09/03_engineered_features) straight to LightGBM. After costs, **engineered won**. Here is why.

## Reason 1 — the encoder is out of distribution

DINOv2 was trained, self-supervised, on a corpus of *natural photographs* — textures, objects, animals, scenes. A sparse black line on a white field is about as far from that distribution as an input can get. The model never had a reason to learn features that mean "this chart is forming a base" or "this is a parabolic blow-off." Its early layers most plausibly fire on **line density, local curvature, stroke thickness, and where ink sits in the frame** — geometric properties of a polyline, not the chart-pattern *semantics* a technician would read. We are asking a photo encoder to do chart reading, and getting an embedding that is *about the picture* rather than *about the price action*.

## Reason 2 — PCA is unsupervised, and this is the real asterisk

This is the most important caveat, so read it carefully. PCA keeps the directions of **maximum variance** in the 384-dim embedding. It has no idea what the label is. If the predictive signal — the tiny part of the embedding that actually correlates with "this name returns ≥25% in 40 days" — lives in a **low-variance direction**, [PCA](/story/09/02_pca_math) throws it away before LightGBM or LR ever sees it. The 64-dim cut could be discarding exactly the columns that mattered.

That means the falsification verdict might be **partly PCA's fault, not DINOv2's**. We genuinely do not know how much of the loss is "the encoder has no signal" versus "the encoder has signal and we threw it out." The honest position is that an *unsupervised* bottleneck handicaps the image track in a way the engineered track never suffers — its 40 features go to the model un-projected. The [roadmap](/story/09/47_roadmap) flags PLS as the fix.

## Reason 3 — engineered features skip the lossy round-trip

The engineered track encodes the economically-relevant content of the chart **directly**: momentum, realized volatility, max drawdown, moving-average ratios. These are the things the chart is *a picture of*. The image track takes that same information, throws it onto a 224×224 raster — quantizing a year of prices into ~224 x-pixels and one log-scaled y-axis — then asks a photo model to *re-extract* it. Every step of that render-then-re-embed round-trip is lossy. The engineered features are a clean shortcut to the destination; the image pipeline is a scenic detour through a pixel grid and a foreign encoder. (See the [384-number deep-dive in chapter 9](/story/09_pipeline_walkthrough) for what those embedding dimensions actually contain.)

## Reason 4 — 128 dims into a small fold is data-hungry

Even granting the embedding some signal, the concatenated 128-dim vector is a lot of input for the amount of data per fold. With [degenerate / small folds](/story/09/48_degenerate_folds) — some so thin that `_split_calib` has to drop stratification because a class has only one member — a 128-wide feature space invites overfitting and unstable calibration. The engineered track's smaller, denser feature set is simply easier to fit honestly on this much data. LR especially (see [LightGBM internals](/story/09/31_lightgbm_internals) for the GBM contrast) wants more rows per dimension than these folds provide.

## What would make the comparison fair

The verdict stands, but it stands **with an asterisk**, and the asterisk is removable. Three experiments would make this a fair fight:

1. **Supervised reduction instead of PCA.** Swap PCA-64 for **PLS-64 or LDA** — methods that project toward directions correlated with the label, not just high-variance ones. This directly tests the Reason-2 hypothesis, and is the single highest-value next experiment.
2. **A chart-native encoder.** Stop borrowing a photo model. Train a small CNN from scratch on charts, or skip rendering entirely and run a **1D PatchTST** on the raw price series. An encoder that has actually seen charts removes the OOD handicap of Reason 1.
3. **Per-fold supervised fine-tuning.** Unfreeze DINOv2 and fine-tune on the chart task per fold, so the backbone adapts to the new distribution rather than emitting frozen photo features.

Until at least the PLS swap is run, we cannot say DINOv2 has *no* signal — only that, **as wired** (frozen photo encoder → unsupervised PCA → small fold), it does not beat 40 hand features after costs. That is a true statement about this configuration, not a verdict on vision-for-charts in general.

## Why we keep the losing track

We do not delete `lgbm_image`. Two reasons. First, **faithful reproduction**: the original document proposed the DINOv2 pipeline, and removing it would quietly erase the thing we set out to test. Second, **the falsification only works if both tracks run side-by-side**. A claim that "engineered beats image after costs" is empty unless the image track is still there, still running, still losing under identical folds and identical costs every time. The loser is load-bearing — it is the control. See [the verdict](/story/09/55_the_verdict) for the full numbers and [the hardening story](/story/09/39_hardening_story) for how we kept both tracks reproducible under degenerate conditions.
