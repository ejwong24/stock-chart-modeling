# The fixed log-y axis (−90% to +1000%) — why it's the single most important rendering choice

The stock-modeling pipeline renders 252-day price windows as 224×224 grayscale images in `src/stock_chart/render.py`. Those images get fed into a [DINOv2](/story/09/01_dinov2_architecture) encoder, which produces embeddings that we later compress via [PCA](/story/09/02_pca_math) and benchmark against the [engineered features track](/story/09/03_engineered_features). The entire image-track hypothesis lives or dies on whether the rendering step preserves economically meaningful structure. The original document had a flaw severe enough to invalidate every downstream conclusion — the project's audit catalogs it as "flaw #4".

## The original flaw, in concrete terms

The original renderer used a per-window auto-scaled y-axis: every chart was independently rescaled so that its own min and max prices spanned the full 224-pixel canvas. The intent was probably "make every chart use the available pixels efficiently." The effect was catastrophic.

Imagine two stocks over the same 252-day window:

- **Stock A:** closes at 100, drifts to 105, ends at 105 (+5% over the year).
- **Stock B:** closes at 100, ramps to 600, ends at 600 (+500% over the year).

With per-chart auto-scaling, **both** fill the same 224-pixel canvas. Stock A's tiny 5% drift becomes a visually huge wave. Stock B's parabolic +500% move becomes a visually similar wave. [DINOv2](/story/09/01_dinov2_architecture) sees almost identical images. The magnitude of the move — arguably the single most important property of a stock chart — has been destroyed by the renderer before the encoder ever runs.

## The fix: a fixed log-y axis with bounds [−90%, +1000%]

Every chart now shares the same y-axis, anchored to the anchor day's closing price and expressed as a multiplicative return. Stock A's +5% move occupies a tiny vertical band near the middle of the canvas. Stock B's +500% move occupies most of the canvas. The image now **encodes magnitude**.

## Why log-y, not linear-y

Stock returns are log-normal. A 50% gain and a 33% loss are economically equivalent — they cancel exactly: `1.5 × (2/3) = 1`. On a linear-y axis the +50% move would be visually 1.5× larger than the −33% move, even though they undo each other. Log-y gives them equal visual weight.

## The exact range: log(0.1) to log(11)

```
y_lo = log(0.1)   # -90% return  → stock keeps 1/10 of its value
y_hi = log(11)    # +1000% return → stock 11× its value
```

The −90% bound captures complete blowups. The +1000% bound captures parabolic blow-offs. Anything outside this range gets **clipped** to the bound — the line just touches the top or bottom of the canvas. A stock that went +5000% shouldn't visually dominate every chart it appears next to; it's just "off the top."

## The implementation

```
log_ratios = log(close[t] / anchor_close)
y_lo = log(0.1)
y_hi = log(11)
norm = (log_ratios - y_lo) / (y_hi - y_lo)
ys   = (1.0 - norm) * (image_size - 1)   # invert: low y = top of image
```

## The defensive guard (Bug #7 fix)

The audit flagged a separate bug: if `y_lo >= y_hi` (degenerate or inverted range), the normalization divides by zero. The fix synthesizes a unit range:

```
if y_hi <= y_lo:
    y_hi = y_lo + 1.0
```

A regression test asserts that no `RuntimeWarning` is raised. See [Reproducibility seeds](/story/09/12_reproducibility_seeds) for the broader determinism discipline.

## Why this matters for [DINOv2](/story/09/01_dinov2_architecture)

The encoder's job is to differentiate "+5%" charts from "+500%" charts. With auto-scaling those collapse to the same image; with fixed log-y they don't. This is the single most important pre-DINOv2 design choice — every architectural tweak downstream of the renderer is irrelevant if the renderer itself is destroying signal.

## Why this matters for the falsification test

The [engineered features track](/story/09/03_engineered_features) doesn't render images at all. So the fixed log-y axis is irrelevant to that track. The point of fixing the renderer isn't to make the image track win — it's to give the image track a **fair** shot, so that when the image track still loses post-cost in [the falsification chapter](/story/04_costs), we can honestly conclude that the loss is about information content, not a rendering bug.
