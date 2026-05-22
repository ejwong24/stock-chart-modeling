# How DINOv2 ViT-S/14 actually works

When you click into BNTX on 2021-07-02 in /story/09, the chart image you're looking at was once a 384-number vector. That vector came out of a frozen DINOv2 ViT-S/14 model. This chapter explains what that sentence actually means, layer by layer.

## Vision Transformers in one paragraph

A Vision Transformer (ViT) is the same architecture as a language transformer, but the "words" are square patches of pixels. You take an image, slice it into a grid of non-overlapping tiles, flatten each tile, and project it into a vector. Now you have a sequence of vectors and you do exactly what BERT or GPT does: feed them through stacked self-attention + MLP blocks. There are no convolutions. The model learns spatial relationships purely from attention between patch-vectors.

The "S/14" part of `ViT-S/14` is two pieces of metadata:

- **S** = Small. Roughly 22M parameters, 12 transformer layers, 6 attention heads per layer, 384-dim hidden state.
- **/14** = patch size in pixels. Each patch is a 14×14 pixel tile.

## What gets fed in for a 224×224 image

```
image:           3 x 224 x 224     (RGB)
patches:         (224/14) x (224/14) = 16 x 16 = 256 patches
patch embedding: 256 x 384         (each patch -> 384-dim vector)
+ [CLS] token:   1 x 384           (a learned vector prepended)
+ positional:    257 x 384         (added, not concatenated)

token sequence:  257 x 384         -> input to layer 1
```

The `[CLS]` token is not derived from the image. It is a single 384-dim parameter vector, learned during pretraining, that gets prepended to the sequence on every forward pass. Think of it as a scratchpad slot the model is allowed to write a summary into.

After 12 layers of self-attention, the `[CLS]` row of the output tensor is the 384-dim vector we save. The other 256 rows (one per patch) are discarded.

## What attention is actually doing

Each layer's self-attention is the same scaled dot-product you've seen before:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
```

For ViT-S, `d_k = 384 / 6 = 64` per head. The 257×257 attention matrix at each head answers "for each token, how much should I look at every other token, including myself?" Six heads do this in parallel with different learned projections, results are concatenated back to 384-dim, fed through an MLP, residual-added, layer-normed. Repeat 12 times.

The `[CLS]` token gets to attend to all 256 patches at every layer. It has no spatial location of its own and no image content to defend — it is a pure aggregator. By layer 12 it has had 12 rounds to pull whatever it wants from whichever patches it wants. The output `[CLS]` vector is the model's compressed answer to "what's in this picture?"

## What each of the 12 layers tends to do

You shouldn't read too literally into this — transformer layers don't have crisp jobs — but probing studies of ViTs roughly find:

- **Layers 1–3**: local features, edge orientation, color blobs. Attention is mostly to nearby patches.
- **Layers 4–8**: mid-level structure. Heads start specializing — some track texture, some track shape, some track relative position. Attention spans widen.
- **Layers 9–12**: object-level abstraction. The `[CLS]` token's attention becomes broad and selective, focusing on a few "interesting" patches. This is where the final semantic content of the embedding is decided.

## Why DINOv2 specifically

You have three common ways to pretrain a vision encoder:

1. **Supervised ImageNet.** Tell the model "this is a cat" on 1.3M labeled photos. The encoder learns features useful for ImageNet's 1000 classes. Narrow.
2. **CLIP.** Train on image–text pairs scraped from the web. The encoder learns features that align with captions. Great for retrieval, biased toward whatever the internet writes about.
3. **DINOv2** (Meta, 2023). Pure self-supervised. No labels, no captions. Trained on LVD-142M, a curated 142M-image dataset. Uses self-distillation: a "student" network and a "teacher" network see different crops of the same image, and the student is trained to predict the teacher's `[CLS]` output. Teacher weights are an exponential moving average of student weights.

The result is an encoder whose features generalize to almost any downstream task without fine-tuning. That is exactly the property we wanted: drop in a chart, get out a vector, no training required.

## What "frozen" means in our pipeline

```
chart.png -> ViT-S/14 (no grad) -> 384-dim CLS -> PCA -> 64-dim
```

"Frozen" means we never compute gradients, never update weights. The model runs in `torch.no_grad()` mode (or equivalent) and is purely a feature extractor. On the Oracle ARM64 4-core CPU we get roughly 6 images/second, which is the bottleneck of the precompute step. No GPU is involved.

## Why this is out-of-distribution for line charts

DINOv2 has never seen a stock chart. LVD-142M is photographs of physical things: animals, food, vehicles, landscapes, faces. A line chart is a sparse 2D drawing on a near-white background — maybe 1% non-background pixels. The encoder almost certainly does not "see" the chart the way a human trader does. It is probably activating on:

- Line density per patch (more line = more "edge energy").
- Local slope and curvature within a patch.
- Coarse top-vs-bottom asymmetry across the canvas.
- Color of any non-grayscale elements (axis labels, gridlines).

It is **not** recognizing "double bottom" or "head and shoulders" as semantic objects. Those shapes don't exist in its training distribution. So the 384-dim vector is best understood as a rich-but-generic visual fingerprint of the image, not a chart-pattern classifier. That's both a feature (no chart-pattern bias baked in) and a limitation (no chart-pattern intelligence either).

## Why we use the final-layer [CLS] output, not intermediate features

```
forward pass produces 12 hidden states, each shape (257, 384)
we take:    hidden_states[-1][0]    # last layer, CLS row
shape:      (384,)
```

Earlier layers carry different information — more local, less abstract — and you can sometimes get better task-specific features by averaging patch tokens or pulling from layer 9 or 10. We did not explore this. The standard DINOv2 recipe is "use the final `[CLS]`," it's what the model was trained to produce as the summary, and changing it would force us to re-validate everything downstream.

## Callout: why we kept the simpler engineered features

> The 64 PCA components from DINOv2 ended up explaining less variance in next-day returns than the volume z-score and the 20-day return alone. DINOv2 is doing real work — it gives a dense, model-free description of the chart — but for predicting price movement on US equities, the engineered features encode the relevant signal more directly. We keep the DINOv2 features in the model as a regularizer and as a hook for future visual-pattern research, but they are not carrying the prediction. The chart-as-image hypothesis is, on this dataset, weaker than the chart-as-numbers hypothesis.
