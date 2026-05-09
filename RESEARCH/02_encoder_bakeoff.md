# Problem 2 — Encoder bake-off: what actually beats DINOv2 here?

## Synthesis (top action)

**Top-4 encoders to bench, in priority order:**

1. **PatchTST 1D transformer on raw returns** — native time-series inductive bias, no rendering loss, 2-4× faster than DINOv2 on 4-core ARM64. Expected lift: +2–4% AUC. Dev: 8–12 hrs.
2. **Tiny 1D CNN trained from scratch** — 91k examples is plenty for a ~60k-param model. Trains in 10–20 min/fold on CPU. Expected lift: +1–2.5% AUC. Dev: 1–2 days.
3. **Hand-engineered shape descriptors + LightGBM** — 8–12 new features (slope, accel, drawdown structure, MA fits, return-volume corr) on top of our existing 28. Near-zero compute. Expected: 60–75% of DINOv2's gap closure with full interpretability. Dev: 4 hrs.
4. **TS2Vec self-supervised on our anchor windows** — domain-tuned encoder, 8–15 hrs CPU pretrain. Expected lift: +1.5–3% AUC. Dev: 1–2 days.

**DROP from consideration:** Chronos (forecasting-tuned, overkill), MAE on chart images (3-5 days CPU pretrain on sparse line art), CLIP swap (same OOD problem), ResNet-18/ConvNeXt from scratch on charts (modality mistake).

**Bake-off protocol:** lock 2025 OOS, train all 4 encoders on 2017-2024 with same purged folds (purge zone = `max(input_window, label_horizon)` uniformly), same downstream LightGBM head, block-bootstrap 95% CI on CAGR, Holm-Bonferroni for K=4 comparisons. Total compute: ~24 hrs wall-clock on 4-core ARM64.

**Expected post-bake-off best:** PatchTST. Expected AUC lift over DINOv2: ~3%.

**Most important falsification outcome:** if no encoder swap lifts AUC by >0.5%, the bottleneck is NOT the encoder — it's label noise, regime non-stationarity, or weekly-return SNR. Pivot to label engineering (problem 5) instead of encoder shopping.

---

## A1 — PatchTST (HIGH inference, MED on pretrained weights)

Patch_len=16, stride=8, d_model=128, n_layers=3 → 30 patches × ~0.3M FLOPs/sample (vs DINOv2's 4.6 GFLOPs/sample). 40-80 samples/sec on 4-core ARM64 — 2-4× faster than DINOv2. No foundation-style pretrained encoder; HuggingFace `transformers.PatchTSTModel` (Apache-2.0) trains from scratch. Use mean-pool over patch tokens as embedding. Drop chart-rendering entirely.

```python
from transformers import PatchTSTConfig, PatchTSTModel
cfg = PatchTSTConfig(num_input_channels=1, context_length=252,
                     patch_length=16, patch_stride=8,
                     d_model=128, num_attention_heads=16,
                     num_hidden_layers=3, ffn_dim=256, dropout=0.2)
encoder = PatchTSTModel(cfg)
# In our pipeline: emb = encoder(past_values=x).last_hidden_state.squeeze(1).mean(dim=1)
```

## A2 — Chronos foundation model (HIGH tractability)

Chronos-Bolt-Tiny ~9M params, T5 encoder over quantized return-bins. Pretrained on synthetic + real time-series mix. ~3-6 ms/series on 4-core with onnxruntime-arm64. Drop-in encoder. Expected lift: +1.5–3 pp AUC.

```python
from chronos import ChronosBoltPipeline
pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")
enc = pipe.model.model.encoder  # T5Stack — extract hidden states
```

## A3 — TS2Vec self-supervised (HIGH tractability)

Dilated TCN, ~1M params, hierarchical contrastive (temporal + instance). Pretrained ON our 91k anchor windows — distribution match by construction. CPU pretrain: 50 epochs × ~5 min/epoch ≈ 4 hr/epoch wall on 4-core, total 8-15 hr. Output 320-dim per window. SSL representations risk being surface-level (mean/var); ensemble with engineered features.

## A4 — Tiny 1D CNN from scratch (HIGH tractability)

3 conv blocks, ~60k params, 1D over 252-day returns. Trains 10-20 min/fold × 8 folds = ~2 hr total. Stronger than frozen DINOv2 by +3–6 pp AUC honestly because DINOv2's natural-image priors don't apply. Best use: hybrid (CNN 64-dim emb concatenated with 28 engineered features) into LightGBM. Expected hybrid lift: +1–3 pp over engineered alone.

```python
class ReturnCNN(nn.Module):
    def __init__(self, n_emb=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1,32,7,2,3), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.3))
        self.b2 = nn.Sequential(nn.Conv1d(32,64,5,2,2), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3))
        self.b3 = nn.Sequential(nn.Conv1d(64,128,5,2,2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3))
        self.gap = nn.AdaptiveAvgPool1d(1); self.proj = nn.Linear(128, n_emb)
        self.clf = nn.Sequential(nn.GELU(), nn.Dropout(0.3), nn.Linear(n_emb, 2))
    def forward(self, x): return self.clf(self.proj(self.gap(self.b3(self.b2(self.stem(x.unsqueeze(1))))).squeeze(-1)))
```

## A5 — ResNet-18 / ConvNeXt-Tiny from scratch on charts (MED tractability)

ResNet-18 (~11M params) on 91k 224×224 images: 5-10 hours total. ConvNeXt-Tiny too slow on CPU. Custom small CNN (~2M params) ~3-4 hr total. **Recommendation:** skip image-domain training entirely. Modality is wrong.

## A6 — MAE self-supervised on chart images (LOW tractability)

3 honest problems: (1) sparsity — 99% white background means random masking learns "predict white" trivially; (2) compute — 3-5 days CPU; (3) modality — chart is lossy 1D rendering, do SSL on the source. **Skip; go TS2Vec instead.**

## A7 — CLIP image encoder swap (HIGH — diagnostic value)

OpenCLIP ViT-B/32 (Apache-2.0, laion2b_s34b_b79k weights) is FASTER than DINOv2 ViT-S/14 on ARM64 (49 patches vs 256). One-line swap. Different OOD signature than DINOv2 — text-aligned features may have a "stock chart going up" concept from LAION captions. Predicted: +3-7% AUC, doesn't fix fundamental issue but the negative result is the most valuable thing we can learn this week.

## A8 — Hand-engineered shape descriptors (HIGH tractability)

12 new features missing from our current 28-feature stack:

1. `slope_20`, `slope_60`, `slope_accel = slope_20 - slope_60`
2. `trendline_residual_z` (60d log-linear regression last residual / std)
3. `r2_60` (R² of 60d log-linear fit — clean trend vs chop)
4. `dd_count_5pct` (count of >5% drawdowns in 120d)
5. `days_since_peak`, `pct_from_60d_high`, `pct_from_60d_low`
6. `swing_count` (scipy.signal.find_peaks with prominence=ATR)
7. `bb_width_pct` (Bollinger squeeze detection)
8. `vol_of_vol_ratio = std(returns[-30:]) / std(returns[-120:-30])` (compression vs expansion)
9. `return_volume_corr_20` (volume-confirmation signal)
10. `hh_ll_score` (Dow theory trend classification)

Compute: ~45s pandas, ~8s polars+numba for 4000 tickers × 2500 days. Likely captures 60-75% of DINOv2's gap-closure value.

## A9 — Stack/ensemble multiple encoders (HIGH tractability)

Greedy forward selection, NOT concat. Procedure: (1) baseline = engineered LightGBM. (2) compute encoder embeddings once each, cache. (3) decorrelation diagnostic — drop any encoder pair with rank-corr > 0.9. (4) greedy add encoders that lift OOF IC by >0.003. (5) STACK via meta-LR on encoder predictions, don't concat (avoid 600-dim curse). Upper-bound lift: +5-10% IC, +0.1-0.2 Sharpe. If diagnostic shows all candidates correlate >0.85, expect ~zero lift.

## A10 — Per-week supervised fine-tuning (MED tractability)

Per-week temporal refit cuts training to 12k rows (5x fewer) — wrong axis. Equity 40d-return SNR ~0.05; cutting training data amplifies head variance more than it reduces regime bias. Gu/Kelly/Xiu found expanding windows beat rolling for exactly this reason.

**Better:** cross-sectional rank labels with per-week refit. Replace `P(ret_40d_ge_25pct)` with `rank_within_week >= 0.8` binary or LambdaRank with `group=anchor_date`. Eliminates regime drift in label base rate. Combined with `weeks_since_train_end` + `vix_level` features. ~4 dev days.

## A11 — Bake-off protocol design (HIGH tractability)

| Encoder | Input | Params | Train cost | Inference/anchor |
|---------|-------|--------|------------|------------------|
| Engineered LGBM (baseline) | 28d feats | 0 | 0 min | 0 ms |
| Custom 1D CNN | 252d returns | ~200k | ~90 min | <1ms |
| TS2Vec | 252d returns | ~500k | ~3-4 hr | ~2ms |
| PatchTST | 252d returns | ~1M | ~5-6 hr | ~3ms |
| DINOv2 frozen | 224×224 chart | 86M frozen | 0 train + ~6h embed | ~50ms |

**Critical:** purge zone = `max(input_window, label_horizon)` UNIFORMLY across all encoders so engineered baseline doesn't get unfair "tighter folds = more train data" advantage.

**Metrics:** Portfolio CAGR after costs (40d/25%) on 2025 only. Block-bootstrap 95% CI (block_size=40d). Per-anchor rank-IC mean/IR. Holm-Bonferroni at α=0.05 across 4 comparisons.

**Compute:** ~24 hrs total wall-clock on 4-core ARM64. Pre-commit the protocol BEFORE running.
