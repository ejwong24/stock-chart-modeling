# Volume processing — log1p, z-score, and PCA again

Volume is the second pillar of the pipeline's "what does the market look like" signal. Where the image embedding captures the *shape* of recent price action, the volume track captures *conviction*. This note walks through how a raw 252-day volume series gets turned into 64 features that sit next to the 64 image PCs in the LR/LGBM head.

## The transform, end to end

For each `(ticker, anchor_date)` we pull the past 252 trading days of share volume and run three nested operations:

```
log_volume_t   = log(1 + volume_t)
z_t            = (log_volume_t - mean(log_volume_{1..252})) / std(log_volume_{1..252})
features_64    = PCA-64(StandardScaler(z_{1..252}))
```

The output is a 64-dim vector that is then concatenated with the 64-dim image PCA vector and handed to the downstream model.

## Why `log1p(volume)`

Share volume is brutally heavy-tailed. A megacap's typical day might be 30M shares, but a single earnings surprise or short squeeze can push it to 300M — a 10x spike. Across a year, the maximum is often an order of magnitude above the median. Feeding raw volume into a linear model (or even PCA) lets those few outlier days dominate every principal component; you end up with PCs that essentially encode "was there a spike day, yes/no" instead of structure.

Taking a log compresses the dynamic range so that a 10x spike becomes a +2.3 bump rather than a +900M outlier. The `1+` matters because some sessions have zero volume — halts, holidays that slipped past the calendar filter, delisted-then-relisted tickers. `log(0) = -inf` would poison the whole window; `log1p(0) = 0` handles it cleanly. It's a small thing but it removes an entire class of NaN-propagation bugs.

## Why z-score over the trailing 252, not globally

Different tickers trade at completely different baselines. Apple does ~50M shares/day; a microcap might do 50k. If we z-scored against a *global* mean, every microcap day would look like "extremely low volume" and every megacap day would look "extremely high," and the resulting feature would mostly encode market cap — which we already capture elsewhere.

By z-scoring each stock against its **own** trailing year, we ask a different question: *is today's volume high or low relative to what this stock normally does?* A microcap doing 200k shares (4x its baseline) and Apple doing 200M shares (4x its baseline) now both score around z = +2. That's the comparable signal we want.

The window is also strictly causal — only days `1..252` before the anchor, no leakage from the prediction window.

## What the 252-dim vector encodes

After the transform, the 252-dim vector is a *normalized volume history* — for each day in the lookback, how unusual was the trading activity relative to the stock's own past year. Values cluster around zero, with positive numbers for above-baseline days and negatives for quiet ones. The shape of this vector — where the spikes sit, whether they cluster recently, whether there's a slow drift up — is what PCA is meant to summarize.

## Why [PCA-64](/story/09/02_pca_math) again

Same reasoning as the image track. 252 dimensions is too many to feed a linear head with the data we have, and adjacent days are highly correlated, so the effective rank is much lower than 252. [PCA-64](/story/09/02_pca_math) picks off the leading patterns:

- PC1-ish: overall volume level vs the year (was the last quarter louder than the first?)
- PC2-ish: trend (is volume drifting up over the window?)
- PC3-ish: recency (was the spike near the anchor date or months ago?)
- ...and 60 more, capturing increasingly fine wiggles.

The StandardScaler before PCA matters: even after the per-stock z-score, the variance at different lags isn't identical, and we don't want the PCs to be dominated by whichever day happens to have the widest spread.

## Why concatenate, not stack two models

We feed `[image_PC_64 ; volume_PC_64]` = 128 features into one LR (and one LGBM). The alternative — train an image model and a volume model separately, then average — throws away the cross-signal. The whole point of a joint linear head is that it can learn weights like *"price-pattern PC3 is bullish ONLY when volume PC7 is also elevated"*. A momentum signal with confirming volume is different from a momentum signal on thin tape, and only a joint model can express that.

## The awkward redundancy with [engineered features](/story/09/03_engineered_features)

In the engineered track we also compute `log_dollar_vol_z252`, `vol_ratio_20_252`, and `vol_trend_slope_60d` — three hand-crafted scalars that summarize roughly the same information the 64-dim PCA is trying to extract automatically. In ablation, the engineered scalars do most of the work; dropping the PCA-64 volume block barely moves CV score. The PCA view is a more flexible representation, but with our data volume it can't pay rent against three well-chosen summary stats.

## BNTX as a concrete read

On the BNTX anchor, the engineered volume features read:

```
log_dollar_vol_z252  = -0.3526
vol_ratio_20_252     =  0.7553
```

Both numbers say the same thing: recent dollar volume was slightly *below* the trailing-year average, and the 20-day average was about 76% of the 252-day average. So whatever the rally was, it wasn't a parabolic blow-off on euphoric volume — it was orderly, almost quiet. The volume-PCA features for that anchor presumably tell a similar story across their 64 dims, but we can't read them as directly.

> **Callout — interpretability gap:** The 64 volume PCs contribute *something* to the final LR score, but how much, and through which PCs, is genuinely unclear. The LR coefficients can distinguish the image half from the volume half, but within a small-data fold the individual PC coefficients aren't independently estimable — they're correlated, regularized, and partially redundant with the engineered scalars. The honest answer is: the [engineered features](/story/09/03_engineered_features) carry the volume signal, and the PCA-64 block is along for the ride.
