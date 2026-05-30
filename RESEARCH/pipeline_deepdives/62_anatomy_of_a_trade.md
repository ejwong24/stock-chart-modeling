# Anatomy of a trade — BNTX from raw bars to realized P&L

Chapter 9 walked the pipeline as a narrative; this is the same walk done with a ledger. One trade, BNTX, dragged through every module with the concrete value it carried at each step. The point is not that BNTX won — it's that you can put your finger on the exact number each stage emitted, watch it become the input to the next, and see how one row in a parquet file becomes a realized 53% gain. The prose version is [chapter 9](/story/09_pipeline_walkthrough); this companion follows the [data-flow contract](/story/09/60_data_flow_contract) end to end.

The trade: **BNTX, anchor 2021-07-02, entry ~$221.07, exit 2021-08-30 at ~$340.49 — +53.7% over 40 trading days.**

## The walk, one row at a time

| # | Module | Concrete value for BNTX @ 2021-07-02 |
|---|---|---|
| 1 | data_acq | `BNTX.parquet`, row at positional `anchor_idx` (adjusted daily bar, close ≈ $221.07) |
| 2 | labels | `fwd_ret_40d=+0.5394`, `ret_40d_ge_25pct=1`, `fwd_mfe_40d≈+1.00`, `fwd_mae_40d≈-0.05`, `resolution_date_40d=2021-08-30` |
| 3 | features | `ret_252d=+1.23`, `vol_252d=0.766`, `ratio_ma200=1.74`, `pct_from_252d_high=-0.072`, `beta_spy_63d` (now nonzero) |
| 4 | render+embed | 224×224 fixed-log-y PNG → 384-dim [DINOv2](/story/09/01_dinov2_architecture) → PCA-64 vector |
| 5 | model | LightGBM score `0.72`, rank **#10** that week |
| 6 | portfolio | passes ADV filter → buy ~18 shares @ ~$221 (~$4k notional) |
| 7 | simulator | hold 40 trading days, exit 2021-08-30 @ ~$340.49, `trade_return_pct ≈ +53%` net |
| 8 | lesson | rank #10 carried the cohort; the #1 pick (DBI, score 1.0) *lost* |

## 1. data_acq — the row exists and is the right row

Everything downstream is a function of one positional index. BNTX's adjusted daily bars live in `BNTX.parquet`, sorted ascending by date and deduplicated on the date key, so 2021-07-02 maps to exactly one row at positional index `anchor_idx`. That guarantee is not incidental — if two bars shared a date or the file were out of order, `anchor_idx` would point at the wrong day and every number here would be silently wrong. See [data acquisition](/story/09/13_data_acquisition) for where the bars come from and [atomic writes/dedup](/story/09/50_atomic_writes_integrity) for why the file you read is never half-written or duplicate-laden. The adjusted close at this row is ≈ $221.07.

## 2. labels — what the future held, computed forward-only

The label module looks 40 trading days ahead from `anchor_idx`, using only future bars so nothing leaks back into the anchor. For BNTX the forward return is `fwd_ret_40d=+0.5394` — a clean +53.94% — which trips the binary target `ret_40d_ge_25pct=1`. The path matters as much as the endpoint: `fwd_mfe_40d≈+1.00` means it touched roughly +100% maximum favorable excursion intra-window before settling back, while `fwd_mae_40d≈-0.05` says the worst drawdown along the way was a mild -5%. The window resolves on `resolution_date_40d=2021-08-30`. This whole row is grade-key material — see the [label grid](/story/09/19_label_grid). Note the asymmetry: a +100% spike against a -5% pit. BNTX was a low-pain, high-gain path, exactly the shape the strategy hunts.

## 3. features — what the model was allowed to see

Crucially, the model sees *none* of section 2. It sees only backward-looking features computed at the anchor. BNTX in mid-2021 was a textbook strong-momentum, high-volatility name: trailing-year return `ret_252d=+1.23` (up 123% over the prior year), annualized `vol_252d=0.766` (a wild 77% — this is a biotech that just shipped a vaccine), `ratio_ma200=1.74` (price 74% above its 200-day average, deeply extended), and `pct_from_252d_high=-0.072` (just 7.2% off its 52-week high, riding near the top). And `beta_spy_63d` is finally a real number here. Pre-fix it was hard-zero for every name — the [beta-zero bug](/story/09/42_beta_zero_bug) silently nulled a whole feature — so the model used to be blind to market sensitivity entirely. The full set is documented in [engineered features](/story/09/03_engineered_features).

## 4. render + embed — the chart as a vector

In parallel, the anchor's trailing window is rendered to a 224×224 PNG on a **fixed log-y axis** so a doubling looks like a doubling whether the stock is at $20 or $200 — see [fixed log-y](/story/09/29_fixed_log_y_axis) for why a floating axis would leak scale. That image goes through a frozen DINOv2 encoder into a 384-dim embedding, then compresses to a PCA-64 vector ([PCA](/story/09/02_pca_math)). For BNTX the picture is a steep, clean staircase — the visual signature of exactly the run-up the scalar features quantified. The chart and the numbers tell the same story in two languages.

## 5. model — a 0.72, ranked tenth

The PCA-64 vector plus the engineered features feed LightGBM ([LightGBM internals](/story/09/31_lightgbm_internals)), which after [isotonic calibration](/story/09/30_isotonic_calibration) scores BNTX at **0.72** — a confident but not maximal score. That places it **#10** in its weekly cohort. Worth pausing on: the model liked BNTX, but it liked nine other names more. By score alone, BNTX was a mid-tier conviction call.

## 6. portfolio construction — score becomes shares

A score is not a position. [Portfolio construction](/story/09/61_portfolio_construction) applies the liquidity gate first: BNTX's average dollar volume clears the ADV filter comfortably, so it survives where an illiquid micro-cap would be dropped regardless of score. The [position-size formula](/story/09/32_position_size_formula) then allocates a roughly equal-weight ~$4k slice, which at ~$221/share is about **18 shares**. The score got it into the room; the sizing rule decided how much it could matter.

## 7. simulator — slippage, commission, and the realized number

The [simulator loop](/story/09/06_simulator_loop) holds the 18 shares for the full 40-day horizon and exits on 2021-08-30 at ~$340.49. The gross move is +53.9%, matching the label. But the realized line is *net* of the [cost stack](/story/09/56_cost_stack) — entry slippage, commission, exit slippage — which shaves it to `trade_return_pct ≈ +53%`. On a +54% winner the costs barely register; on a coin-flip trade they are the whole margin. That ~18-share lot returned roughly $2.1k on ~$4k risked.

## 8. the lesson — rank #10 carried the cohort

Here is the uncomfortable payoff. BNTX, the model's *tenth* pick, returned +53%. The **#1 pick that week — DBI, calibrated score 1.0, maximum confidence — lost money.** The cohort was profitable not because the ranking was sharp but because one mid-ranked outlier dominated everything around it. This is the central, humbling finding of [the verdict](/story/09/55_the_verdict): in a fat-tailed return distribution, a single BNTX-shaped winner swamps the noise at the top of the board, and the model's per-name calibration is far weaker than its aggregate edge suggests. You don't need to pick the best stock — you need to be in the room when one does a BNTX, and not let costs, sizing, or a label bug ([hardening story](/story/09/39_hardening_story)) rob you of it.
