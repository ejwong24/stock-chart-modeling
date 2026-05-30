# The verdict — why this pipeline shows no demonstrated edge, stated plainly

## The one-sentence version

After fixing survivorship bias, walk-forward leakage, chart auto-scaling, transaction costs, and [21 bugs across 18 hardening passes](/story/09/39_hardening_story), this pipeline shows **no statistically demonstrated edge over a simple momentum baseline** that uses no machine learning at all.

That is the whole finding. The rest of this page exists to make sure the sentence means exactly what it says — no more, no less — and to lay every supporting thread side by side so you can check the conclusion yourself.

## What the original document claimed

[The original claim](/story/01_claim) was a clean, seductive number: **+61% CAGR**, turning $100k into $3.23M over seven years, off a deep-learning pipeline that read price charts as images. That is the headline we set out to reproduce honestly. Not to debunk by assertion — to *rebuild*, instrument, and let the corrected machinery report whatever it reports.

What it reports is a much smaller number with a much wider error bar.

## Thread 1 — the headline gap is real but indistinguishable from zero

On the `full` run, the best learned configuration beats the **best simple baseline** by **+5.26% CAGR**. Taken alone, that looks like an edge. It is the strongest single fact in favor of the pipeline, and we lead with it precisely because we are not trying to bury the good news.

But a point estimate without an error bar is a rumor. The [paired gap test](/story/09/52_paired_gap_test) — a paired bootstrap that resamples the *same* time blocks for the model and the baseline so their shared market exposure cancels — puts a **95% confidence interval of roughly [−15%, +49%]** around that +5.26%. The probability that the true gap is zero or negative, `p(gap ≤ 0)`, comes out around **0.27**.

> A 27% chance the "edge" is actually a deficit is not an edge. It is noise wearing an edge's clothing.

The interval straddles zero by a wide margin. We cannot reject the hypothesis that the model and the baseline are the same strategy with extra steps.

## Thread 2 — the best Sharpe fails [multiple-comparison](/story/09/23_multiple_comparison_landscape) correction

Across **N = 108 configurations** tried, the best [Sharpe ratio](/story/09/27_sharpe_ratio) we observed is **~0.51**. The [Deflated Sharpe Ratio](/story/09/10_deflated_sharpe) machinery asks the right question: *given that we tried 108 things and reported the maximum, how high would the best Sharpe be by pure luck?* The answer sets a threshold around **0.75**.

Our best, 0.51, sits **below** that threshold — exactly the kind of number you would expect to surface by chance from trying a hundred-plus variants. It does not survive the correction for having looked many times.

## Thread 3 — the sample is far smaller than it looks

The `lgbm_engineered` run executes **1,104 round-trip trades**. That sounds like a lot of evidence. It is not, because the trades overlap in time — roughly 25 positions are open on any given day, so when the market moves, dozens of "independent" trades move together.

The [effective sample size](/story/09/24_effective_sample_size) calculation (López de Prado average uniqueness) collapses those 1,104 trades to about **44 independent bets**. Standard errors scale as `1/sqrt(N)`, so:

```
SE widening = sqrt(1104 / 44) ≈ 5.0x
```

The naive, tight confidence interval around the Sharpe widens by a factor of five and swallows zero. We made 1,104 trades but effectively saw ~44 decisions — and you cannot demonstrate a subtle edge on 44 decisions.

## Thread 4 — the falsification test the expensive track failed

This is the result that ought to sting most for anyone who believed in the original premise. The whole point of the original document was that **AI reading charts as images** was the source of the edge. So we ran the [falsification test](/story/05_falsification): pit the [DINOv2](/story/09/01_dinov2_architecture) image-embedding track against a track of plain [engineered features](/story/09/20_simple_baselines) — 40 hand-computed numbers, barely "AI" at all.

After costs, the **engineered-features track beats the image track.** The [image track postmortem](/story/09/59_image_track_postmortem) lays out why: the vision pipeline's marginal signal didn't clear its own marginal cost and complexity. The expensive, headline-grabbing component — the part that made the original claim feel like the future — **did not earn its place in the pipeline.** If anything carried weight, it was boring momentum features that have been in the literature since 1993.

## Thread 5 — costs and taxes erase most of what's left

Backtests live in a frictionless fantasy until you make them pay. The [cost stack](/story/09/56_cost_stack) — spread, [Almgren-Chriss](/story/09/09_almgren_chriss) market impact, commissions, borrow — eats **30–50% of the pre-cost CAGR**. Layer realistic short-term-gains tax on the high-turnover weekly rotation and the **post-tax CAGR lands around 2.6%.**

A pre-tax point estimate of a few percent over a baseline becomes, after the government and the market makers take their cut, a number that rounds to "roughly a Treasury bill with extra variance." None of the [survivorship accounting](/story/09/57_survivorship_accounting) corrections — putting the delisted, dead, and acquired tickers back into the universe so we stop only trading the winners that survived — push it back up. They push it down.

## Thread 6 — the one test that could still vindicate it hasn't run long enough

Here is where intellectual honesty cuts both ways. Every test above is *retrospective*: it interrogates history we already have. The one test that is immune to the entire catalog of backtest sins — survivorship, leakage, overfitting to a fixed past — is **live forward paper-trading**, the [forward validation mode](/story/09/21_validation_modes) driven by the [settle loop](/story/09/53_settle_loop).

That harness is running. It places picks in real time and settles them as the future actually arrives, with no opportunity to peek. But it **has not accumulated enough trades** to say anything. The forward record is the only court that could overturn the verdict, and that court is still in session.

## "No demonstrated edge" is not "definitely no edge"

This distinction is the entire ethic of the project, so read it twice.

- **No demonstrated edge** means: *with the evidence we have, corrected the way it must be corrected, we cannot distinguish this pipeline from a free momentum rule.*
- It does **not** mean the strategy is proven worthless. Absence of demonstrated edge is a statement about the strength of our evidence, not a proof of the null.

Three concrete things would change the verdict:

1. **A supervised-reduction image rerun** — replacing PCA-on-DINOv2 with a label-aware dimensionality reduction could give the image track the signal it was starved of, possibly reversing Thread 4. See the [roadmap](/story/09/47_roadmap).
2. **An accumulated forward record** — enough settled live trades for the [settle loop](/story/09/53_settle_loop) to produce a forward Sharpe with a confidence interval that clears zero.
3. **A tighter universe** — if the edge is real but concentrated, restricting to a defensible sub-universe (sector, liquidity band, regime) could lift the gap out of the noise.

Any one of those, done honestly, could move `p(gap ≤ 0)` decisively below 0.05. We would report it the moment it did.

## The negative result is the product

It is tempting to read all this as failure. It is not. The original document sold a number; this reproduction sells a **method**. What we built is a rigorous, honest, fully reproducible **template for evaluating any claim of this shape** — the kind of claim that arrives weekly, dressed in deep learning and a hockey-stick equity curve.

Drop a new strategy into this harness and it will be run through survivorship correction, [walk-forward embargo](/story/09/08_walkforward_embargo), realistic costs, paired-gap bootstrap, deflated Sharpe, and effective-sample-size deflation — automatically, the same way, every time. The verdict here happens to be "no demonstrated edge." The *value* is that the verdict is **trustworthy**, falsifiable, and would have said "yes" just as readily if the evidence had supported it.

A pipeline that can honestly tell you "no" is worth far more than one that only knows how to say "+61%."

---

> **See also**
> - [/story/01_claim](/story/01_claim) — the original +61% CAGR claim, stated in full.
> - [/story/05_falsification](/story/05_falsification) — the engineered-vs-image falsification test.
> - [/story/09/52_paired_gap_test](/story/09/52_paired_gap_test) — the paired bootstrap behind the [−15%, +49%] interval.
> - [/story/09/10_deflated_sharpe](/story/09/10_deflated_sharpe) — the multiple-comparison correction the best Sharpe fails.
> - [/story/09/24_effective_sample_size](/story/09/24_effective_sample_size) — 1,104 trades → ~44 independent.
> - [/story/09/56_cost_stack](/story/09/56_cost_stack) and [/story/09/57_survivorship_accounting](/story/09/57_survivorship_accounting) — where the CAGR went.
> - [/story/09/53_settle_loop](/story/09/53_settle_loop) and [/story/09/21_validation_modes](/story/09/21_validation_modes) — the live test still in progress.
> - [/story/09/47_roadmap](/story/09/47_roadmap) — what would change the verdict.
