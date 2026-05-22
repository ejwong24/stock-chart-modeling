# The Close > 1.5 × SMA250 prefilter — what it does and why

## What it is, in plain English

Before any of the model scoring runs, the pipeline throws away every stock whose closing price isn't more than 1.5 times its own 250-day simple moving average. The SMA250 is just the average closing price over the prior 250 trading days — roughly one calendar year of market activity. A stock that closes at 1.5× its own SMA250 has, by definition, rallied at least 50% above its own one-year average level. In plainer terms: it's "extended," "in a strong uptrend," or in trader vernacular, "ripping."

This is the **momentum prefilter** inherited from the original research document. It is not a model output. It is a hard gate applied to the universe before any scoring happens.

## Why 250 days specifically

There are roughly 252 trading days in a US calendar year — 5 trading days per week times approximately 50.4 active weeks once holidays are removed. 250 is the round-number convention that approximates this. The choice is mostly cosmetic.

- **200-day MA** is more common among classical trend-followers (think Stan Weinstein, CANSLIM-adjacent practitioners).
- **365-day** would be the naive calendar-year choice, but markets aren't open on weekends or holidays, so it doesn't map cleanly onto bar data.
- **250** sits between these and is the convention the original document picked.

Anything in the ±20 day band would produce nearly identical eligible pools. The lookback is not load-bearing; the **multiplier** is where the real selection happens.

## Why 1.5× specifically

The 1.5× cutoff is taken directly from the original document and is, frankly, arbitrary. Different practitioners use different numbers:

- **1.2×** — looser, captures more steady uptrends, larger eligible pool.
- **1.5×** — our cutoff, captures meaningfully extended names.
- **2.0×** — stricter, isolates only the most parabolic movers.

Tightening the multiplier shrinks the candidate pool but raises the average momentum exposure of every name that survives. Loosening it does the opposite. There is no principled derivation — it's a tunable knob that the original document happened to fix at 1.5×.

## How many stocks survive at a given anchor

On the **2021-07-02 anchor week**, 373 stocks out of roughly 6,500 active US common stocks passed the filter — about **5.7%** of the active universe. That was near the peak of a historic bull market, when momentum exposure across the tape was unusually broad.

Contrast that with stress periods:

- **March 2020 (COVID crash)** — the eligible pool collapsed to **fewer than 100 stocks**. Almost nothing was trading 50% above its own one-year average.
- **2022 bear market** — a similar collapse. The filter starved.

The eligible pool is **small in bear markets and large in bull markets**, by construction. That makes the entire universe the model operates on **regime-dependent**: it effectively only exists when momentum is the dominant cross-sectional factor.

## The interaction with random baselines

This is the part that matters and is easy to miss. The pipeline benchmarks the model against random portfolios drawn from the **same prefiltered universe**. Because the random baseline inherits the universe, it also inherits the universe's regime tilt — it is implicitly long momentum too.

So when the model beats random, the claim being tested is **not** "the model has predictive skill." It is "the model has predictive skill *over and above* the existing momentum tilt baked into the eligible pool." A meaningful but much narrower claim.

## What's typically inside the filter

The kinds of names that survive the 1.5× SMA250 gate tend to share a profile:

- **Recent IPOs that have rallied** — fresh names, short price history, often parabolic.
- **Biotech winners** — vaccine plays like BNTX, MRNA, NVAX during the COVID era.
- **Energy in commodity booms** — BTU, VET, AR during 2021-2022 oil and coal spikes.
- **Single-ticker story stocks** — SAVA, AAOI, and other names that captured retail attention with a binary catalyst.

## What's filtered out

Equally important to understand what the filter *excludes*:

- **Large-cap blue chips with modest gains** — Apple up 25% YoY doesn't clear 1.5×, even though it's outperforming most stocks.
- **Sideways bear-market survivors** — names that drifted laterally and avoided drawdown but never rallied hard.
- **Distressed names below their MA250** — anything in a downtrend is gone, regardless of how cheap it might be.

The filter is explicitly a **momentum filter**, not a quality filter, not a value filter, not a mean-reversion filter. It selects for one thing only: recent and substantial relative strength against the name's own one-year average.

## Concrete worked example — BNTX

BioNTech (BNTX) is a canonical example for the 2021-07-02 anchor:

- **Close on 2021-07-02:** $221.07
- **SMA250 (prior 250 trading days):** $115.53
- **Ratio:** **1.914×**

BNTX clears 1.5× comfortably — the ratio is nearly 1.9×. The stock entered the eligible pool entirely because of its post-vaccine rally throughout the first half of 2021. Nothing about the model's scoring logic put it there; the prefilter did, before any model code ran.

This is the right mental model: BNTX wasn't selected by the model. It was **handed to the model** by the prefilter, along with 372 other momentum names that summer.

---

> See [/story/05_falsification](/story/05_falsification) for how the prefilter and the random baseline combine to inflate apparent edge — and why "model beats random" inside this universe is a much weaker claim than it first appears.
