# Survivorship bias — the full accounting

Every backtest that has ever looked too good has usually been lying in the same way. It studied the winners. The losers — the companies that went bankrupt, got minimum-bid delisted, reverse-split into oblivion, or got bought for cash at a discount — were quietly absent from the data, because the data vendor only carries names that still trade. Train on that, test on that, and you have built a machine that confirms a tautology: stocks that survived to today went up on the way to surviving. This page is the quantitative accounting of how this project fights that bias, how far the fight gets, and exactly what residual remains.

## The mechanism, and why it's the #1 flaw

The original incarnation of this project defined its universe as "tickers that existed on 2026-04-20." That single line is the most expensive sentence in any naive backtest — not a small sampling artifact but the dominant one. A name that fell 90% and got delisted in 2019 is simply not in the file. The model never sees it, never trains on its terminal collapse, never gets penalized for a signal that would have walked straight into it. Apparent returns inflate, apparent drawdowns shrink, and the whole edge is an illusion stitched out of absences.

Survivorship correction is the act of putting the dead back into the room. You cannot grade a strategy honestly if the strategy is only ever shown companies that, by construction, did not die. See [/research/01_survivorship_bias](/research/01_survivorship_bias) for the full problem framing and the four-source plan it proposes.

## The three sources of delisted names

The project does not have a paid point-in-time database (Polygon/CRSP/EODHD), so it reconstructs the graveyard from free sources. Three of them feed [universe construction](/story/09/14_universe_construction):

**1. The hand-curated seed.** `config/delisted_seed.txt` is a conservative, manually assembled list of US names known to have delisted, gone bankrupt, gone private, or hit severe distress between 2016 and 2026 — SIVB, SBNY, FRC, BBBY, the SPAC casualties, the meme-stock blowups. It is loaded by `load_delisted_seed()` in `universe.py`, which strips comments and marks every entry `active=False`. It is small and human-trustworthy, the floor of coverage rather than the ceiling.

**2. SEC EDGAR Form 25 scraping.** `scripts/scrape_edgar_form25.py` hits EDGAR full-text search (`efts.sec.gov/LATEST/search-index`) for Form 25 and Form 25-NSE — the canonical exchange delisting notices, which together cover essentially all NYSE/NASDAQ/AMEX delistings from 2005 onward. The scraper paginates each form, rate-limits to ~7.7 req/s under SEC's 10/s cap, and extracts tickers from the `display_names` field with a regex tuned for SPAC unit/warrant tokens (`CNDA-UN`, `CNDA-WT`), falling back to SEC's CIK→ticker master JSON when the display name carries no symbol. This is the heavy lifter: thousands of unique delisted tickers, dwarfing the seed.

**3. Inferred delistings from data we already hold.** `scripts/detect_inferred_delistings.py` exploits a yfinance quirk — it forward-pads zero-volume bars to the most recent trading day instead of exposing delisting status — so a dead ticker leaves a fingerprint. The detector reads every adjusted parquet and `classify()`s it against five distress signatures:

- **early_terminal** — the last bar predates the corpus-wide terminal date (2026-04-20); a live ticker would reach it.
- **zerovol** — a trailing run of ≥10 zero-volume bars (the forward-pad tail).
- **subdime** — a final close at or below $0.10 (with a softer penny-distress tier for sub-$0.50 on near-zero volume).
- **bagel_drop** — a ≥90% drawdown from the trailing 20-day peak.
- **zero_close** — a literal $0.00 final close.

SPAC warrant/unit/rights suffixes (W/U/R) are skipped to cut false positives. Any name tripping at least one signature is emitted as an incremental candidate to augment the seed — no paid feed required.

## Coverage: honest about partial

Stack the three and the picture is real but incomplete. The hand-curated seed is on the order of a few hundred names against an estimated ~1,200 US stocks delisted across 2017–2024 — roughly 28% on the seed alone. EDGAR closes most of the rest, pushing combined capture toward the ~85–90% of historical attrition the research doc targets. But "most" is not "all." Free sources structurally miss pre-2005 delistings, OTC/pink-sheet drop-offs with no formal event, foreign ADR terminations, and share-class consolidations. We are correcting the bias, not eliminating it.

## What "beats random" actually means here

This matters because two load-bearing comparisons draw from this same partially-corrected pool. The [MA250 prefilter](/story/09/07_ma250_prefilter) screens anchors out of this universe, and the [random baselines](/story/09/20_simple_baselines) sample entries from it. So when a strategy "beats random," the honest reading is: it beats random *within this universe*. If the universe still over-represents survivors, the random baseline is itself slightly inflated, and the margin over it is measured on a board that still tilts — less than the naive board, but not flat. The correction tightens the comparison; it does not make it absolute.

## Why the label-corruption bug was uniquely poisonous here

Of all the places the [label-corruption bug](/story/09/43_label_corruption) could have struck, this was the worst. The bug injected false-positive labels, and the names most exposed to mislabeling were precisely the delisted ones — the very stocks we paid real effort to drag back into the dataset. Survivorship correction exists to teach the model what failure looks like. A false-positive label on a delisted name does the exact opposite: it tags a company's death spiral as a *win*. That is not a neutral error. It actively inverts the lesson, telling the model that the road to delisting is something to chase. The bug threatened to undo the entire point of the graveyard. See the [hardening story](/story/09/39_hardening_story) for how the labels were rebuilt.

## The residual gap, and the fix that closes it

The clean fix is a true **point-in-time universe**: a listing/delisting calendar so that each anchor date sees exactly — and only — the set of tickers that were actually tradable on that day. No name leaks in from the future; no dead name silently vanishes from the past. That is on the [roadmap](/story/09/47_roadmap), gated on either a paid feed (EODHD ~$20/mo, Tiingo free-with-caps) or the Wayback iShares/NASDAQ-Trader membership scrape that `01_survivorship_bias` sketches. Until then, our universe is "today's names plus a reconstructed graveyard," which is close to point-in-time but not identical to it. See [data acquisition](/story/09/13_data_acquisition) for where the bars themselves come from.

## Why partial correction still matters

It would be easy to dismiss 85% coverage as "still biased, so why bother." That misreads the stakes. The gap between a 0% corrected backtest and an 85% corrected one is the gap between a number that is *dishonest* and a number that is *honest but imperfect*. The first lies about the direction of the result; the second is merely fuzzy about its magnitude. A strategy that survives a graveyard-aware test has cleared a bar the naive version never saw. That is the foundation [the verdict](/story/09/55_the_verdict) rests on.
