# The trading calendar — Fridays, holidays, half-days, and the ISO-week rule

The pipeline anchors weekly on Fridays, but "Friday" is a fiction the calendar happily punctures with holidays, early closes, and the occasional ISO week that doesn't have a Friday at all. This document explains the calendar mechanics that the [weekly anchor](/story/09/18_weekly_anchors) logic, [label grid](/story/09/19_label_grid), and [walk-forward embargo](/story/09/08_walkforward_embargo) all silently depend on.

## The US equity trading year

The NYSE/Nasdaq calendar averages about **252 trading days per year**. Arithmetic: 52 weeks × 5 weekdays = 260, minus roughly 8 federal market holidays = 252. Early-close half-days are still counted as full trading days because they generate a complete OHLCV row.

## Holidays we drop

These are the eleven days the US equity market closes (or partially closes) each year:

- **New Year's Day** (Jan 1) — closed
- **Martin Luther King Jr. Day** (3rd Monday in January) — closed
- **Presidents Day** (3rd Monday in February) — closed
- **Good Friday** (the Friday before Easter) — closed
- **Memorial Day** (last Monday in May) — closed
- **Juneteenth** (Jun 19) — closed since 2022
- **Independence Day** (Jul 4) — closed
- **Labor Day** (1st Monday in September) — closed
- **Thanksgiving** (4th Thursday in November) — closed
- **Day after Thanksgiving** ("Black Friday") — early close at 1pm ET, counted as a full trading day
- **Christmas Day** (Dec 25) — closed

When a date falls on a Saturday, the market is typically closed Friday (observed); when on Sunday, closed Monday.

## Half-days

Three sessions end at 1pm ET instead of 4pm ET: the day before Independence Day, the day after Thanksgiving, and Christmas Eve. yfinance reports a single OHLCV row per half-day, so the [label grid](/story/09/19_label_grid) treats them as full trading days. Volume is depressed on half-days, which slightly biases [ADV20](/story/09/33_adv20_metric) calculations downward.

## The Friday-anchor rule

Each ISO calendar week, we anchor on the **last trading day**. Usually Friday. Exceptions:

- **Good Friday week**: anchor moves to Thursday.
- **Thanksgiving week**: Friday is a half-day and counts as a full session, so the anchor stays Friday.
- **Christmas week**: behavior depends on the year.

Downstream code should never assume `anchor.weekday() == 4`. Use the weekly-anchor helper, which resolves "last trading day of ISO week" against the real calendar.

## The ISO-week >= 3 trading days rule

We require **at least 3 trading days in an ISO week** for an anchor to exist. Fewer than 3, and the week is dropped entirely (no anchor, no row in the [label grid](/story/09/19_label_grid)):

- **Memorial Day week** (Mon closed): Tue-Fri = 4 days, kept.
- **July 4 week**: usually 4 days, kept.
- **Christmas + New Year's weeks**: typically only 2-3 trading days each, may drop ONE.

## Why this matters for labels

The `fwd_ret_40d` label spans **exactly 40 trading days** from the anchor — not 40 calendar days. So an anchor on Friday 2021-07-02 resolves on Friday 2021-08-27, having skipped July 5 plus all weekends. The label is **calendar-day fuzzy but trading-day exact**.

## How splits.py uses this

The [walk-forward embargo](/story/09/08_walkforward_embargo) converts trading days to calendar days via a **1.6× multiplier**: 40 trading days ≈ 64 calendar days of embargo. Intentionally conservative. Thanksgiving and Christmas weeks compress the trading calendar more aggressively than 1.6× anticipates (3 trading days in a 7-day window = 2.33× compression), so the embargo slightly over-purges in those windows. Better than under-purging.

## Practical implications

- **Don't generate calendars from `pd.date_range`.** If a local parquet skips Memorial Day, the 40-trading-day-forward index lookup will misalign. Always derive the trading calendar from yfinance — see [Data acquisition](/story/09/13_data_acquisition).
- **Forward paper-trading** runs at Friday 4pm ET (see [Validation modes](/story/09/21_validation_modes)). If that Friday is a holiday, it runs Thursday or skips entirely.

## The DST trap

US equities trade on Eastern Time. Daylight Saving transitions (March + November) don't affect trading days, but shift every UTC↔ET boundary by an hour. Our pipeline uses **tz-naive datetimes throughout** (see [Reproducibility seeds](/story/09/12_reproducibility_seeds)). This trades a small amount of timestamp precision for immunity to DST bugs. For daily-resolution modeling, that's the right tradeoff.
