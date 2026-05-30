# Closing the loop — settle_picks.py and the realized forward record

For most of this project's life the forward harness was a one-way street. Every Friday after close, [`forward_pick.py`](/story/09/46_forward_pick_harness) froze a model on data through today, scored the freshest tradable anchor per ticker, and wrote the week's top-5 to `data/forward_picks/<date>.csv`. Then nothing. The picks sat on disk accruing future, but no script ever went back to ask "what actually happened?" The loop was open. `scripts/settle_picks.py` closes it: it walks the picks directory and, for every pick whose 40-trading-day window has now resolved, computes the realized return and appends it to `data/forward_realized/realized.csv`.

## The labeling–settling symmetry

Forward picking and settling are the same arithmetic run at two different moments in a bar's life. `labels.py` can only label an anchor whose forward window has *already* resolved — it trims the trailing ~H bars precisely because a 40-day-ahead label is undefined for the most recent 40 days. `forward_pick.latest_candidates()` does the deliberate opposite: it generates the latest tradable anchor at/before today *whose forward window has not resolved*. That's the whole point of a forward pick — you bet on the unresolved present.

`settle_picks.py` is what waits for that present to become the past. Once 40 trading days have elapsed, the anchor the picker scored is now a labelable anchor, and settling grades it with exactly the label-time arithmetic: same entry, same positional exit, same return formula. The picker scores the unresolved anchor; the settler grades it once resolved. Two halves of one operation.

## The realized_return() core

The heart of the script is small and matches `labels.py` bar-for-bar:

```python
def realized_return(df, anchor_date, horizon):
    if df is None or len(df) == 0:
        return None
    dates = pd.to_datetime(df["date"]).to_numpy()
    anchor = np.datetime64(pd.Timestamp(anchor_date))
    pos = np.flatnonzero(dates == anchor)
    if len(pos) == 0:
        return None                  # anchor not in series
    entry_idx = int(pos[0])
    exit_idx = entry_idx + horizon
    if exit_idx >= len(df):
        return None                  # window not yet resolved
    closes = df["close"].to_numpy(dtype=np.float64)
    entry_close = float(closes[entry_idx])
    exit_close = float(closes[exit_idx])
    if entry_close <= 0:
        return None                  # meaningless entry
    realized = exit_close / entry_close - 1.0
    return realized, pd.Timestamp(dates[exit_idx]), entry_close, exit_close
```

The exit is positional — `entry_idx + horizon` indexing into the ticker's sorted parquet — not a calendar offset, which is why this respects the [trading calendar](/story/09/37_trading_calendar) the same way `labels.py` does: 40 *bars*, not 40 days. Three guards return `None`: the anchor date isn't in the series (a parquet got re-pulled and the bar shifted), the exit index runs off the end (the window hasn't resolved yet), or `entry_close <= 0`. That last guard is the scar tissue from [label corruption](/story/09/43_label_corruption) — a non-positive entry close produces a garbage ratio, so we refuse to record it.

## Idempotency and atomic write

`settle_all()` is a pure function of disk state — it can be re-run any number of times and only ever settles *newly resolvable* picks. It loads the already-settled keys first:

```python
return {(str(r.pick_generated), str(r.ticker), int(r.horizon_days))
        for r in df.itertuples(index=False)}
```

The `(pick_generated, ticker, horizon)` triple is the dedup key. Any pick whose key is already in `realized.csv` is skipped before we even touch prices, so a daily cron that runs after a holiday — or twice in one day — produces no duplicate rows. New rows are written through the project's standard [atomic write](/story/09/50_atomic_writes_integrity) discipline: build the combined frame, write to `realized.csv.tmp`, then `os.replace()`.

## Why this is the strongest evidence

Among the [validation modes](/story/09/21_validation_modes), the realized forward record is categorically different. [Walk-forward CV](/story/09/08_walkforward_embargo), purged splits, the backtest simulator — all are replays of history the model was tuned against, however carefully we wall off leakage. The forward picks are the only trades chosen by a model frozen *before* the outcome existed, executed against bars that hadn't printed yet. They are out-of-sample in the only sense that can't be argued with, and they are regime-exposed. After ~26 weeks the file holds ~130 such trades — what [the verdict](/story/09/55_the_verdict) ultimately has to be read against.

## The operational step and what's still manual

The code is done and tested; the remaining work is wiring. Per the [roadmap](/story/09/47_roadmap), `settle_picks.py` should run daily under system cron — a cheap, no-LLM "run script, settle what resolved" job, exactly the kind the project convention keeps in the system crontab. It is safe to schedule blindly: idempotent, atomic, and a no-op on days when nothing new resolves.

What's still manual: nobody yet *reads* `realized.csv` and reports the aggregate (hit rate, mean realized return, distribution versus the backtest). The settler produces the ledger; turning that ledger into a periodic scorecard is the next piece. The behavior is pinned by regression tests covering the four cases that matter — resolvable→correct return, unresolved→None, anchor-absent→None, and `settle_all` idempotency — the same hardening posture as the rest of [the hardening story](/story/09/39_hardening_story), applied to the one part of the pipeline whose output we can never regenerate after the fact.
