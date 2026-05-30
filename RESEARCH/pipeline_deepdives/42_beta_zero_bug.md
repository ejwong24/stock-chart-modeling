# Why beta_spy_63d was silently zero — a dead-code feature and a doc that lied

There is a particular flavor of bug that is worse than a crash, worse than a wrong number, and worse than a flaky test. It is the bug that produces a *plausible* number — one so plausible that a human looks at it, nods, and writes a paragraph of economic theory explaining why the universe produced exactly that value. `beta_spy_63d` was that bug. It was hard-wired to `0.0` for every stock on every anchor, fed to the model anyway, and then — most embarrassingly — *interpreted as a finding* in this very story series. This page is the correction.

## The off-by-one that became dead code

Here is the original `_compute_window_features` logic, simplified:

```python
spy_w   = spy_closes[ti - 62:ti + 1]   # 63 closes
spy_rets = np.diff(np.log(spy_w))      # -> 62 returns
my_rets  = rets[-63:]                  # 63 returns
if len(spy_rets) == len(my_rets) and spy_rets.std() > 0:
    cm = np.cov(my_rets, spy_rets, ddof=1)
    f["beta_spy_63d"] = float(cm[0, 1] / (cm[1, 1] + 1e-12))
else:
    f["beta_spy_63d"] = 0.0            # <- this branch ALWAYS ran
```

Trace the lengths. Slicing `[ti-62 : ti+1]` yields **63** closes. `np.diff(np.log(...))` turns N prices into N−1 returns, so `spy_rets` has **62** elements. But `my_rets = rets[-63:]` has **63**. The guard `len(spy_rets) == len(my_rets)` is `62 == 63`, which is *never* true. Every call fell through to the `else`, and beta was set to `0.0`.

This is dead code by off-by-one. The `if` branch — the entire point of the feature — was unreachable. And the reason nobody noticed for so long is the most insidious part: **the guard didn't crash, it routed to a plausible default.** A `0.0` beta is not absurd. Low-beta stocks exist. A reader sees `0.0` and supplies their own story. The failure mode of a length mismatch was silently disguised as a measurement.

## The second, latent bug: position vs. date

Even if you fixed the off-by-one tomorrow, a deeper bug was waiting. `spy_closes` was indexed by the **ticker's** positional `anchor_idx`. That is, `spy_closes[ti]` meant "the `ti`-th row of SPY's price history," not "SPY's close on the date of this ticker's `ti`-th row."

For a ticker whose history is identical in length and alignment to SPY's, those coincide. For everything else — a stock that IPO'd late, had a trading halt, or simply has a different number of rows than SPY (see [trading calendar](/story/09/37_trading_calendar)) — row `ti` of the stock and row `ti` of SPY are *different dates*. You would be regressing AAPL's June returns against SPY's returns from some unrelated week. This is the canonical financial-data bug: **align by date, never by position.** Two series can share an integer index and still be temporally unrelated — the same theme as [the row-alignment landmine](/story/09/41_row_alignment).

## The fix

Two changes, both required:

```python
# features.py — slice 64 closes -> 63 returns, matching my_rets
spy_w = spy_closes[ti - 63:ti + 1]   # 64 closes -> 63 log-returns
```

```python
# compute_for_anchors — align SPY to THIS ticker's dates first
spy_closes = (spy_by_date.reindex(df["date"].to_numpy())
                         .ffill().bfill()
                         .to_numpy())
```

The slice now produces 63 returns so the guard can pass, and SPY is reindexed onto each ticker's own date axis (forward/back-filled across gaps) *before* being indexed by `ti`. Now `spy_closes[ti]` is genuinely SPY's close on `closes[ti]`'s calendar date.

## Why the tests missed it

The test suite asserted that every feature was **finite** — no NaN, no inf. That is a real and useful invariant. But `0.0` is finite. A feature that is hard-wired to a constant sails through a finiteness check forever. This is the gap in the original [testing philosophy](/story/09/45_testing_philosophy): liveness, not just validity.

The regression test that now guards this is the one I wish had existed from day one. Feed the function a "market" series that **is the stock's own series**. CAPM beta is `Cov(r_i, r_m) / Var(r_m)` (recapped in [SPY beta](/story/09/38_spy_beta)); when `r_i == r_m`, that ratio is exactly `1.0`. So:

```python
beta = compute_beta(stock_closes, spy_closes=stock_closes)
assert abs(beta - 1.0) < 1e-6
```

The buggy code **provably cannot** produce `1.0` — it always returns `0.0`. A second test builds a ticker that shares SPY's closes on shared dates but starts 50 bars later, and asserts the date-aligned beta is ≈1.0 (a position-aligned implementation cannot produce that). These pin specific, derivable values rather than a vague "looks reasonable" band.

## The doc that lied

Now the uncomfortable part. The deep-dive [Market beta — why BNTX's beta was 0.00](/story/09/38_spy_beta) presented `BNTX beta_spy_63d = 0.00` and wrapped a whole economic narrative around it: *BNTX traded on vaccine catalysts, idiosyncratic, uncorrelated with the broad market.* It reads beautifully. It is also **fabricated**, in the precise sense that there was no measurement behind it. BNTX's `0.00` was not BNTX's beta. It was the `else` branch. **Every** stock read `0.00`. The narrative was pattern-matching on an artifact.

That 38_spy_beta page needs a correction banner at the top, stating plainly that the original `0.00` was a bug, not a finding, and pointing here. (It is listed as an action item in the [feature-gaps roadmap](/story/09/47_roadmap).)

This is the entire reason an honest-reproduction project flags its own mistakes loudly instead of quietly patching the code. The [hardening story](/story/09/39_hardening_story) is full of fixes; this is the one with a *narrative cost*. We didn't just ship a wrong number — we *explained* a wrong number, which is far more dangerous, because an explanation lends false confidence. The lesson generalizes: any "interesting result" that lands on a suspiciously round constant deserves a second look before it earns a paragraph. The [engineered features](/story/09/03_engineered_features) overview lists `beta_spy_63d` as a real input; for the entire pre-fix history of this project, it was a zero column wearing a feature's name. Honest reporting means saying so out loud.
