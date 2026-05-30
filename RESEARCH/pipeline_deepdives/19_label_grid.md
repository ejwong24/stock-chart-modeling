# The label grid — 36 binary labels, plus mfe/mae/path-dependent variants

The labeling layer in `src/stock_chart/labels.py` is where raw OHLCV becomes supervised-learning targets. For every `(ticker, anchor_date)` row, we emit a dense bundle of labels: 36 binary outcome flags arranged on a horizon × threshold grid, plus a family of path-dependent excursion labels per horizon, plus bookkeeping columns the downstream splitter needs to stay leak-free.

## Naming convention

Every binary label follows the same shape:

```
ret_{H}d_ge_{T*100}pct = 1  iff  fwd_ret_{H}d >= T
```

where `fwd_ret_{H}d = anchor_close_at(t+H) / anchor_close_at(t) - 1`. So `ret_40d_ge_25pct = 1` exactly when the forward 40-trading-day simple return is at least +25%.

For BNTX on 2021-07-02:

| Column | Value |
|---|---|
| `fwd_ret_40d` | **+0.5394** |
| `ret_40d_ge_5pct` | 1 |
| `ret_40d_ge_25pct` | **1** |
| `ret_40d_ge_30pct` | 1 |

That single anchor lights up every cell at H=40 from 5% through 30% — a clean positive.

## Why six horizons

`H ∈ {5, 10, 15, 20, 30, 40}` trading days, each mapping to a natural trading style:

| Horizon | Style it tests |
|---|---|
| 5d | "Does momentum continue this week?" |
| 10d | Two-week follow-through |
| 15d, 20d | Classic swing-trade window |
| 30d | Position trade / earnings-cycle |
| 40d | The canonical "40-day hold" from the original document |

## Why six thresholds

`T ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}`. The threshold answers "what counts as success?"

- **5% in 40 days** is achievable for ~30% of stocks
- **30% in 40 days** is achievable for ~3% — a rare big winner

## The 6×6 grid

Cross-product gives 36 binary labels. Why 36?

1. **Faithful reproduction.** The original document tested 6×6 = 36.
2. **Coverage.** Each `(H, T)` cell is a *different* classification problem.
3. **[Deflated Sharpe](/story/09/10_deflated_sharpe) trial count.** Clean `N_trials = 36`.

## MFE / MAE: peak excursion labels

For each `H`:

```
fwd_mfe_{H}d = max(0,   max( close[t+1..t+H] / anchor_close - 1 ))
fwd_mae_{H}d = min(0,   min( close[t+1..t+H] / anchor_close - 1 ))
```

**Sign convention (Bug #2 fix from the audit):**

- `fwd_mfe_{H}d >= 0` — peak unrealized *gain*
- `fwd_mae_{H}d <= 0` — peak unrealized *loss*

For BNTX on 2021-07-02 at H=40:

| Column | Value | Interpretation |
|---|---|---|
| `fwd_ret_40d` | +0.5394 | finished +54% |
| `fwd_mfe_40d` | **+1.00** | touched +100% during the hold |
| `fwd_mae_40d` | **-0.05** | worst drawdown from anchor was only -5% |

BNTX is the dream shape: huge upside excursion, almost no downside.

## Path-dependent labels

| Label | Formula | What it captures |
|---|---|---|
| `tb_approx_{H}d_T{T}_S{S}` | `(mfe >= T) & (mae > -S)` | Triple-barrier approximation |
| `mfe_mae_ratio_{H}d` | `mfe / max(abs(mae), 1e-4)` | Risk-adjusted excursion |
| `sortino_label_{H}d` | `max(ret,0) / max(abs(mae), 1e-4)` | Downside-risk-adjusted return |
| `upside_dominance_{H}d` | `mfe > abs(mae)` | Was the up move bigger than the down? |
| `clean_run_{H}d` | `(mfe >= 0.05) & (mae > -0.02)` | Trended up cleanly |

`EPS = 1e-4` is the floor in divisors to avoid divide-by-zero.

For BNTX 2021-07-02 at H=40: `mfe=1.00, mae=-0.05` gives `mfe_mae_ratio_40d ≈ 20.0` and `clean_run_40d = 1` — a textbook clean run.

## Why both threshold and path-dependent

A `ret_40d_ge_25pct = 1` label is silent about *how* the gain happened. The same +25% endpoint can come from:

- a smooth trend with -3% max drawdown, or
- a roller coaster that drew down -20% before rallying.

Psychologically and stop-loss-wise, those are entirely different trades.

## `resolution_date_{H}d` — the leakage guard

For each label, we also store `resolution_date_{H}d` — the calendar date on which the label becomes *known*. `splits.py` uses this to enforce purging: any training row whose `resolution_date_{H}d >= test_fold_start` is dropped.

## `prefilter_ma250_15`

A single boolean: `1` iff `close > 1.5 * SMA250` on anchor_date. Cheap precomputed regime gate.

---

> See [/story/09/06](/story/09/06) for how the simulator consumes these labels, and [/story/09/11](/story/09/11) for how the [trailing stop](/story/09/11_trailing_stop_interactions) tracks MFE/MAE-like signals.
