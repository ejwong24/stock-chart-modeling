# The blotter, the equity curve, and the summary — how a backtest result is shaped

The [simulator](/story/09/06_simulator_loop) at `src/stock_chart/simulator.py` returns a single dict with exactly three keys: `blotter`, `equity`, `summary`. Everything downstream — the web UI, the [bootstrap CIs](/story/09/22_block_bootstrap_params), the [report cards](/story/04_costs) — is just a view over those three artifacts.

## The blotter — the trade tape

`out['blotter']` is a DataFrame with **one row per round-trip trade**:

- `ticker` (str): the stock
- `entry_date` (datetime): when we bought
- `exit_date` (datetime): when we sold (may equal horizon resolution OR be earlier due to [trailing stop](/story/09/11_trailing_stop_interactions))
- `entry_price` (float): fill price including buy slippage
- `exit_price` (float): fill price including sell slippage
- `shares` (int): integer shares bought (the [position sizer](/story/09/32_position_size_formula) rounds down)
- `cost` (float): cash spent on entry — `shares × entry_price + commission`
- `proceeds` (float): cash received on exit
- `commission_total` (float): sum of entry + exit commissions
- `trade_return_pct` (float): `(proceeds - cost) / cost`
- `exit_slippage_bps` (float): the bps slippage on the SELL side
- `forced_close` (bool, optional): True if force-closed at end-of-data

Typical size: ~1,100 rows for a 8-year backtest with 25 simultaneous positions.

## The equity curve — the portfolio-level view

`out['equity']` is a DataFrame with **one row per trading day**:

- `date` (datetime): the trading day
- `equity` (float): total portfolio value = `cash + sum(shares × today's close)` for all open positions
- `cash` (float): unspent cash
- `n_positions` (int): how many positions are open at end-of-day

Typical size: ~2,000 rows for an 8-year backtest.

## The summary — the scalar reduction

`out['summary']` is a single flat dict of scalar metrics:

- `start_equity`, `end_equity`, `total_return`
- `years` (calendar years elapsed)
- `cagr` — see [CAGR + drawdown](/story/09/28_cagr_drawdown_calmar)
- `sharpe` — see [Sharpe ratio](/story/09/27_sharpe_ratio)
- `max_dd`
- `n_trades`, `winrate`, `avg_trade_return`
- Sortino, [Calmar](/story/09/28_cagr_drawdown_calmar), etc.

## How they relate

The blotter is the **trade tape** — what we bought and sold. The equity curve is the **portfolio-level value series** — what we'd see if we marked-to-market every day. The summary is the **scalar reduction** of both.

You can recompute the summary from the other two, but you cannot reconstruct the equity curve from just the blotter (you'd lose the unrealized-MTM path between trades).

## The conservation invariant

From the [Simulator inner loop](/story/09/06_simulator_loop):

```
At every step:
  equity[t] = cash[t] + sum(position.shares × close[ticker, t]) for open positions
At end-of-backtest (all positions force-closed):
  cash[final] = equity[final]
  Σ(blotter.proceeds - blotter.cost) ≈ equity[final] - start_equity
                                          (within commissions)
```

## How the web UI uses these

- `/runs/{tag}` displays the summary as a table + the equity curve as a Plotly chart
- `/api/runs/{tag}/equity/{track}` returns the equity DataFrame as JSON
- `/api/runs/{tag}/blotter/{track}` returns the blotter as JSON

## Persistence

After a run, each track gets:

- `reports/{tag}/blotter_{track}.parquet`
- `reports/{tag}/equity_{track}.parquet`
- `reports/{tag}/scores_{track}.parquet` (the model's pre-simulator probability outputs)
- `reports/{tag}/headline.json` (aggregated summary across tracks)

## The bootstrap CI plumbing

`stats.bootstrap_cagr_ci(equity_df)` reads the equity DataFrame, computes daily [log returns](/story/09/35_log_vs_simple_returns) via `np.diff(np.log(equity))` (see [Log vs simple returns](/story/09/35_log_vs_simple_returns)), and runs the [stationary block bootstrap](/story/09/22_block_bootstrap_params) to produce 95% CIs.

## Debugging starts with the blotter

If the headline CAGR looks off, load `reports/full/blotter_lgbm_engineered.parquet` and look at:

- **`n_trades`**: should be ~1,100. If it's 200 or 5,000, something in the picker or the stop logic is wrong.
- **`winrate`**: should be ~30-40% for a momentum strategy. Higher than 50% is a red flag for [walk-forward leakage](/story/09/08_walkforward_embargo).
- **`mean(trade_return_pct)`**: positive but small (~+2%).
- **`max(trade_return_pct)`**: the biggest single winner (typically 100-300% for a momentum strategy).

The summary is what you show people. The equity curve is what you chart. But the blotter is where you actually find the bug.
