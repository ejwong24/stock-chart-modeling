# Problem 5 — Path-dependent exits + relabeling

## Synthesis (top action)

**First-week change (4 hours, no relabeling required):**

Add a **volatility-scaled trailing stop** in `simulator.py`. Track running max of close since entry, exit when `close < max_close - k × ATR_20`. Calibrate `k` on a held-out fold (grid {1.5, 2.0, 2.5, 3.0, 3.5}). Existing `ret_40d_ge_25pct` head still trains; only simulator's realized P&L changes. Captures 60-70% of the Sharpe lift available from path-dependent exits — without the relabeling cost.

**Six-month migration to gold standard:**

| Week | Add | Effort |
|---|---|---|
| 1 | Vol-scaled trailing stop + drawdown floor in simulator | 4 hrs |
| 1 | Calibrate stop level on held-out fold (counts as N_trials += 6) | 2 hrs |
| 3 | Triple-barrier labels in labels.py (parallel to legacy binary) — leverage existing `fwd_mfe_{H}d` / `fwd_mae_{H}d` columns | 3 hrs |
| 6 | Multi-class head trained on triple-barrier label, paper-traded vs binary | 1.5 days |
| Mo 3 | Cut over primary signal to multi-class if it dominates 2 CV folds | — |
| Mo 5-6 | Position sizing as secondary head | — |

**Estimated Sharpe lift over fixed-horizon exit: +0.25–0.40** (mostly from tail truncation, ~25-40% smaller max DD; not from picking better names).

**Drop:**
- RL hold/sell agent (sample efficiency + credit assignment + hyperparameter fragility = wrong tool)
- Sector-relative stops (extra data, marginal benefit over trailing stop)
- Volume-collapse exit (single-stock SNR is poor)

**Keep `ret_40d_ge_25pct` head as legacy benchmark forever** — it's the regression-test anchor. Removing it kills your ability to detect silent regressions.

---

## A1 — Volatility-scaled trailing stop (HIGH tractability — simulator-only)

Recommended starter: **fixed-percent trailing stop @ 18% from running peak**, simpler than ATR-scaling for v1.

```python
TRAIL_DD = 0.18
SLIPPAGE_BPS = 5
for pos in list(open_positions):
    pos.peak_close = max(pos.peak_close, bars.close[pos.ticker, t])
    dd = bars.close[pos.ticker, t] / pos.peak_close - 1.0
    if dd <= -TRAIL_DD and not pos.stop_armed:
        pos.stop_armed = True; continue   # arm; exit at NEXT open (no look-ahead)
    if pos.stop_armed:
        exit_px = bars.open[pos.ticker, t] * (1 - SLIPPAGE_BPS / 1e4)
        pos.close(exit_px, t, reason="trailing_stop")
        open_positions.remove(pos); continue
    if (t - pos.entry_t) >= 40:
        pos.close(bars.close[pos.ticker, t], t, reason="horizon_40d")
        open_positions.remove(pos)
```

**Honest effect:** Max DD: 35-45% relative reduction (-35% → -22%). CAGR: -1 to -3 pp. Sharpe: roughly flat. **Calmar (CAGR/MaxDD) clearly improves** — this is the right metric.

Sweep {0.15, 0.18, 0.20, 0.25} on Calmar, not Sharpe.

## A2 — Triple-barrier (López de Prado) (HIGH tractability)

Each anchor labeled by which barrier hits FIRST: profit_target (+T·σ), stop_loss (-S·σ), or time_limit (+H days). 3-class label.

```python
def triple_barrier_label(prices, anchor_idx, pt_mult, sl_mult, horizon, sigma):
    p0 = prices.iloc[anchor_idx]
    upper = p0 * (1 + pt_mult * sigma); lower = p0 * (1 - sl_mult * sigma)
    end = min(anchor_idx + horizon, len(prices) - 1)
    window = prices.iloc[anchor_idx + 1: end + 1]
    hit_up = window >= upper; hit_dn = window <= lower
    t_up = hit_up.idxmax() if hit_up.any() else None
    t_dn = hit_dn.idxmax() if hit_dn.any() else None
    if t_up is None and t_dn is None: return 0   # expired
    if t_up is None: return -1                    # stop first
    if t_dn is None: return 1                     # target first
    return 1 if t_up <= t_dn else -1
```

LightGBM `objective="multiclass", num_class=3`. Score = `P(target) - P(stop)` (bounded [-1,1], well-behaved when one class collapses, asymmetric weighting if TP/SL ratio is asymmetric: `2*P(target) - P(stop)` for 2R:1R).

Asymmetric defaults `pt=2.0, sl=1.0, H=40` per AFML baseline. Per-ticker σ scaling makes barriers volatility-adaptive without tuning.

**Expected lift:** +0.3 absolute Sharpe, ~30% MaxDD reduction. Most of win from simulator-label alignment (no more leakage from fixed-horizon label vs stop-loss simulator), not the multi-class model itself.

## A3 — Continuous regression label (HIGH — recommended default)

Replace `ret_40d_ge_25pct` with `ret_40d` (continuous) or `fwd_log_ret_40d`. Eliminates threshold-induced signal loss (a +28% pick and a +65% pick have same binary label). Class imbalance disappears. Better cross-regime calibration. Robust loss handles fat tails.

```python
def build_regressor(kind="lgbm", huber_delta=0.10):
    if kind == "lgbm":
        return lgb.LGBMRegressor(objective="huber", alpha=huber_delta,
                                  n_estimators=800, learning_rate=0.03, num_leaves=63,
                                  min_data_in_leaf=200, feature_fraction=0.8)
```

**Make regression default. Keep classifier behind `task="classification"` for leaderboard parity.** This is a one-evening change.

## A4 — RL hold/sell agent (LOW — DON'T DO THIS)

Category error. We have 91k anchor-week trajectories ≈ 3.6M decision steps; PPO needs millions of env steps to converge. Sparse reward over 40d horizon = high-variance credit assignment. Hyperparameter fragility (GAE λ, clip ratio, entropy coef, LR schedule) all interact, reproducibility across seeds famously poor.

Triple-barrier (A2) and trailing stop (A1) capture 95% of the policy-improvement value with 1% of the engineering. The honest "good RL question" — does a learned policy beat a hand-tuned trailing stop? — can be answered with a 100-line contextual-bandit experiment over discrete exit thresholds, not a Gymnasium environment.

## A5 — Calibrate stop on held-out fold (HIGH tractability)

```python
STOP_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, None]
PRIOR = {0.05:0.5, 0.10:0.9, 0.15:1.0, 0.20:0.7, 0.25:0.4, None:0.3}  # gauss-around-15%

def calibrate_stop(folds):
    for train, val, test in folds:
        model = fit_model(train)
        sharpe = {s: simulate(model, val, stop_loss=s).sharpe for s in STOP_GRID}
        scored = {s: sharpe[s] * PRIOR[s] for s in STOP_GRID}  # MAP
        fold_choice = max(scored, key=scored.get)
    chosen = mode(per_fold_choices)  # rank-robust
    return chosen
```

**Yes, this counts as N_trials += 6 for deflated Sharpe.** Pretending N_trials=1 because "we selected on validation" is leakage — the prior was hand-picked from domain intuition encoding prior backtests.

Reporting: "Sharpe 1.2-1.8 across 6 stop levels; selected 15% (val 1.65, test 1.4); deflated Sharpe with N=6 is 1.1."

## A6 — Drawdown-control per-trade stop (HIGH tractability — simplest)

Fixed: exit if `close[t] / entry_price ≤ (1 - X)`, X ∈ {10%, 15%, 20%}. Differs from A1 (trailing) — fixed-entry stop only protects vs downside from cost basis; trailing also locks in gains.

**Recommendation:** 15% with sensitivity sweep. Combine with time stop + profit target = mini triple-barrier WITHOUT relabeling.

```python
STOP_PCT = 0.15
for ticker, pos in list(positions.items()):
    px = close[today, ticker]
    if px / pos.entry_price - 1.0 < -STOP_PCT:
        exit_px = open_[next_day, ticker] * (1 - SLIPPAGE_BPS / 1e4)
        record_trade(ticker, pos, exit_px, reason="stop_loss")
        del positions[ticker]
```

Three notes: (1) check stop BEFORE signal exits; (2) close-to-stop trigger but next-day-open execution to avoid look-ahead; (3) tag exits with reason for attribution split.

## A7 — Leverage existing mfe/mae (HIGH — zero new compute)

`labels.py` already emits `fwd_mfe_{H}d` and `fwd_mae_{H}d`. Add 5 new label families per horizon — pure column algebra:

```python
EPS = 1e-4
for H in [5, 10, 20, 40]:
    mfe, mae, ret = df[f"fwd_mfe_{H}d"], df[f"fwd_mae_{H}d"], df[f"fwd_ret_{H}d"]
    abs_mae = mae.abs().clip(lower=EPS)
    # 1. Approximate triple-barrier (no ordering)
    for T, S in [(0.02, 0.02), (0.05, 0.02), (0.05, 0.05), (0.10, 0.05)]:
        df[f"tb_approx_{H}d_T{int(T*100)}_S{int(S*100)}"] = ((mfe >= T) & (mae > -S)).astype("int8")
    # 2. Risk-adjusted asymmetry
    df[f"mfe_mae_ratio_{H}d"] = mfe / abs_mae
    # 3. Sortino-style: realized upside per unit drawdown
    df[f"sortino_label_{H}d"] = ret.clip(lower=0) / abs_mae
    # 4. Path quality
    df[f"upside_dominance_{H}d"] = (mfe > abs_mae).astype("int8")
    # 5. Clean run
    df[f"clean_run_{H}d"] = ((mfe >= 0.05) & (mae > -0.02)).astype("int8")
```

The approximate triple-barrier (no first-touch ordering) is biased toward 1 — anchors where stop hit first then target later get labeled wins. On daily bars over short H this miscount is small. Treat as screening label; escalate to true first-touch walk only if model performance suggests path-order matters.

## A8 — Position sizing as secondary head (MED — opt-in)

Four options ranked:
1. **Vol-scaling** (recommended first): `size = base / σ_20d`. Robust, well-documented (AQR, Asness). Counter-momentum risk mild at weekly rebalance.
2. **Softmax conviction**: defensible if classifier calibrated. Requires reliability-diagram check first; tree models often overconfident on tails.
3. **Two-head model**: best in theory but adds training complexity, regression head noisy on weekly returns.
4. **Kelly fractional**: mathematically elegant, empirically punishing on 5-name books with noisy edge estimates. **Skip.**

Hidden virtue of equal-weight: Bayesian shrinkage toward "we don't really know which of our 5 picks is best." Concentration assumes you DO know. Need a lot of OOS data to prove conviction signal beats uniform.

```python
def compute_position_sizes(picks, equity, max_pct, mode="equal"):
    if mode == "vol_scaled":
        inv_vol = {p.ticker: 1.0 / max(p.sigma_20d, 0.005) for p in picks}
        total = sum(inv_vol.values())
        budget = equity * max_pct * len(picks)  # preserve gross exposure
        return {t: budget * w / total for t, w in inv_vol.items()}
    return {p.ticker: equity * max_pct for p in picks}  # equal default
```

Wire via `--sizing {equal, vol_scaled, conviction}` flag. Default equal until vol-scaled beats by ≥5% Sharpe net of costs over 3+ years.

**Sharpe lift estimate:** vol-scaling +8-12%; conviction (if calibrated) +5-10% but **0 to -15% if not**.

## A9 — Sector-relative stops (LOW-MED — skip, prefer trailing)

Mar-2020 case is real (15% absolute stop liquidated 40% of positions, most recovered by June). But strong-sector / lagging-name rallies trip relative stop on positive-P&L positions. Vol-scaled trailing stop already solves most of this — widens during high-vol drawdowns AND ratchets up during rallies — without extra data.

Sector-relative adds maintenance burden (sector_lookup.csv, sector ETF panel) for typically 20-40 bps Sharpe lift over a well-tuned trailing stop. Not worth it as primary stop. Revisit only if trailing-stop benchmark shows clear sector-beta failure mode.

## A10 — Volume-collapse exit (LOW-MED — skip)

Single-stock daily volume dominated by idiosyncratic noise (rebalances, OpEx, blocks). SNR ~10x worse than at index aggregate. Failure modes: holiday weeks trigger across portfolio simultaneously; post-earnings volume always collapses; low-float momentum spikes/crashes.

**Better volume formulations IF you insist:** relative volume vs SPY; dollar-volume trend slope OLS; OBV/A-D divergence. Skip for v1, opt-in flag in v2 only if backtests suggest residual edge after simpler stops exhausted.

## A11 — Multi-class outcome model architecture (HIGH tractability)

Skip ordinal regression for v1 — equal-log-odds spacing assumption isn't true. Flat 3-class softmax lets model learn geometry, recover ordering via score function.

```python
params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
          "learning_rate": 0.03, "num_leaves": 63, "feature_fraction": 0.85}
model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=2000,
                  callbacks=[lgb.early_stopping(50)])
proba = model.predict(X_test)
score = proba[:, 0] - proba[:, 1]   # P(target) - P(stop), bounded [-1, 1]
```

**Calibration:** multiclass LightGBM is reasonably calibrated OOB; if you'll threshold the score (e.g., `score > 0.15` filter), wrap with `CalibratedClassifierCV(method="isotonic", cv="prefit")` on a held-out fold. Don't calibrate on training fold.

Trade simulator already respects barriers natively, so model's only job is ranking. Expired trades correctly get middle scores (low magnitude either sign), matching their economic role.

Ship multiclass alongside binary as fallback; retire binary only after one walk-forward confirms score-AUC actually improves.
