"""Honest statistical reporting.

Replaces the original document's '100th percentile vs 100 random seeds'
headline with three structurally honest reports:

  1. Block-bootstrap p-value on daily returns (block size = horizon),
     non-parametric, autocorrelation-aware.
  2. Deflated Sharpe Ratio (Bailey & López de Prado 2014) accounting for
     the number of model configurations actually tried.
  3. Out-of-sample-only report: freeze the winner, evaluate on the held-
     out last fold, no further selection.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def stationary_block_bootstrap(returns: np.ndarray, n_resamples: int,
                               block_size: int, seed: int = 42) -> np.ndarray:
    """Politis-Romano stationary block bootstrap. Returns Sharpe per resample."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    p = 1.0 / max(block_size, 1)
    out = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        sample = np.empty(n, dtype=np.float64)
        i = 0
        idx = int(rng.integers(0, n))
        while i < n:
            sample[i] = returns[idx % n]
            i += 1
            if rng.random() < p:
                idx = int(rng.integers(0, n))
            else:
                idx += 1
        out[r] = _annualized_sharpe(sample)
    return out


def block_bootstrap_p_value(model_returns: np.ndarray,
                            null_returns: np.ndarray,
                            block_size: int, n_resamples: int = 10000,
                            seed: int = 42) -> dict:
    """Test H0: model Sharpe <= null Sharpe via paired bootstrap."""
    obs_diff = _annualized_sharpe(model_returns) - _annualized_sharpe(null_returns)
    diff_returns = model_returns - null_returns
    boot = stationary_block_bootstrap(diff_returns, n_resamples, block_size, seed)
    p = float((boot <= 0).mean())
    return {
        "obs_sharpe_diff": obs_diff,
        "boot_mean_diff": float(boot.mean()),
        "boot_se": float(boot.std()),
        "p_value": p,
    }


def deflated_sharpe_ratio(observed_sr: float, n_obs: int, n_trials: int,
                          skew: float = 0.0, kurt: float = 3.0,
                          variance_of_trial_sr: float | None = None) -> dict:
    """Bailey & López de Prado deflated Sharpe ratio.

    `variance_of_trial_sr` defaults to the rough heuristic
        Var(SR) ~= (1 - skew*SR + ((kurt-1)/4)*SR**2) / (n - 1)
    when not supplied.
    """
    SR = observed_sr / np.sqrt(252)  # convert to per-period (daily)
    if variance_of_trial_sr is None:
        variance_of_trial_sr = (1.0 - skew * SR + ((kurt - 1) / 4) * SR ** 2) / max(n_obs - 1, 1)
    sigma_max = math.sqrt(variance_of_trial_sr)

    euler_mascheroni = 0.5772156649
    # Expected max of N i.i.d. N(0,1) ~= sqrt(2*ln(N)) - (sqrt(2*ln(N)))^-1*(γ + ln(ln(N)))
    if n_trials <= 1:
        e_max = 0.0
    else:
        a = math.sqrt(2.0 * math.log(n_trials))
        e_max = a - (math.log(math.log(n_trials)) + euler_mascheroni) / a if n_trials > 2 else a

    SR_threshold = e_max * sigma_max
    psr = stats.norm.cdf((SR - SR_threshold) * math.sqrt(max(n_obs - 1, 1)) /
                          math.sqrt(max(1.0 - skew * SR + ((kurt - 1) / 4) * SR ** 2, 1e-9)))
    return {
        "observed_sharpe_annualized": observed_sr,
        "observed_sharpe_daily": SR,
        "n_trials": n_trials,
        "n_observations": n_obs,
        "expected_max_sharpe_under_null": e_max * sigma_max * np.sqrt(252),
        "deflated_sharpe_threshold_annualized": SR_threshold * np.sqrt(252),
        "deflated_sharpe_p_value_significant": float(psr),
    }


def percentile_vs_distribution(model_value: float, null_distribution: np.ndarray) -> float:
    return float((null_distribution <= model_value).mean())


def report_oos_only(equity_oos: pd.DataFrame, blotter_oos: pd.DataFrame,
                    label: str) -> dict:
    from .simulator import compute_summary, SimConfig
    summary = compute_summary(equity_oos, blotter_oos, SimConfig())
    summary["label"] = label
    summary["report_type"] = "oos_only_last_fold"
    return summary


def daily_returns_from_equity(equity: pd.DataFrame) -> np.ndarray:
    if len(equity) < 2:
        return np.array([])
    return equity["equity"].pct_change().dropna().to_numpy()


# ─────────────────────────────────────────────────────────────────────────
# Honest reporting additions (per 60-subagent research synthesis P3)
# ─────────────────────────────────────────────────────────────────────────


def effective_sample_size(blotter_df: pd.DataFrame, hold_days: int = 40) -> float:
    """López de Prado average uniqueness.

    For each trade i, uniqueness_i = mean(1/concurrency over its hold days).
    Effective N = sum(uniqueness). For our 1,104 trades with ~25 concurrent
    positions, returns ~44 — not 1,104. The 5x widening of SE(Sharpe)
    collapses many naive-significant results.
    """
    if blotter_df is None or len(blotter_df) == 0:
        return 0.0
    df = blotter_df.copy()
    if "exit_date" not in df.columns and "entry_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["entry_date"]) + pd.Timedelta(days=hold_days)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    # Defensive: drop rows with NaT dates that would crash pd.date_range
    df = df.dropna(subset=["entry_date", "exit_date"])
    if len(df) == 0:
        return 0.0
    days = pd.date_range(df["entry_date"].min(), df["exit_date"].max(), freq="B")
    if len(days) == 0:
        return 0.0
    concurrency = pd.Series(0.0, index=days)
    for _, r in df.iterrows():
        concurrency.loc[r["entry_date"]:r["exit_date"]] += 1.0
    uniqueness = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        slice_ = concurrency.loc[r.entry_date:r.exit_date]
        if len(slice_) == 0:
            uniqueness[i] = 0.0
        else:
            uniqueness[i] = (1.0 / slice_.replace(0, np.nan)).mean()
    return float(np.nansum(uniqueness))


def bootstrap_cagr_ci(equity_df: pd.DataFrame, n_resamples: int = 10000,
                       block_size: int = 40, seed: int = 42) -> dict:
    """Stationary block bootstrap CI on CAGR, Sharpe, max drawdown.

    Resamples daily log-returns of the equity curve (NOT trade returns —
    trades overlap). Block_size defaults to the holding horizon = the
    natural autocorrelation length of overlapping holds.
    """
    eq = equity_df["equity"].to_numpy(dtype=np.float64)
    if len(eq) < 5:
        return {"cagr_ci": (0, 0, 0), "sharpe_ci": (0, 0, 0), "maxdd_ci": (0, 0, 0)}
    r = np.diff(np.log(eq))
    n = len(r)
    p = 1.0 / max(block_size, 1)
    yrs = max(n / 252.0, 1e-9)
    cagrs = np.empty(n_resamples)
    sharpes = np.empty(n_resamples)
    mdds = np.empty(n_resamples)
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = np.empty(n, dtype=np.int64)
        t = 0
        while t < n:
            start = int(rng.integers(0, n))
            L = int(rng.geometric(p))
            take = min(L, n - t)
            for k in range(take):
                idx[t + k] = (start + k) % n
            t += take
        rs = r[idx]
        eq_s = np.exp(np.cumsum(rs))
        cagrs[i] = float(eq_s[-1] ** (1.0 / yrs) - 1.0)
        sharpes[i] = (rs.mean() / rs.std() * np.sqrt(252)) if rs.std() > 0 else 0.0
        peak = np.maximum.accumulate(eq_s)
        mdds[i] = float((eq_s / peak - 1.0).min())
    q = lambda x: tuple(float(v) for v in np.quantile(x, [0.025, 0.5, 0.975]))
    return {
        "cagr_ci": q(cagrs),
        "sharpe_ci": q(sharpes),
        "maxdd_ci": q(mdds),
        "n_resamples": int(n_resamples),
        "block_size": int(block_size),
    }


def reality_check_spa(returns_matrix: np.ndarray,
                      benchmark_returns: np.ndarray | None = None,
                      block_size: int = 10, n_resamples: int = 5000,
                      seed: int = 42) -> dict:
    """Hansen's Superior Predictive Ability (SPA) test via `arch.bootstrap`.

    `returns_matrix` shape: [T, K] where K = number of strategies tried.
    Tests H0: max_k E[returns_k] <= E[benchmark]. Replaces the broken
    "best of K beats random p95" framing.
    """
    try:
        from arch.bootstrap import SPA
    except ImportError:
        return {"error": "arch package not installed; pip install arch"}
    R = np.asarray(returns_matrix, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] < 50:
        return {"error": f"expected [T,K] with T>=50; got {R.shape}"}
    if benchmark_returns is None:
        bench = np.zeros(R.shape[0])
    else:
        bench = np.asarray(benchmark_returns, dtype=np.float64)
    losses = -(R - bench[:, None])
    bench_loss = np.zeros_like(bench)
    spa = SPA(bench_loss, losses, block_size=block_size, reps=n_resamples,
              studentize=True, bootstrap="stationary", seed=seed)
    spa.compute()
    pv = spa.pvalues
    return {
        "p_lower": float(pv["lower"]),
        "p_consistent": float(pv["consistent"]),  # Hansen's recommended
        "p_upper": float(pv["upper"]),
        "n_strategies": int(R.shape[1]),
        "n_obs": int(R.shape[0]),
        "block_size": int(block_size),
        "n_resamples": int(n_resamples),
    }


def post_tax_cagr(blotter: pd.DataFrame, federal_rate: float = 0.32,
                   state_rate: float = 0.06, niit_rate: float = 0.038,
                   start_equity: float = 100_000.0) -> dict:
    """Post-tax CAGR for short-term-cap-gains strategy (40d holds → 100% STCG).

    Default rates approximate a $200k California earner. Adjust per user.
    Loss carryforward and $3k/yr offset ignored — conservative assumption.
    """
    df = blotter.copy()
    if len(df) == 0 or "exit_date" not in df.columns:
        return {"pre_tax_cagr": 0.0, "post_tax_cagr": 0.0, "blended_rate": 0.0,
                "total_tax": 0.0, "years": 0.0}
    if "trade_return_pct" in df.columns and "cost" in df.columns:
        df["pnl_dollars"] = df["trade_return_pct"] * df["cost"]
    elif "proceeds" in df.columns and "cost" in df.columns:
        df["pnl_dollars"] = df["proceeds"] - df["cost"]
    else:
        return {"error": "blotter missing pnl columns"}

    df["year"] = pd.to_datetime(df["exit_date"]).dt.year
    rate = federal_rate + state_rate + niit_rate
    equity = start_equity
    tax = 0.0
    for year, g in df.groupby("year"):
        gross = float(g["pnl_dollars"].sum())
        y_tax = max(gross, 0.0) * rate
        equity += gross - y_tax
        tax += y_tax
    years = float(df["year"].nunique() or 1)
    pre = float((1 + df["pnl_dollars"].sum() / start_equity) ** (1 / years) - 1)
    post = float((equity / start_equity) ** (1 / years) - 1)
    return {
        "pre_tax_cagr": pre,
        "post_tax_cagr": post,
        "blended_rate": rate,
        "total_tax": tax,
        "years": years,
    }


def trial_count_from_registry(registry_path) -> int:
    """Count unique (model, label, horizon, threshold, universe) tuples logged."""
    from pathlib import Path as _P
    p = _P(registry_path)
    if not p.exists():
        return 1
    seen = set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = __import__("json").loads(line)
            key = (d.get("model"), d.get("label"), d.get("horizon_d"),
                   d.get("threshold_q"), d.get("universe"))
            seen.add(key)
        except Exception:
            continue
    return max(len(seen), 1)

