"""Random and simple-momentum baselines under identical portfolio rules.

Five simple, deterministic baselines that the original document never
compared against. The ML model must beat the BEST of these to justify
its complexity:

  1. rank_252d_return
  2. rank_60d_return
  3. rank_ma250_extension   (Close / MA250 ratio)
  4. rank_52w_high_distance (closeness to 252d high)
  5. rank_inv_60d_vol       (low-vol within momentum)

Plus the doc's existing comparison: random selection (1000 seeds).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .simulator import simulate, SimConfig


def _score_from_features(features: pd.DataFrame, baseline: str) -> pd.Series:
    if baseline == "rank_252d_return":
        return features["ret_252d"]
    if baseline == "rank_60d_return":
        return features["ret_63d"]
    if baseline == "rank_ma250_extension":
        return features["ratio_ma200"]
    if baseline == "rank_52w_high_distance":
        return -features["pct_from_252d_high"].abs()
    if baseline == "rank_inv_60d_vol":
        return -features["vol_63d"]
    raise ValueError(f"unknown baseline: {baseline}")


def run_simple_baseline(features: pd.DataFrame, anchor_meta: pd.DataFrame,
                        price_lookup: dict, baseline: str, cfg: SimConfig,
                        horizon: int) -> dict:
    """Run a deterministic simple-rank baseline.

    `features` must include the FEATURE_COLS named above plus
    [ticker, anchor_date]. `anchor_meta` provides anchor_close,
    adv20_usd, anchor_idx, resolution_date_{H}d for each
    (ticker, anchor_date) row.
    """
    score = _score_from_features(features, baseline)
    df = features[["ticker", "anchor_date"]].copy()
    df["score"] = score.to_numpy()
    df = df.merge(
        anchor_meta[["ticker", "anchor_date", "anchor_close", "adv20_usd",
                     "anchor_idx", f"resolution_date_{horizon}d"]],
        on=["ticker", "anchor_date"], how="inner",
    )
    df = df.rename(columns={f"resolution_date_{horizon}d": "resolution_date"})
    return simulate(df, price_lookup, cfg)


def run_random(anchor_meta: pd.DataFrame, price_lookup: dict, cfg: SimConfig,
               horizon: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    df = anchor_meta.copy()
    df["score"] = rng.random(len(df))
    df = df.rename(columns={f"resolution_date_{horizon}d": "resolution_date"})
    return simulate(df[["ticker", "anchor_date", "score", "anchor_close",
                        "adv20_usd", "anchor_idx", "resolution_date"]],
                    price_lookup, cfg)


def run_random_seeds(anchor_meta: pd.DataFrame, price_lookup: dict, cfg: SimConfig,
                     horizon: int, n_seeds: int = 1000, base_seed: int = 1000) -> pd.DataFrame:
    """Run n_seeds random portfolios; return a summary DataFrame, one row per seed."""
    rows = []
    for i in range(n_seeds):
        out = run_random(anchor_meta, price_lookup, cfg, horizon, seed=base_seed + i)
        s = out["summary"]
        s["seed"] = base_seed + i
        rows.append(s)
    return pd.DataFrame(rows)


SIMPLE_BASELINES = [
    "rank_252d_return",
    "rank_60d_return",
    "rank_ma250_extension",
    "rank_52w_high_distance",
    "rank_inv_60d_vol",
]
