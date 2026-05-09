"""Purged walk-forward splits with embargo (López de Prado).

Fix for the original document's HIGH-severity walk-forward leakage flaw:
weekly anchors near a year-end have forward-H labels that resolve INTO
the test fold. Without purging the test-fold prices are baked into
training labels even though the features are clean.

This module:
  1. Defines test folds = calendar years (configurable).
  2. Embargo = `embargo_horizon_multiplier * H` trading days.
  3. Drops any training row where label_resolution_date overlaps the
     purge zone [test_start, test_end + embargo_days].
  4. Asserts no leakage post-split (in tests).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FoldSpec:
    test_year: int
    horizon: int
    embargo_days: int


def _yearly_bounds(df: pd.DataFrame, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(f"{year}-01-01")
    e = pd.Timestamp(f"{year}-12-31")
    return s, e


def split(df: pd.DataFrame, fold: FoldSpec,
          anchor_col: str = "anchor_date",
          resolution_col_template: str = "resolution_date_{H}d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) DataFrames for the given fold spec.

    `df` must contain `anchor_col` and the per-horizon resolution column
    formed by `resolution_col_template.format(H=fold.horizon)`.
    """
    res_col = resolution_col_template.format(H=fold.horizon)
    if res_col not in df.columns:
        raise KeyError(f"missing column {res_col} for horizon {fold.horizon}")

    test_start, test_end = _yearly_bounds(df, fold.test_year)

    # Expanding-window walk-forward: training data must be strictly BEFORE the
    # test fold (no future leakage), then purged near the boundary.
    is_test = (df[anchor_col] >= test_start) & (df[anchor_col] <= test_end)
    is_strictly_before = df[anchor_col] < test_start
    train_pool = df[is_strictly_before].copy()

    # Purge: drop training rows whose forward window resolves on/after test_start.
    # Embargo extends the purge window backward by embargo_days additional trading days.
    embargo = pd.Timedelta(days=int(fold.embargo_days * 1.6))
    overlap = train_pool[res_col] >= (test_start - embargo)
    train = train_pool[~overlap].copy()
    test = df[is_test].copy()
    return train, test


def yearly_walk_forward(df: pd.DataFrame, horizon: int,
                        test_years: list[int],
                        embargo_horizon_multiplier: int = 1,
                        anchor_col: str = "anchor_date",
                        resolution_col_template: str = "resolution_date_{H}d",
                        lockbox_year: int | None = None,
                        unlock_lockbox: bool = False) -> dict:
    """Build {year -> (train, test)} mapping with purged splits.

    If `lockbox_year` is set, that year is excluded from `test_years` unless
    `unlock_lockbox=True`. This implements problem-3 lockbox protocol.
    """
    out = {}
    embargo_days = horizon * embargo_horizon_multiplier
    effective_years = [y for y in test_years if y != lockbox_year]
    if unlock_lockbox and lockbox_year is not None:
        effective_years.append(lockbox_year)
    elif lockbox_year is not None and lockbox_year in test_years:
        # Caller passed lockbox year explicitly without unlock flag — drop silently
        pass
    for y in effective_years:
        spec = FoldSpec(test_year=y, horizon=horizon, embargo_days=embargo_days)
        tr, te = split(df, spec, anchor_col=anchor_col,
                       resolution_col_template=resolution_col_template)
        if len(tr) == 0 or len(te) == 0:
            continue
        out[y] = (tr, te)
    return out


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                      embargo_days: int,
                      anchor_col: str = "anchor_date",
                      resolution_col_template: str = "resolution_date_{H}d") -> None:
    res_col = resolution_col_template.format(H=horizon)
    if len(train) == 0 or len(test) == 0:
        return
    tmin = test[anchor_col].min()
    embargo = pd.Timedelta(days=int(embargo_days * 1.6))
    bad_pre = train[train[res_col] >= (tmin - embargo)]
    bad_future = train[train[anchor_col] >= tmin]
    if len(bad_future) > 0:
        raise AssertionError(
            f"LEAKAGE: {len(bad_future)} train rows have anchor_date >= test start "
            f"(future-data leakage)")
    if len(bad_pre) > 0:
        raise AssertionError(
            f"LEAKAGE: {len(bad_pre)} train rows have resolution_date >= test_start - embargo "
            f"(horizon={horizon}, embargo_days={embargo_days})")
