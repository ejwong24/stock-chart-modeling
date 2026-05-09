"""Purged walk-forward must not leak label-resolution dates into the test fold."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart.splits import yearly_walk_forward, assert_no_leakage


def _toy_anchors(horizon: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-02", "2025-12-31", freq="W-FRI")
    rows = []
    for d in dates:
        for t in [f"T{i:03d}" for i in range(20)]:
            rows.append({"ticker": t, "anchor_date": d,
                         f"resolution_date_{horizon}d": d + pd.Timedelta(days=int(horizon * 1.45))})
    df = pd.DataFrame(rows)
    return df


def test_no_leakage_basic():
    horizon = 40
    df = _toy_anchors(horizon)
    folds = yearly_walk_forward(df, horizon=horizon,
                                 test_years=[2019, 2020, 2021, 2022, 2023],
                                 embargo_horizon_multiplier=1)
    for y, (tr, te) in folds.items():
        assert len(tr) > 0 and len(te) > 0
        assert_no_leakage(tr, te, horizon=horizon, embargo_days=horizon)


def test_embargo_drops_more_than_zero():
    """The embargo window should drop at least some training rows near each fold."""
    horizon = 40
    df = _toy_anchors(horizon)
    folds = yearly_walk_forward(df, horizon=horizon, test_years=[2020],
                                 embargo_horizon_multiplier=1)
    tr, te = folds[2020]
    # Without purging the train pool would include all anchors not in 2020.
    naive_train = df[df["anchor_date"].dt.year != 2020]
    assert len(tr) < len(naive_train), "purging didn't drop any rows — embargo broken"


if __name__ == "__main__":
    test_no_leakage_basic()
    test_embargo_drops_more_than_zero()
    print("PASS purged walk-forward, no leakage")
