"""Reproducibility audit — verifies the artifacts we expect to be deterministic."""
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import render
from stock_chart import features as feat
from stock_chart import labels as lbl
from stock_chart import models as mdl


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def test_chart_render_bit_identical_for_same_input():
    rng = np.random.default_rng(2026)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, 252))
    arr1 = np.array(render.render_one(closes))
    arr2 = np.array(render.render_one(closes))
    assert _hash_array(arr1) == _hash_array(arr2)


def test_labels_bit_identical_for_same_input(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "close": 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, 500)),
        "volume": rng.lognormal(15, 0.4, 500),
    })
    a = lbl.label_one(df, "X", horizons=[40], thresholds=[0.25])
    b = lbl.label_one(df, "X", horizons=[40], thresholds=[0.25])
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                     b.reset_index(drop=True))


def test_features_bit_identical_for_same_input():
    rng = np.random.default_rng(13)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, 260))
    vols = rng.lognormal(14, 0.4, 260)
    f1 = feat._compute_window_features(closes, vols, None, 259)
    f2 = feat._compute_window_features(closes, vols, None, 259)
    for k in f1:
        assert f1[k] == f2[k], f"feature {k} differs: {f1[k]} vs {f2[k]}"


def test_lgbm_bit_identical_with_same_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 30)).astype(np.float32)
    y = (rng.random(300) > 0.7).astype(np.int8)
    if y.sum() < 5: y[:5] = 1
    a = mdl.fit_lgbm(X, y, seed=42)
    b = mdl.fit_lgbm(X, y, seed=42)
    p_a = mdl.predict_lgbm(a, X)
    p_b = mdl.predict_lgbm(b, X)
    np.testing.assert_allclose(p_a, p_b, rtol=1e-12, atol=1e-15)


def test_lr_bit_identical_with_same_seed():
    rng = np.random.default_rng(0)
    X_img = rng.normal(0, 1, (200, 50)).astype(np.float32)
    X_vol = rng.normal(0, 1, (200, 30)).astype(np.float32)
    y = (rng.random(200) > 0.7).astype(np.int8)
    if y.sum() < 5: y[:5] = 1
    a = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=16, seed=99)
    b = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=16, seed=99)
    p_a = mdl.predict_lr_baseline(a, X_img, X_vol)
    p_b = mdl.predict_lr_baseline(b, X_img, X_vol)
    np.testing.assert_allclose(p_a, p_b, rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
