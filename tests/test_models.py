"""Tests for models.py — 3 model tracks."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import models as mdl


def _make_training_set(n=500, seed=42):
    rng = np.random.default_rng(seed)
    # Image features: 384 dims (mimicking DINOv2)
    X_img = rng.normal(0, 1, (n, 384)).astype(np.float32)
    # Volume features: 252 dims
    X_vol = rng.normal(0, 1, (n, 252)).astype(np.float32)
    # Engineered features: 40 dims
    X_eng = rng.normal(0, 1, (n, 40)).astype(np.float32)
    # Labels: imbalanced (15% positive)
    y = (rng.random(n) > 0.85).astype(np.int8)
    if y.sum() < 10:  # ensure both classes
        y[:10] = 1
        y[-10:] = 0
    return X_img, X_vol, X_eng, y


def test_fit_lr_baseline_basic():
    X_img, X_vol, _, y = _make_training_set()
    art = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=32, seed=0)
    assert "img_pca" in art and "vol_pca" in art and "clf" in art \
        and "calibrator" in art
    # Predict on the same data
    preds = mdl.predict_lr_baseline(art, X_img, X_vol)
    assert preds.shape == (len(y),)
    assert 0.0 <= preds.min() <= preds.max() <= 1.0


def test_fit_lgbm_engineered_basic():
    _, _, X_eng, y = _make_training_set()
    art = mdl.fit_lgbm(X_eng, y, seed=0)
    assert "clf" in art and "calibrator" in art
    preds = mdl.predict_lgbm(art, X_eng)
    assert preds.shape == (len(y),)
    assert 0.0 <= preds.min() <= preds.max() <= 1.0


def test_fit_lgbm_image_basic():
    X_img, X_vol, _, y = _make_training_set()
    art = mdl.fit_lgbm_image(X_img, X_vol, y, pca_dim=32, seed=0)
    preds = mdl.predict_lgbm_image(art, X_img, X_vol)
    assert preds.shape == (len(y),)
    assert 0.0 <= preds.min() <= preds.max() <= 1.0


def test_predictions_are_deterministic_same_seed():
    """Same seed → identical predictions."""
    X_img, X_vol, _, y = _make_training_set()
    art_a = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=16, seed=99)
    art_b = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=16, seed=99)
    p_a = mdl.predict_lr_baseline(art_a, X_img, X_vol)
    p_b = mdl.predict_lr_baseline(art_b, X_img, X_vol)
    np.testing.assert_allclose(p_a, p_b, rtol=1e-8, atol=1e-10)


def test_save_fold_roundtrips(tmp_path):
    _, _, X_eng, y = _make_training_set()
    art = mdl.fit_lgbm(X_eng, y, seed=0)
    out_path = mdl.save_fold(art, tmp_path, "ret_40d_ge_25pct", 2022,
                                "lgbm_engineered")
    assert out_path.exists()
    import joblib
    art2 = joblib.load(out_path)
    # Round-trip prediction identical
    p1 = mdl.predict_lgbm(art, X_eng)
    p2 = mdl.predict_lgbm(art2, X_eng)
    np.testing.assert_allclose(p1, p2, rtol=1e-10)


def test_pca_dim_clamped_to_input():
    """If pca_dim > n_features, PCA should silently clamp without crashing."""
    X_img = np.random.randn(200, 10).astype(np.float32)
    X_vol = np.random.randn(200, 10).astype(np.float32)
    y = (np.random.random(200) > 0.5).astype(np.int8)
    art = mdl.fit_lr_baseline(X_img, X_vol, y, pca_dim=64, seed=0)
    # Should succeed with PCA(min(64, 10)) = PCA(10)
    preds = mdl.predict_lr_baseline(art, X_img, X_vol)
    assert preds.shape == (200,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
