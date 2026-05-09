"""Three side-by-side model tracks per (label, fold).

Track A: 'lr_baseline'         — matches the original document.
  StandardScaler -> PCA64 (image) + PCA64 (volume) -> StandardScaler ->
  LogisticRegression(C=1, class_weight='balanced').

Track B: 'lgbm_image'           — same image+volume features, GBM head.
Track C: 'lgbm_engineered'      — engineered features only (no images);
                                  THE FALSIFICATION TEST.

All scalers and PCAs are fit on the train fold only (leakage-safe).
Probabilities are calibrated on a held-out 10% slice of the train fold
via isotonic regression so that scores are comparable across folds.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class ModelArtifacts:
    track: str
    fitted: dict


def _calibrate(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, y)
    return iso


def _split_calib(X: np.ndarray, y: np.ndarray, seed: int = 42):
    return train_test_split(X, y, test_size=0.10, random_state=seed, stratify=y)


def fit_lr_baseline(X_img: np.ndarray, X_vol: np.ndarray, y: np.ndarray,
                    pca_dim: int = 64, seed: int = 42, lr_kwargs: dict | None = None) -> dict:
    lr_kwargs = lr_kwargs or {}
    s1 = StandardScaler().fit(X_img)
    Xi = s1.transform(X_img)
    pi = PCA(n_components=min(pca_dim, Xi.shape[1]), random_state=seed).fit(Xi)
    Xi = pi.transform(Xi)

    s2 = StandardScaler().fit(X_vol)
    Xv = s2.transform(X_vol)
    pv = PCA(n_components=min(pca_dim, Xv.shape[1]), random_state=seed).fit(Xv)
    Xv = pv.transform(Xv)

    X = np.concatenate([Xi, Xv], axis=1)
    s3 = StandardScaler().fit(X)
    X = s3.transform(X)

    X_fit, X_calib, y_fit, y_calib = _split_calib(X, y, seed=seed)
    clf = LogisticRegression(
        C=lr_kwargs.get("C", 1.0),
        class_weight=lr_kwargs.get("class_weight", "balanced"),
        solver=lr_kwargs.get("solver", "lbfgs"),
        max_iter=lr_kwargs.get("max_iter", 1000),
        random_state=seed,
    )
    clf.fit(X_fit, y_fit)
    raw = clf.predict_proba(X_calib)[:, 1]
    iso = _calibrate(raw, y_calib)
    return {"img_scaler": s1, "img_pca": pi, "vol_scaler": s2, "vol_pca": pv,
            "post_scaler": s3, "clf": clf, "calibrator": iso}


def predict_lr_baseline(art: dict, X_img: np.ndarray, X_vol: np.ndarray) -> np.ndarray:
    Xi = art["img_pca"].transform(art["img_scaler"].transform(X_img))
    Xv = art["vol_pca"].transform(art["vol_scaler"].transform(X_vol))
    X = art["post_scaler"].transform(np.concatenate([Xi, Xv], axis=1))
    raw = art["clf"].predict_proba(X)[:, 1]
    return art["calibrator"].predict(raw)


def fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 42, lgbm_kwargs: dict | None = None) -> dict:
    if not HAS_LGBM:
        raise RuntimeError("lightgbm is not installed")
    lgbm_kwargs = lgbm_kwargs or {}
    X_fit, X_calib, y_fit, y_calib = _split_calib(X, y, seed=seed)
    clf = LGBMClassifier(
        n_estimators=lgbm_kwargs.get("n_estimators", 400),
        learning_rate=lgbm_kwargs.get("learning_rate", 0.05),
        num_leaves=lgbm_kwargs.get("num_leaves", 63),
        min_child_samples=lgbm_kwargs.get("min_child_samples", 200),
        reg_alpha=lgbm_kwargs.get("reg_alpha", 0.1),
        reg_lambda=lgbm_kwargs.get("reg_lambda", 0.1),
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_fit, y_fit)
    raw = clf.predict_proba(X_calib)[:, 1]
    iso = _calibrate(raw, y_calib)
    return {"clf": clf, "calibrator": iso}


def predict_lgbm(art: dict, X: np.ndarray) -> np.ndarray:
    raw = art["clf"].predict_proba(X)[:, 1]
    return art["calibrator"].predict(raw)


def fit_lgbm_image(X_img: np.ndarray, X_vol: np.ndarray, y: np.ndarray,
                   pca_dim: int = 64, seed: int = 42, lgbm_kwargs: dict | None = None) -> dict:
    s1 = StandardScaler().fit(X_img)
    Xi = s1.transform(X_img)
    pi = PCA(n_components=min(pca_dim, Xi.shape[1]), random_state=seed).fit(Xi)
    Xi = pi.transform(Xi)

    s2 = StandardScaler().fit(X_vol)
    Xv = s2.transform(X_vol)
    pv = PCA(n_components=min(pca_dim, Xv.shape[1]), random_state=seed).fit(Xv)
    Xv = pv.transform(Xv)

    X = np.concatenate([Xi, Xv], axis=1)
    art = fit_lgbm(X, y, seed=seed, lgbm_kwargs=lgbm_kwargs)
    art.update({"img_scaler": s1, "img_pca": pi, "vol_scaler": s2, "vol_pca": pv})
    return art


def predict_lgbm_image(art: dict, X_img: np.ndarray, X_vol: np.ndarray) -> np.ndarray:
    Xi = art["img_pca"].transform(art["img_scaler"].transform(X_img))
    Xv = art["vol_pca"].transform(art["vol_scaler"].transform(X_vol))
    X = np.concatenate([Xi, Xv], axis=1)
    return predict_lgbm(art, X)


def save_fold(art: dict, out_dir: Path, label_key: str, year: int, track: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{track}__{label_key}__{year}.joblib"
    joblib.dump(art, p, compress=3)
    return p
