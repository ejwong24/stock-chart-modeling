"""Render edge-case tests (beyond the basic determinism suite)."""
import sys
from pathlib import Path
import warnings
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart import render as r


def test_degenerate_log_y_no_warning():
    """Bug #7 regression: log_y_min == log_y_max must not divide-by-zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        closes = np.linspace(50, 100, 252)
        img = r.render_one(closes, log_y_min=1.0, log_y_max=1.0)
        assert img.size == (224, 224)


def test_inverted_log_y_no_warning():
    """log_y_min > log_y_max should be silently handled."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        closes = np.linspace(50, 100, 252)
        img = r.render_one(closes, log_y_min=10.0, log_y_max=0.5)
        assert img.size == (224, 224)


def test_size_8():
    img = r.render_one(np.linspace(50, 100, 252), image_size=8)
    assert img.size == (8, 8)


def test_size_huge():
    img = r.render_one(np.linspace(50, 100, 252), image_size=512)
    assert img.size == (512, 512)


def test_line_width_zero():
    img = r.render_one(np.linspace(50, 100, 252), line_width=0)
    # PIL should not crash with width=0 — line just won't be visible
    assert img.size == (224, 224)


def test_negative_anchor_close_falls_to_flat():
    """If first close is negative or zero, the renderer should degenerate gracefully."""
    closes = np.array([-1.0] + [10.0] * 251)
    img = r.render_one(closes)
    assert img.size == (224, 224)


def test_all_inf_window():
    """Inf values in window get clipped by np.clip(ratios, 1e-6, None)."""
    closes = np.array([100.0] + [np.inf] * 251)
    img = r.render_one(closes)
    assert img.size == (224, 224)


def test_image_to_bytes_roundtrip():
    closes = np.linspace(50, 100, 252)
    img = r.render_one(closes)
    b = r.image_to_bytes(img)
    assert isinstance(b, (bytes, bytearray))
    assert b[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_hash_stable_across_render_calls():
    """Same input → same hash (calls to render_one are deterministic)."""
    closes = 100.0 * np.cumprod(1.0 + np.random.default_rng(42).normal(0.001, 0.02, 252))
    h1 = r.image_sha256(r.render_one(closes))
    h2 = r.image_sha256(r.render_one(closes))
    assert h1 == h2


def test_render_to_array_shape():
    closes = np.linspace(50, 100, 252)
    arr = r.render_to_array(closes)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
