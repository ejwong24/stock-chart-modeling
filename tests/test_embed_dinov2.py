"""Tests for embed_dinov2.py — preprocessing-only (no model load)."""
import sys
from pathlib import Path
import torch
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart.embed_dinov2 import (
    _preprocess_batch, commit_hash, IMAGENET_MEAN, IMAGENET_STD)


def test_preprocess_basic_shape_dtype():
    imgs = [Image.new("RGB", (224, 224), (255, 255, 255)) for _ in range(3)]
    t = _preprocess_batch(imgs)
    assert t.shape == (3, 3, 224, 224)
    assert t.dtype == torch.float32


def test_preprocess_normalization_applies():
    """White image → (1.0 - mean) / std for each channel."""
    imgs = [Image.new("RGB", (224, 224), (255, 255, 255))]
    t = _preprocess_batch(imgs)
    expected = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert abs(t[0, 0, 0, 0].item() - expected) < 1e-4


def test_preprocess_rgba_handled():
    """RGBA input should be silently converted to RGB."""
    rgba = Image.new("RGBA", (224, 224), (200, 100, 50, 128))
    t = _preprocess_batch([rgba])
    assert t.shape == (1, 3, 224, 224)


def test_preprocess_empty_batch():
    """Bug #8 regression: empty list must return empty tensor, not crash."""
    t = _preprocess_batch([])
    assert t.shape == (0, 3, 224, 224)
    assert t.dtype == torch.float32


def test_preprocess_single_image():
    t = _preprocess_batch([Image.new("RGB", (224, 224), (128, 128, 128))])
    assert t.shape == (1, 3, 224, 224)


def test_preprocess_non_224():
    """Different image sizes pass through (caller responsibility to size)."""
    small = Image.new("RGB", (100, 100), (128, 128, 128))
    t = _preprocess_batch([small])
    assert t.shape == (1, 3, 100, 100)


def test_commit_hash_returns_string():
    """commit_hash should always return a string, even if no git cache."""
    h = commit_hash()
    assert isinstance(h, str)
    assert len(h) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
