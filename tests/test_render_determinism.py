"""Renderer must be deterministic: same input -> same SHA-256 pixel hash."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_chart.render import render_one, image_sha256


def test_determinism_random_path():
    rng = np.random.default_rng(1234)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, size=252))
    img1 = render_one(closes)
    img2 = render_one(closes)
    assert image_sha256(img1) == image_sha256(img2)


def test_magnitude_is_encoded():
    """The corrected fixed log-y axis must distinguish +5% from +500% paths."""
    flat = np.linspace(100.0, 105.0, 252)
    parabolic = np.linspace(100.0, 600.0, 252)
    h_flat = image_sha256(render_one(flat))
    h_para = image_sha256(render_one(parabolic))
    assert h_flat != h_para, "renderer collapses different magnitudes — y-axis still autoscales!"


def test_clip_range():
    """Values beyond [log_y_min, log_y_max] must clip silently."""
    closes = 100.0 * np.array([(2.0 ** (i / 50)) for i in range(252)])  # huge growth
    img = render_one(closes)
    assert img.size == (224, 224)


if __name__ == "__main__":
    test_determinism_random_path()
    test_magnitude_is_encoded()
    test_clip_range()
    print("PASS render determinism + magnitude")
