"""Deterministic chart-image renderer with FIXED log-y axis.

CORRECTED DESIGN (vs original document):
- Original: 'first close = 100' + per-chart y autoscale -> +5% and +500%
  charts render to visually identical line silhouettes. Information about
  return magnitude is destroyed BEFORE the encoder sees it. Since the
  classifier label is fundamentally about return magnitude (>= 25% in
  40d), this is a structural information bottleneck.
- Corrected: Fixed log-y axis spanning [0.1x, 11x] of the anchor close
  (i.e. -90% to +1000%). Magnitude is now spatially encoded; a flat-line
  +5% and a parabolic +500% land in different pixel regions.

The renderer:
- 224 x 224 RGB
- White background, single black polyline (line_width=2)
- No labels, axes, ticks, grid, MA overlays, or future information
- Deterministic: same input -> identical pixel bytes (assert tested)
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import numpy as np
from PIL import Image, ImageDraw


def render_one(close_window: np.ndarray, image_size: int = 224,
               log_y_min: float = 0.1, log_y_max: float = 11.0,
               line_width: int = 2) -> Image.Image:
    """Render a 252-day close window into a 224x224 RGB image.

    `close_window` is a 1D numpy array of length L (typically 252) of
    closing prices. The first element is treated as the anchor for
    log-ratio normalization. Out-of-range values are clipped to the axis.
    """
    assert close_window.ndim == 1 and close_window.size >= 2
    anchor = float(close_window[0])
    if anchor <= 0:
        # degenerate: render a flat line at log-ratio 1.0
        ratios = np.ones_like(close_window, dtype=np.float64)
    else:
        ratios = close_window.astype(np.float64) / anchor

    log_ratios = np.log(np.clip(ratios, 1e-6, None))
    y_lo = float(np.log(log_y_min))
    y_hi = float(np.log(log_y_max))
    # Defensive: degenerate or inverted axis collapses to a centered flat line
    # rather than triggering a divide-by-zero RuntimeWarning.
    if y_hi <= y_lo or not np.isfinite(y_hi - y_lo):
        y_hi = y_lo + 1.0  # synthesize a unit range so the math is finite
        log_ratios = np.full_like(log_ratios, (y_lo + y_hi) / 2.0)

    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    L = len(log_ratios)
    if L < 2:
        return img

    xs = np.linspace(0, image_size - 1, L)
    norm = (log_ratios - y_lo) / (y_hi - y_lo)
    norm = np.clip(norm, 0.0, 1.0)
    ys = (1.0 - norm) * (image_size - 1)

    pts = list(zip(xs.tolist(), ys.tolist()))
    draw.line(pts, fill=(0, 0, 0), width=line_width)
    return img


def image_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def image_sha256(img: Image.Image) -> str:
    arr = np.array(img, dtype=np.uint8).tobytes()
    return hashlib.sha256(arr).hexdigest()


def render_to_array(close_window: np.ndarray, **kwargs) -> np.ndarray:
    """Render and return a uint8 array of shape (H, W, 3)."""
    img = render_one(close_window, **kwargs)
    return np.asarray(img, dtype=np.uint8)
