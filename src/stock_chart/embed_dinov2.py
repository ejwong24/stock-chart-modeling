"""Frozen DINOv2 ViT-S/14 chart-image embedder (CPU, ARM64).

Loads DINOv2 ViT-S/14 from torch.hub, runs in eval mode, no grad. Outputs
384-dim float32 embeddings. We embed once and cache; downstream PCA is
re-fit per fold so this stays leakage-safe.

Caveat (from analysis): DINOv2 is pretrained on natural images; line
charts on white background are far OOD. The companion engineered-features
LightGBM (features.py) is the falsification baseline. If the engineered
baseline matches or beats the DINOv2 stack on honest stats, the image
pipeline is not justified.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(num_threads: int = 4) -> nn.Module:
    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False, trust_repo=True)
    model.eval()
    return model


def _preprocess_batch(images: list[Image.Image]) -> torch.Tensor:
    if len(images) == 0:
        # Empty input → empty NCHW tensor with correct trailing dims (224×224×3)
        return torch.empty((0, 3, 224, 224), dtype=torch.float32)
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.float32) for im in images])
    arr = arr / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
    return torch.from_numpy(arr).contiguous()


def embed_batch(model: nn.Module, images: list[Image.Image]) -> np.ndarray:
    x = _preprocess_batch(images)
    with torch.inference_mode():
        feats = model(x)
    return feats.detach().cpu().numpy().astype(np.float32)


def embed_arrays(model: nn.Module, arrays: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Embed a batch of pre-rendered uint8 (N, 224, 224, 3) arrays."""
    out = []
    n = len(arrays)
    if n == 0:
        # np.concatenate([]) raises; return a correctly-shaped empty result.
        return np.empty((0, 384), dtype=np.float32)  # ViT-S/14 CLS dim
    for i in range(0, n, batch_size):
        batch = arrays[i:i + batch_size]
        ims = [Image.fromarray(b) for b in batch]
        out.append(embed_batch(model, ims))
    return np.concatenate(out, axis=0)


def commit_hash() -> str:
    """Return DINOv2 hub cache commit hash (best-effort)."""
    try:
        cache = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        head = cache / ".git" / "HEAD"
        if head.exists():
            return head.read_text().strip()
    except Exception:
        pass
    return "unknown"
