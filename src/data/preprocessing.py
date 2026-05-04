"""Image / mask preprocessing (CLAHE, resize, normalize) — logic aligned with notebooks."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_rgb_uint8(image_rgb: np.ndarray, clip_limit: float = 4.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space (as in CLAHE-enabled notebooks).

    Parameters
    ----------
    image_rgb :
        ``uint8`` array, shape ``(H, W, 3)``, RGB ordering.
    """
    if image_rgb.dtype != np.uint8:
        raise TypeError("apply_clahe_rgb_uint8 expects uint8 RGB input")
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(
    image_bgr: np.ndarray,
    image_size: tuple[int, int],
    use_clahe: bool,
    *,
    clip_limit: float = 4.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    BGR (OpenCV default) → RGB, resize, optional CLAHE, scale to ``[0, 1]`` float32.

    ``image_size`` is ``(height, width)`` as in the YAML configs; OpenCV expects ``(width, height)``.
    """
    height, width = int(image_size[0]), int(image_size[1])
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    if use_clahe:
        rgb = apply_clahe_rgb_uint8(rgb, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def preprocess_mask(mask_gray: np.ndarray, image_size: tuple[int, int], threshold: int = 128) -> np.ndarray:
    """Resize mask, binarize, add channel dimension for Keras ``(H, W, 1)``."""
    height, width = int(image_size[0]), int(image_size[1])
    m = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)
    m = (m > threshold).astype(np.float32)
    return np.expand_dims(m, axis=-1)
