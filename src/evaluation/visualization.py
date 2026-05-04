"""Save qualitative prediction figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_mask_preview(
    image_hwc: np.ndarray,
    mask_hw: np.ndarray,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Save a side-by-side RGB image and predicted mask (2D float in ``[0,1]``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(image_hwc, 0.0, 1.0))
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(np.clip(mask_hw, 0.0, 1.0), cmap="magma")
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
