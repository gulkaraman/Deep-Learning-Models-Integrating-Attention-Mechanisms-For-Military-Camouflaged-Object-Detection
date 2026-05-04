"""Path helpers for the recommended dataset layout."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def split_image_mask_dirs(dataset_root: Path | str, split: str) -> tuple[Path, Path]:
    """
    Return ``(images_dir, masks_dir)`` for layout::

        dataset_root / split / images
        dataset_root / split / masks
    """
    root = Path(dataset_root)
    return root / split / "images", root / split / "masks"
