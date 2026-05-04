"""Dataset discovery and loading for ``dataset/{split}/{images,masks}`` layout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.data.preprocessing import preprocess_image, preprocess_mask
from src.utils.paths import split_image_mask_dirs


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg")


def discover_pairs(images_dir: Path | str, masks_dir: Path | str) -> List[Tuple[Path, Path]]:
    """
    Match images under ``images_dir`` to masks in ``masks_dir`` by common stem.

    Prefers ``.png`` masks, then ``.jpg`` / ``.jpeg``, mirroring notebook logic.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

    stems: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            stems[p.stem] = p

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for stem, img_path in sorted(stems.items()):
        mask_path = None
        for ext in MASK_EXTENSIONS:
            cand = masks_dir / f"{stem}{ext}"
            if cand.is_file():
                mask_path = cand
                break
        if mask_path is None:
            missing.append(stem)
            continue
        pairs.append((img_path, mask_path))

    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(f"No mask found for image stems: {preview}{more}")

    if not pairs:
        raise ValueError(f"No image/mask pairs found under {images_dir} / {masks_dir}")

    return pairs


def load_split_arrays(
    dataset_root: Path | str,
    split: str,
    image_size: tuple[int, int],
    use_clahe: bool,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load every pair in ``split`` into float32 NumPy arrays ``(N,H,W,3)`` and ``(N,H,W,1)``."""
    img_dir, mask_dir = split_image_mask_dirs(dataset_root, split)
    pairs = discover_pairs(img_dir, mask_dir)
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    iterator: Iterator[Tuple[Path, Path]] = pairs
    if verbose:
        iterator = tqdm(pairs, desc=f"Loading {split}")

    for img_path, mask_path in iterator:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        images.append(preprocess_image(bgr, image_size, use_clahe))
        masks.append(preprocess_mask(gray, image_size))

    x = np.stack(images, axis=0).astype(np.float32)
    y = np.stack(masks, axis=0).astype(np.float32)
    return x, y


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` from in-memory arrays."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 1024), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
