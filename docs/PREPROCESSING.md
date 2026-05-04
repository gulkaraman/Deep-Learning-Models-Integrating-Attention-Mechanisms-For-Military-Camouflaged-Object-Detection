# Preprocessing

Implementation lives in the **Jupyter notebooks** (not in standalone `.py` modules). Search for the functions below inside the notebook you are running.

## CLAHE (Contrast Limited Adaptive Histogram Equalization)

**CLAHE** is applied in the **L channel** of the **LAB** color space after resizing and before scaling pixel values to \([0,1]\).

Typical steps (as in CLAHE-enabled notebooks):

1. Convert RGB → LAB.
2. Run `cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))` on the L channel.
3. Convert LAB → RGB.

**Purpose:** improve local contrast so that **low-contrast camouflaged regions** may exhibit slightly **clearer boundaries** and texture cues prior to CNN feature extraction.

**Notebooks without CLAHE** (filenames containing `clahesiz`) **omit** this block and proceed directly to normalization after resize.

## Normalization

Images are scaled with **`img = img / 255.0`** so intensity values lie in \([0, 1]\).

## Resizing

Images and masks are resized to **`IMAGE_SIZE`**, typically **`(256, 256)`**, using OpenCV (`cv2.resize`).

## Mask preparation

Masks are read in grayscale, resized, then binarized with a threshold such as **`mask = (mask > 128).astype(np.float32)`**, and expanded with **`np.expand_dims(..., axis=-1)`** for channel dimension compatibility with the network.

## Data augmentation (where enabled)

**Albumentations** (`albumentations as A`) is used in training helpers such as **`augment_data`** with geometric transforms, for example:

- `RandomRotate90`
- `HorizontalFlip`
- `VerticalFlip`

Exact probabilities and disabled transforms are defined per notebook (see the `A.Compose([...])` block).

## Where in the repository

| Step | Typical notebook symbols |
|------|---------------------------|
| Load + resize + CLAHE + normalize | `load_data` |
| Augmentation | `augment_data`, `get_training_augmentation` (names may vary slightly) |

## Pseudo-code (illustrative only)

```text
for each image path:
    img = read_bgr -> rgb
    mask = read_grayscale
    resize both to (H, W)
    optionally apply CLAHE on L channel in LAB space
    img = img / 255.0
    mask = (mask > threshold).float()
    mask = add_channel_dim(mask)
```

This mirrors the notebook logic; **always treat the notebook as the source of truth** for parameters.
