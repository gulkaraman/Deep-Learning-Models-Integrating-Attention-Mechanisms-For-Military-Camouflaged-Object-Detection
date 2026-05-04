# Dataset (ACD1K)

## Role in this project

The **Adaptive Camouflage Dataset (ACD1K)** is used for **military camouflaged object detection and semantic segmentation**: each RGB image is paired with a pixel-wise mask that separates camouflaged military objects from the background.

## Train / validation / test protocol

In the published study, the dataset is organized into **training**, **validation**, and **test** subsets for supervised learning and evaluation.

**Current implementation (notebooks):** notebooks load **training** and **testing** folders from a configurable root path (`PATH`, `TRAIN_*`, `TEST_*`). **Validation** is obtained during training via `validation_split` on the training tensor, and experiments additionally use **K-fold cross-validation** on the training set (see `docs/TRAINING.md`). This differs slightly from a static `val/` folder on disk; you may mirror the paper’s split by preparing a dedicated validation directory and adjusting the loading cells accordingly.

## Redistribution policy

**The ACD1K dataset is not bundled with this repository.** Do not commit raw images, masks, or archives to GitHub. Obtain the dataset from the official source / licence terms associated with ACD1K and place it on your machine locally.

## Manual download and expected layout (recommended)

Create a root folder (for example `dataset/`) with the following structure so paths are easy to map from `config/example_config.yaml`:

```text
dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

When using the **existing notebooks unchanged**, you will typically set `PATH` to a directory whose **internal names match the notebook** (for example `Training/images`, `Training/GT`, `Testing/images`, `Testing/GT`). Align your local folder names with the variables at the top of each notebook, or update those variables to match your `dataset/` layout.

## File pairing

Notebooks assume image files (commonly `.jpg`) and masks that share the same basename with `.png` or `.jpg` extension. See `load_data` in any experiment notebook for the exact pairing logic.
