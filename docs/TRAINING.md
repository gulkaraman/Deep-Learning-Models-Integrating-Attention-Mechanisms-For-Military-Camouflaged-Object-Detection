# Training

Experiments are executed **end-to-end in Jupyter notebooks** (TensorFlow / Keras). The following is the **logical training pipeline** reflected in the notebooks.

## Step-by-step training flow

1. **Dataset loading** — Paths (`PATH`, `TRAIN_IMG_PATH`, `TRAIN_MASK_PATH`, …) point to image and mask folders; `load_data` builds NumPy arrays of image–mask pairs.
2. **Preprocessing** — Resize, optional CLAHE (see `docs/PREPROCESSING.md`), intensity scaling, mask binarization.
3. **Model creation** — `build_unet` (Attention U-Net) or `build_resnet50_attention_unet_plus_plus` (Attention U-Net++).
4. **Loss** — Most notebooks use **`binary_focal_crossentropy`** in `model.compile`, but some runs (for example certain **Adamax** notebooks) use **`binary_crossentropy`**. Always read the **`model.compile`** cell of the notebook you execute.
5. **Optimizer** — `Adam`, `Adamax`, or `Adadelta` depending on the notebook filename and compile cell.
6. **Training loop** — `model.fit(...)` with **`validation_split`** (e.g. 0.2) on the training array, plus callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau—see notebook).
7. **K-fold (optional / default in many notebooks)** — Outer loop over `KFold` splits on `(images, masks)` trains one model per fold and logs metrics.
8. **Checkpointing** — Best weights saved under paths such as `Project_Outputs/model_outputs/` inside the notebook (see `docs/REPRODUCIBILITY.md` for aligning with `checkpoints/`).

## Hyperparameters: paper vs current notebooks

### Paper configuration (reported results)

These values correspond to the **published experimental highlights** (see `docs/RESULTS.md`):

| Setting | Attention U-Net (paper row) | Attention U-Net++ (paper row) |
|--------|-----------------------------|--------------------------------|
| Backbone | ResNet-50 | ResNet-50 |
| Preprocessing | CLAHE | Similar experimental setup |
| Optimizer | **Adamax** | **Adam** |
| Learning rate | **1e-5** | (see paper; notebooks use `LR=1e-5` for many `att-unet++` runs) |
| Dropout | **0.2** | (paper table uses “-”; several notebooks fix dropout at 0.2 or 0.5—match your run) |

### Current implementation (repository notebooks)

| Aspect | Typical notebook values |
|--------|-------------------------|
| Image size | 256×256 |
| Batch size | 32 |
| Epochs | 150 |
| K-fold | `KFOLD_NUM = 10` |
| Learning rate | `1e-5` (often `0.00001` or `LR=0.00001`) |
| Dropout | `DROPOUT_RATE` per notebook filename (0.2 or 0.5) |
| CLAHE | Present in `*claheli*` / many non-`clahesiz` Attention U-Net notebooks; **absent** in `*clahesiz*` notebooks |

**TODO (reproducibility):** there is **no single notebook** that simultaneously matches every paper knob for every row (for example, **Adamax + dropout 0.2 + CLAHE** may require merging the CLAHE branch from a CLAHE notebook with the optimizer cell from an Adamax notebook). For exact reproduction, align `load_data`, `DROPOUT_RATE`, and `model.compile` with the paper row you are reproducing.

## Loss and monitoring

Training monitors validation metrics such as **`val_iou_metric`** (exact metric object names appear in the `compile` and `metrics` list in the notebook).
