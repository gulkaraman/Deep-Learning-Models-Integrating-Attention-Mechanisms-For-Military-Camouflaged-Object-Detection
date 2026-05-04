# Reproducibility

## Environment

- **OS:** Windows, Linux, or macOS with a 64-bit Python environment.
- **Python:** 3.10 or 3.11 recommended (match your TensorFlow wheel availability).
- **GPU:** strongly recommended for training; CPU runs may be impractical at the default resolution and epoch count.

## Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Dataset layout

Prepare **ACD1K** locally using the script pipeline layout:

```text
dataset/
  train/images
  train/masks
  val/images
  val/masks
  test/images
  test/masks
```

See [`DATASET.md`](DATASET.md) for acquisition notes and for how **notebook** `PATH` variables differ from this layout.

## Seeds

Notebooks set `SEED`, `tf.random.set_seed`, and `np.random.seed`. The script pipeline calls `src.utils.seed.set_seed` from the YAML `seed` field.

---

## Notebook-based reproduction

1. Launch Jupyter (`jupyter lab`).
2. Open the notebook matching your experiment (`att unet *.ipynb` or `att-unet++*.ipynb`).
3. Point notebook `PATH` variables at your local split (often `Training/` / `Testing/` style names).
4. Run all cells.

This path preserves the **original K-fold** and `Project_Outputs/` logging behavior.

---

## Script-based reproduction

From the **repository root**:

```bash
python -m src.training.train --config config/paper_attention_unet_clahe_adamax.yaml
python -m src.evaluation.evaluate --config config/paper_attention_unet_clahe_adamax.yaml --checkpoint checkpoints/attention_unet_paper.keras --split test
python -m src.evaluation.predict --config config/paper_attention_unet_clahe_adamax.yaml --checkpoint checkpoints/attention_unet_paper.keras --input path/to/images --output outputs/predictions
```

Configs:

- [`config/example_config.yaml`](../config/example_config.yaml) — generic template.
- [`config/paper_attention_unet_clahe_adamax.yaml`](../config/paper_attention_unet_clahe_adamax.yaml) — Attention U-Net paper row (CLAHE + Adamax + lr `1e-5` + dropout `0.2`; loss matches the Adamax notebook compile cell: `binary_crossentropy`).
- [`config/paper_attention_unetpp_adam.yaml`](../config/paper_attention_unetpp_adam.yaml) — Attention U-Net++ Adam row (CLAHE off to match `*clahesiz*` notebooks; `binary_focal_crossentropy`).

**Important:** the scripts **do not claim to reproduce published numbers automatically**. Published Accuracy / IoU remain documented only under [`RESULTS.md`](RESULTS.md).

---

## Notebook-to-script migration notes

This section records what was **ported faithfully**, what is **intentionally simplified**, and what still needs **manual verification**.

### Faithfully mirrored (logic aligned with notebooks)

- **Attention U-Net** decoder + **ResNet-50** encoder stage hooks (`conv1_relu`, `conv2_block3_out`, …) and **attention-gated** skip fusion (`src/models/attention_unet.py`).
- **Attention U-Net++** nested decoder, **named** attention gates, `conv_block` dropout pattern, final upsample + `1x1` sigmoid head (`src/models/attention_unet_plus_plus.py`).
- **CLAHE** in LAB on the L channel (`clipLimit=4.0`, `tileGridSize=(8,8)`) when `use_clahe: true` (`src/data/preprocessing.py`).
- **Mask pairing** logic (image stem; mask `.png` preferred over `.jpg`) (`src/data/dataset.py`).
- **Custom metrics** `dice_coefficient`, `f1_score`, `mcc` and Keras `MeanIoU` / `IoU(target_class_ids=[0])` as in the notebooks (`src/training/metrics.py`).
- **Optimizers** Adamax / Adam / Adadelta wiring with notebook-style default hyperparameters (`src/training/optimizers.py`).
- **Losses present in notebooks:** `binary_crossentropy` (object) and `binary_focal_crossentropy` (string) (`src/training/losses.py`).

### Intentional differences / TODOs

- **K-fold:** notebooks default to **10-fold** on the training tensor; `src/training/train.py` currently trains a **single** model on `dataset/train` with validation on `dataset/val`. Reintroducing K-fold is a **TODO** (track mean/std outside the script or add a `--kfold` flag).
- **Albumentations augmentation:** not wired into `train.py` yet (**TODO**); notebooks apply geometric augmentation in helper cells.
- **Monitor name mismatch:** some notebooks reference `val_iou_metric` in callbacks; Keras often exposes `val_iou` when the metric is registered as `iou`. `src/training/callbacks.py` resolves a small candidate list, but you should still **inspect `CSVLogger` output** if a callback appears inactive.
- **IoU class indexing:** the notebooks keep `IoU(..., target_class_ids=[0])`. Whether this corresponds to **foreground** vs **background** IoU for your mask encoding should be **manually confirmed** before comparing to paper tables.
- **Dice loss:** not found as a training loss in the inspected notebooks — **not implemented** in `src/training/losses.py` (only Dice as a **metric**).
- **Checkpoint format:** training saves a **full Keras model** file (default `*.keras`). The CLI examples also accept `.h5` **weights** via `build_model` + `load_weights` fallback in `src/evaluation/model_loading.py`, but you must use the **same architecture config** as the checkpoint.

### What was not executed here

Automated end-to-end training on ACD1K was **not** run in CI. Expect to download **ImageNet** ResNet-50 weights on first `build_model` call.

---

## Model weights placement

1. Keep local checkpoints under `checkpoints/` (large files remain gitignored).
2. Prefer the script defaults (`checkpoint_filename` in YAML) so evaluate/predict stay in sync.

## Figures and exports

- `outputs/figures/`, `outputs/metrics/`, `outputs/predictions/` (see `.gitignore` rules).
- Legacy notebooks may still write to `Project_Outputs/`; that path is gitignored.
