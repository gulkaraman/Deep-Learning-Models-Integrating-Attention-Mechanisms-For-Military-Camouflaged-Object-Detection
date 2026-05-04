"""Training callbacks (checkpointing, early stopping, CSV log)."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.utils.config import ExperimentConfig


def resolve_val_monitor(model: tf.keras.Model, cfg: ExperimentConfig) -> tuple[str, str]:
    """
    Map a requested ``val_*`` monitor to a name that exists during ``fit``.

    After ``compile``, ``model.metrics_names`` lists training metric names without
    the ``val_`` prefix (e.g. ``iou``); runtime logs add ``val_`` automatically.
    """
    names = list(model.metrics_names)
    requested = cfg.checkpoint_monitor

    def exists_at_runtime(val_name: str) -> bool:
        if val_name == "val_loss":
            return True
        if not val_name.startswith("val_"):
            return False
        base = val_name[len("val_") :]
        return base in names

    candidates = [requested, "val_iou_metric", "val_iou", "val_mean_iou", "val_accuracy", "val_loss"]
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if exists_at_runtime(cand):
            if cand == "val_loss":
                return cand, "min"
            mode = cfg.checkpoint_mode if cand == requested else "max"
            return cand, mode
    return "val_loss", "min"


def build_callbacks(cfg: ExperimentConfig, monitor: str, mode: str, checkpoint_path: Path, csv_path: Path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor=monitor,
            mode=mode,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            monitor=monitor,
            mode=mode,
            min_delta=cfg.early_stopping_min_delta,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), separator=",", append=False),
    ]
