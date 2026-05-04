"""Load saved checkpoints for evaluation / inference."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.models.factory import build_model
from src.training.metrics import custom_objects_dict
from src.utils.config import ExperimentConfig


def load_model_for_evaluation(cfg: ExperimentConfig, checkpoint_path: str | Path) -> tf.keras.Model:
    """
    Try full-model load first; fall back to ``build_model`` + ``load_weights``.

    ``custom_objects`` covers metrics / helpers saved with the notebook-style graph.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    custom = custom_objects_dict()
    try:
        try:
            return tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)
        except TypeError:
            return tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    except Exception:
        model = build_model(cfg)
        model.load_weights(str(path))
        return model
