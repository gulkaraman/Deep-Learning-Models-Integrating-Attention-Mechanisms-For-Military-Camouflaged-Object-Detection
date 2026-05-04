"""Optimizers (hyperparameters aligned with notebooks)."""

from __future__ import annotations

import tensorflow as tf

from src.utils.config import ExperimentConfig


def get_optimizer(cfg: ExperimentConfig) -> tf.keras.optimizers.Optimizer:
    name = cfg.optimizer.lower()
    lr = cfg.learning_rate
    if name == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    if name == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")
