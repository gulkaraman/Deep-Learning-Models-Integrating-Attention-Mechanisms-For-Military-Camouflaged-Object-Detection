"""Build Keras models from ``ExperimentConfig``."""

from __future__ import annotations

import tensorflow as tf

from src.models.attention_unet import build_attention_unet
from src.models.attention_unet_plus_plus import build_attention_unet_plus_plus
from src.utils.config import ExperimentConfig


def build_model(cfg: ExperimentConfig) -> tf.keras.Model:
    if cfg.backbone.lower() != "resnet50":
        raise NotImplementedError(f"Only backbone 'resnet50' is implemented (got {cfg.backbone!r}).")

    h, w, c = cfg.image_size[0], cfg.image_size[1], 3
    name = cfg.model_name.lower().replace("-", "_")

    if name in ("attention_unet", "attention_unet_resnet50"):
        return build_attention_unet(
            (h, w, c),
            dropout_rate=cfg.dropout,
            freeze_resnet_layers=cfg.freeze_resnet_layers,
        )
    if name in ("attention_unet_plus_plus", "attention_unetplusplus", "attention_unet_plus_plus_resnet50"):
        return build_attention_unet_plus_plus(
            (h, w, c),
            dropout_rate=cfg.dropout,
            resnet_trainable=cfg.unetpp_resnet_trainable,
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name!r}")
