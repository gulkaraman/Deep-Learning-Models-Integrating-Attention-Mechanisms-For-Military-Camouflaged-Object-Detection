"""YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import yaml


@dataclass
class ExperimentConfig:
    """Training / evaluation configuration loaded from YAML."""

    model_name: str = "attention_unet"
    backbone: str = "resnet50"
    loss: str = "binary_crossentropy"
    optimizer: str = "adamax"
    learning_rate: float = 1e-5
    dropout: float = 0.2
    use_clahe: bool = True
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 32
    epochs: int = 150
    seed: int = 42
    dataset_path: str = "dataset"
    checkpoint_dir: str = "checkpoints"
    checkpoint_filename: str = "model_best.keras"
    results_dir: str = "outputs/metrics"
    freeze_resnet_layers: int = 50
    unetpp_resnet_trainable: bool = True
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 1e-4
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    csv_log_filename: str = "training_log.csv"
    include_notebook_aux_metrics: bool = True

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        img = raw.get("image_size", [256, 256])
        if isinstance(img, list) and len(img) == 2:
            image_size = (int(img[0]), int(img[1]))
        else:
            image_size = (256, 256)

        return ExperimentConfig(
            model_name=str(raw.get("model_name", "attention_unet")),
            backbone=str(raw.get("backbone", "resnet50")),
            loss=str(raw.get("loss", "binary_crossentropy")),
            optimizer=str(raw.get("optimizer", "adamax")).lower(),
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            dropout=float(raw.get("dropout", 0.2)),
            use_clahe=bool(raw.get("use_clahe", True)),
            image_size=image_size,
            batch_size=int(raw.get("batch_size", 32)),
            epochs=int(raw.get("epochs", 150)),
            seed=int(raw.get("seed", 42)),
            dataset_path=str(raw.get("dataset_path", "dataset")),
            checkpoint_dir=str(raw.get("checkpoint_dir", "checkpoints")),
            checkpoint_filename=str(raw.get("checkpoint_filename", "model_best.keras")),
            results_dir=str(raw.get("results_dir", "outputs/metrics")),
            freeze_resnet_layers=int(raw.get("freeze_resnet_layers", 50)),
            unetpp_resnet_trainable=bool(raw.get("unetpp_resnet_trainable", True)),
            early_stopping_patience=int(raw.get("early_stopping_patience", 40)),
            early_stopping_min_delta=float(raw.get("early_stopping_min_delta", 1e-4)),
            checkpoint_monitor=str(raw.get("checkpoint_monitor", "val_loss")),
            checkpoint_mode=str(raw.get("checkpoint_mode", "min")),
            csv_log_filename=str(raw.get("csv_log_filename", "training_log.csv")),
            include_notebook_aux_metrics=bool(raw.get("include_notebook_aux_metrics", True)),
        )
