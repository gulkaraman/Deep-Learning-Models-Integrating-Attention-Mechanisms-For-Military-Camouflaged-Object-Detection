"""Train segmentation models from a YAML config (script entry point)."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.data.dataset import load_split_arrays, make_tf_dataset
from src.models.factory import build_model
from src.training.callbacks import build_callbacks, resolve_val_monitor
from src.training.losses import get_loss
from src.training.metrics import get_training_metrics
from src.training.optimizers import get_optimizer
from src.utils.config import ExperimentConfig
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Attention U-Net / U-Net++ from YAML config.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML experiment config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    train_x, train_y = load_split_arrays(cfg.dataset_path, "train", cfg.image_size, cfg.use_clahe)
    val_x, val_y = load_split_arrays(cfg.dataset_path, "val", cfg.image_size, cfg.use_clahe)

    train_ds = make_tf_dataset(train_x, train_y, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_ds = make_tf_dataset(val_x, val_y, cfg.batch_size, shuffle=False)

    model = build_model(cfg)
    model.compile(
        optimizer=get_optimizer(cfg),
        loss=get_loss(cfg.loss),
        metrics=get_training_metrics(cfg),
    )

    monitor, mode = resolve_val_monitor(model, cfg)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg.checkpoint_filename

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / cfg.csv_log_filename

    callbacks = build_callbacks(cfg, monitor, mode, ckpt_path, csv_path)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == "__main__":
    main()
