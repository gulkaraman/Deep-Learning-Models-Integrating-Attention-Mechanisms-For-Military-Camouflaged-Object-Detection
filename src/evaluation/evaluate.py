"""Evaluate a trained checkpoint on a split (default: test)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from src.data.dataset import load_split_arrays, make_tf_dataset
from src.evaluation.model_loading import load_model_for_evaluation
from src.training.losses import get_loss
from src.training.metrics import get_training_metrics
from src.training.optimizers import get_optimizer
from src.utils.config import ExperimentConfig


def _jsonify_keras_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def dice_coefficient_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    yt = y_true.astype(np.float64).ravel()
    yp = y_pred.astype(np.float64).ravel()
    inter = np.sum(yt * yp)
    return float((2.0 * inter + eps) / (np.sum(yt) + np.sum(yp) + eps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    x, y = load_split_arrays(cfg.dataset_path, args.split, cfg.image_size, cfg.use_clahe, verbose=True)
    ds = make_tf_dataset(x, y, cfg.batch_size, shuffle=False)

    model = load_model_for_evaluation(cfg, args.checkpoint)
    model.compile(
        optimizer=get_optimizer(cfg),
        loss=get_loss(cfg.loss),
        metrics=get_training_metrics(cfg),
    )

    keras_metrics = model.evaluate(ds, return_dict=True, verbose=1)

    y_prob = model.predict(ds, verbose=1)
    y_bin = (y_prob > args.threshold).astype(np.float32)
    yt = (y > 0.5).astype(np.int32).ravel()
    yp = y_bin.astype(np.int32).ravel()

    numpy_report = {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1_score": float(f1_score(yt, yp, zero_division=0)),
        "iou": float(jaccard_score(yt, yp, zero_division=0)),
        "dice": dice_coefficient_numpy(y, y_bin),
        "threshold": args.threshold,
        "split": args.split,
    }

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "keras_evaluate": _jsonify_keras_dict(keras_metrics),
        "numpy_flat": numpy_report,
        "checkpoint": str(args.checkpoint),
    }
    out_path = out_dir / f"eval_{args.split}_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
