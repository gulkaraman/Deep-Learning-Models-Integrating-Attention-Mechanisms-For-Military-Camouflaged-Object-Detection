"""Keras metrics aligned with ``att unet *.ipynb`` / ``att-unet++*.ipynb``."""

from __future__ import annotations

from typing import Any, List

import tensorflow as tf
from tensorflow import keras as krs
from tensorflow.keras import backend as K

from src.utils.config import ExperimentConfig


def tp_tn_fp_fn(y_true, y_pred, threshold: float = 0.5):
    y_pred_binary = tf.cast(y_pred > threshold, tf.int32)
    y_true_i = tf.cast(y_true > 0.5, tf.int32)
    tp = tf.reduce_sum(tf.cast((y_true_i == 1) & (y_pred_binary == 1), tf.int32))
    tn = tf.reduce_sum(tf.cast((y_true_i == 0) & (y_pred_binary == 0), tf.int32))
    fp = tf.reduce_sum(tf.cast((y_true_i == 0) & (y_pred_binary == 1), tf.int32))
    fn = tf.reduce_sum(tf.cast((y_true_i == 1) & (y_pred_binary == 0), tf.int32))
    return tp, tn, fp, fn


def dice_coefficient(y_true, y_pred, threshold: float = 0.5):
    tp, _tn, fp, fn = tp_tn_fp_fn(y_true, y_pred, threshold)
    tp = tf.cast(tp, tf.float32)
    fp = tf.cast(fp, tf.float32)
    fn = tf.cast(fn, tf.float32)
    return (2 * tp) / (2 * tp + fp + fn + K.epsilon())


def f1_score(y_true, y_pred, threshold: float = 0.5):
    tp, _tn, fp, fn = tp_tn_fp_fn(y_true, y_pred, threshold)
    tp = tf.cast(tp, tf.float32)
    fp = tf.cast(fp, tf.float32)
    fn = tf.cast(fn, tf.float32)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def mcc(y_true, y_pred, threshold: float = 0.5):
    tp, tn, fp, fn = tp_tn_fp_fn(y_true, y_pred, threshold)
    tp = tf.cast(tp, tf.float32)
    tn = tf.cast(tn, tf.float32)
    fp = tf.cast(fp, tf.float32)
    fn = tf.cast(fn, tf.float32)
    numerator = (tp * tn) - (fp * fn)
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return tf.where(denominator > 0, numerator / denominator, tf.constant(0.0, dtype=tf.float32))


def get_training_metrics(cfg: ExperimentConfig) -> List[Any]:
    """Metrics passed to ``model.compile`` (subset configurable)."""
    base: List[Any] = [
        "accuracy",
        krs.metrics.MeanIoU(num_classes=2, name="mean_iou"),
        krs.metrics.IoU(num_classes=2, target_class_ids=[0], name="iou"),
        dice_coefficient,
        f1_score,
        krs.metrics.Recall(name="recall"),
        krs.metrics.Precision(name="precision"),
    ]
    if cfg.include_notebook_aux_metrics:
        base.extend(
            [
                krs.metrics.AUC(name="auc"),
                mcc,
            ]
        )
    return base


def custom_objects_dict() -> dict[str, Any]:
    """For ``keras.models.load_model(..., custom_objects=...)``."""
    return {
        "dice_coefficient": dice_coefficient,
        "f1_score": f1_score,
        "mcc": mcc,
        "tp_tn_fp_fn": tp_tn_fp_fn,
    }
