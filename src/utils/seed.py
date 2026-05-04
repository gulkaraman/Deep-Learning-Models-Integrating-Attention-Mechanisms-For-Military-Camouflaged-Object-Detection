"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds (best-effort)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
