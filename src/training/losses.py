"""Loss functions used in notebooks (string names → Keras callables).

TODO: A dedicated **Dice loss** was not identified in the inspected notebooks (Dice appears as a **metric** / coefficient). If you add a validated Dice loss from the paper supplement, wire it here.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


def get_loss(loss_name: str) -> Any:
    key = loss_name.strip().lower()
    if key in ("binary_crossentropy", "bce"):
        return tf.keras.losses.BinaryCrossentropy()
    if key in ("binary_focal_crossentropy", "focal", "sigmoid_focal_crossentropy"):
        return "binary_focal_crossentropy"
    # TODO: notebooks did not define a standalone Dice *loss*; only Dice *metric* / coefficient.
    raise ValueError(f"Unsupported loss: {loss_name!r}")
