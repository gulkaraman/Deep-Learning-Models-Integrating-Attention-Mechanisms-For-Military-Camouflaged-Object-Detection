"""Attention gate variants matching the two notebook families."""

from __future__ import annotations

from tensorflow.keras import layers


def attention_gate_unet(x, g, inter_channel: int):
    """Attention gate as in ``att unet *.ipynb`` (unnamed layers)."""
    theta_x = layers.Conv2D(inter_channel, (1, 1))(x)
    phi_g = layers.Conv2D(inter_channel, (1, 1))(g)
    f = layers.Activation("relu")(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1))(f)
    rate = layers.Activation("sigmoid")(psi_f)
    return layers.multiply([x, rate])


def attention_gate_unet_plus_plus(x, g, inter_channel: int, name: str):
    """Attention gate as in ``att-unet++*.ipynb`` (named layers, bias pattern preserved)."""
    theta_x = layers.Conv2D(
        inter_channel,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=name + "_theta_x",
    )(x)
    phi_g = layers.Conv2D(
        inter_channel,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        name=name + "_phi_g",
    )(g)
    f = layers.Activation("relu")(layers.Add(name=name + "_add")([theta_x, phi_g]))
    psi_f = layers.Conv2D(
        1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        name=name + "_psi_f",
    )(f)
    alpha = layers.Activation("sigmoid", name=name + "_alpha")(psi_f)
    return layers.Multiply(name=name + "_attended_x")([x, alpha])
