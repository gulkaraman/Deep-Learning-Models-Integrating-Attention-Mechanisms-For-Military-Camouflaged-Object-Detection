"""ResNet-50 Attention U-Net (decoder matches ``att unet *.ipynb``)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input

from src.models.attention_blocks import attention_gate_unet


def build_attention_unet(
    input_shape: tuple[int, int, int],
    *,
    num_classes: int = 1,
    dropout_rate: float = 0.2,
    freeze_resnet_layers: int = 50,
) -> tf.keras.Model:
    """
    Build Attention U-Net with ResNet-50 encoder and attention-gated skip connections.

    ``freeze_resnet_layers`` follows the notebook convention of freezing the first N
    layers of the ImageNet backbone (``0`` disables freezing).
    """
    inputs = Input(input_shape)
    resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    if freeze_resnet_layers > 0:
        for layer in resnet.layers[:freeze_resnet_layers]:
            layer.trainable = False

    c1 = resnet.get_layer("conv1_relu").output
    c2 = resnet.get_layer("conv2_block3_out").output
    c3 = resnet.get_layer("conv3_block4_out").output
    c4 = resnet.get_layer("conv4_block6_out").output
    c5 = resnet.get_layer("conv5_block3_out").output

    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = attention_gate_unet(c4, u6, 512)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(u6)
    u6 = layers.BatchNormalization()(u6)
    u6 = layers.Dropout(dropout_rate)(u6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(u6)
    u7 = attention_gate_unet(c3, u7, 256)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(u7)
    u7 = layers.BatchNormalization()(u7)
    u7 = layers.Dropout(dropout_rate)(u7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u7)
    u8 = attention_gate_unet(c2, u8, 128)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(u8)
    u8 = layers.BatchNormalization()(u8)
    u8 = layers.Dropout(dropout_rate)(u8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u8)
    u9 = attention_gate_unet(c1, u9, 64)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(u9)
    u9 = layers.BatchNormalization()(u9)
    u9 = layers.Dropout(dropout_rate)(u9)

    u10 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(u9)
    u10 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(u10)
    u10 = layers.BatchNormalization()(u10)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(u10)
    return Model(inputs=[inputs], outputs=[outputs], name="AttentionUNet_ResNet50")
