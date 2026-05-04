"""ResNet-50 Attention U-Net++ (matches ``att-unet++*.ipynb`` graph)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50

from src.models.attention_blocks import attention_gate_unet_plus_plus as attention_gate

NB_FILTER = [32, 64, 128, 256]


def conv_block(
    input_tensor,
    num_filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    activation: str = "relu",
    kernel_initializer: str = "he_normal",
    padding: str = "same",
    *,
    use_dropout: bool = True,
    dropout_rate: float = 0.5,
):
    """Two Conv2D blocks with BN / activation / optional dropout (notebook structure)."""
    x = layers.Conv2D(num_filters, kernel_size, kernel_initializer=kernel_initializer, padding=padding)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if use_dropout and dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(num_filters, kernel_size, kernel_initializer=kernel_initializer, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if use_dropout and dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_attention_unet_plus_plus(
    input_shape: tuple[int, int, int],
    *,
    num_classes: int = 1,
    dropout_rate: float = 0.2,
    resnet_trainable: bool = True,
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)
    base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = resnet_trainable

    skip_1_out = base_model.get_layer("conv1_relu").output
    skip_2_out = base_model.get_layer("conv2_block3_out").output
    skip_3_out = base_model.get_layer("conv3_block4_out").output
    skip_4_out = base_model.get_layer("conv4_block6_out").output
    bridge_out = base_model.get_layer("conv5_block3_out").output

    X_00 = skip_1_out

    X_10 = skip_2_out
    Up_10_to_01 = layers.Conv2DTranspose(NB_FILTER[0], (2, 2), strides=(2, 2), padding="same", name="up_10_to_01")(X_10)
    Att_X_00_from_10 = attention_gate(X_00, Up_10_to_01, inter_channel=NB_FILTER[0] // 2, name="ag_00_from_10")
    Merge_01 = layers.concatenate([Up_10_to_01, Att_X_00_from_10], axis=-1, name="merge_01")
    X_01 = conv_block(Merge_01, NB_FILTER[0], dropout_rate=dropout_rate)

    X_20 = skip_3_out
    Up_20_to_11 = layers.Conv2DTranspose(NB_FILTER[1], (2, 2), strides=(2, 2), padding="same", name="up_20_to_11")(X_20)
    Att_X_10_from_20 = attention_gate(X_10, Up_20_to_11, inter_channel=NB_FILTER[1] // 2, name="ag_10_from_20")
    Merge_11 = layers.concatenate([Up_20_to_11, Att_X_10_from_20], axis=-1, name="merge_11")
    X_11 = conv_block(Merge_11, NB_FILTER[1], dropout_rate=dropout_rate)

    Up_11_to_02 = layers.Conv2DTranspose(NB_FILTER[0], (2, 2), strides=(2, 2), padding="same", name="up_11_to_02")(X_11)
    Att_X_00_from_11 = attention_gate(X_00, Up_11_to_02, inter_channel=NB_FILTER[0] // 2, name="ag_00_from_11")
    Merge_02 = layers.concatenate([Up_11_to_02, Att_X_00_from_11, X_01], axis=-1, name="merge_02")
    X_02 = conv_block(Merge_02, NB_FILTER[0], dropout_rate=dropout_rate)

    X_30 = skip_4_out
    Up_30_to_21 = layers.Conv2DTranspose(NB_FILTER[2], (2, 2), strides=(2, 2), padding="same", name="up_30_to_21")(X_30)
    Att_X_20_from_30 = attention_gate(X_20, Up_30_to_21, inter_channel=NB_FILTER[2] // 2, name="ag_20_from_30")
    Merge_21 = layers.concatenate([Up_30_to_21, Att_X_20_from_30], axis=-1, name="merge_21")
    X_21 = conv_block(Merge_21, NB_FILTER[2], dropout_rate=dropout_rate)

    Up_21_to_12 = layers.Conv2DTranspose(NB_FILTER[1], (2, 2), strides=(2, 2), padding="same", name="up_21_to_12")(X_21)
    Att_X_10_from_21 = attention_gate(X_10, Up_21_to_12, inter_channel=NB_FILTER[1] // 2, name="ag_10_from_21")
    Merge_12 = layers.concatenate([Up_21_to_12, Att_X_10_from_21, X_11], axis=-1, name="merge_12")
    X_12 = conv_block(Merge_12, NB_FILTER[1], dropout_rate=dropout_rate)

    Up_12_to_03 = layers.Conv2DTranspose(NB_FILTER[0], (2, 2), strides=(2, 2), padding="same", name="up_12_to_03")(X_12)
    Att_X_00_from_12 = attention_gate(X_00, Up_12_to_03, inter_channel=NB_FILTER[0] // 2, name="ag_00_from_12")
    Merge_03 = layers.concatenate([Up_12_to_03, Att_X_00_from_12, X_01, X_02], axis=-1, name="merge_03")
    X_03 = conv_block(Merge_03, NB_FILTER[0], dropout_rate=dropout_rate)

    X_40 = bridge_out
    Up_40_to_31 = layers.Conv2DTranspose(NB_FILTER[3], (2, 2), strides=(2, 2), padding="same", name="up_40_to_31")(X_40)
    Att_X_30_from_40 = attention_gate(X_30, Up_40_to_31, inter_channel=NB_FILTER[3] // 2, name="ag_30_from_40")
    Merge_31 = layers.concatenate([Up_40_to_31, Att_X_30_from_40], axis=-1, name="merge_31")
    X_31 = conv_block(Merge_31, NB_FILTER[3], dropout_rate=dropout_rate)

    Up_31_to_22 = layers.Conv2DTranspose(NB_FILTER[2], (2, 2), strides=(2, 2), padding="same", name="up_31_to_22")(X_31)
    Att_X_20_from_31 = attention_gate(X_20, Up_31_to_22, inter_channel=NB_FILTER[2] // 2, name="ag_20_from_31")
    Merge_22 = layers.concatenate([Up_31_to_22, Att_X_20_from_31, X_21], axis=-1, name="merge_22")
    X_22 = conv_block(Merge_22, NB_FILTER[2], dropout_rate=dropout_rate)

    Up_22_to_13 = layers.Conv2DTranspose(NB_FILTER[1], (2, 2), strides=(2, 2), padding="same", name="up_22_to_13")(X_22)
    Att_X_10_from_22 = attention_gate(X_10, Up_22_to_13, inter_channel=NB_FILTER[1] // 2, name="ag_10_from_22")
    Merge_13 = layers.concatenate([Up_22_to_13, Att_X_10_from_22, X_11, X_12], axis=-1, name="merge_13")
    X_13 = conv_block(Merge_13, NB_FILTER[1], dropout_rate=dropout_rate)

    Up_13_to_04 = layers.Conv2DTranspose(NB_FILTER[0], (2, 2), strides=(2, 2), padding="same", name="up_13_to_04")(X_13)
    Att_X_00_from_13 = attention_gate(X_00, Up_13_to_04, inter_channel=NB_FILTER[0] // 2, name="ag_00_from_13")
    Merge_04 = layers.concatenate([Up_13_to_04, Att_X_00_from_13, X_01, X_02, X_03], axis=-1, name="merge_04")
    X_04 = conv_block(Merge_04, NB_FILTER[0], dropout_rate=dropout_rate)

    final_upsample = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="final_upsample")(X_04)
    output_activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, (1, 1), activation=output_activation, name="final_output")(final_upsample)

    return Model(inputs=[inputs], outputs=[outputs], name="ResNet50_Attention_UNet_Plus_Plus")
