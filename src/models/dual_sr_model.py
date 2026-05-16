"""Dual-branch SRCNN with shared weights for LR1 + LR2 fusion (ISRO PS-12)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.srcnn import psnr_metric


def build_dual_sr(
    input_shape: tuple[int, int, int] = (384, 384, 2),
    learning_rate: float = 1e-4,
) -> keras.Model:
    """
    Build dual-branch super-resolution model with shared conv weights.

    Branch 1 (LR1) and Branch 2 (LR2) each pass through the same Conv2D layers,
    then features are concatenated and fused to predict HR output.
    """
    inputs = keras.Input(shape=input_shape, name="lr_stack")

    lr1 = layers.Lambda(lambda t: t[..., 0:1], name="lr1_slice")(inputs)
    lr2 = layers.Lambda(lambda t: t[..., 1:2], name="lr2_slice")(inputs)

    branch_conv1 = layers.Conv2D(
        64, 3, padding="same", activation="relu", name="branch_conv1"
    )
    branch_conv2 = layers.Conv2D(
        64, 3, padding="same", activation="relu", name="branch_conv2"
    )

    branch1 = branch_conv2(branch_conv1(lr1))
    branch2 = branch_conv2(branch_conv1(lr2))

    fused = layers.Concatenate(name="branch_concat")([branch1, branch2])
    x = layers.Conv2D(64, 1, padding="same", activation="relu", name="fusion_conv1")(fused)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="fusion_conv2")(x)
    outputs = layers.Conv2D(1, 3, padding="same", activation="linear", name="sr_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dual_sr")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[psnr_metric],
    )

    model.summary()
    return model


if __name__ == "__main__":
    build_dual_sr()
