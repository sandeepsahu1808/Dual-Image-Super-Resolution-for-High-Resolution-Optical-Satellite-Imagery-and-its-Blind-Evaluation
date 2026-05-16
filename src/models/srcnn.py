"""SRCNN baseline for dual-input super-resolution (ISRO PS-12)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def psnr_metric(y_true, y_pred):
    """Peak signal-to-noise ratio for images in [0, 1] (max_val=1.0)."""
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def build_srcnn(
    input_shape: tuple[int, int, int] = (384, 384, 2),
    learning_rate: float = 1e-4,
) -> keras.Model:
    """
    Build and compile SRCNN for stacked LR1+LR2 input.

    Architecture: Conv 9×9 (64) → Conv 1×1 (32) → Conv 5×5 (1, linear).
    """
    inputs = keras.Input(shape=input_shape, name="lr_stack")
    x = layers.Conv2D(64, 9, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.Conv2D(32, 1, padding="same", activation="relu", name="conv2")(x)
    outputs = layers.Conv2D(1, 5, padding="same", activation="linear", name="sr_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="srcnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[psnr_metric],
    )

    model.summary()
    return model


if __name__ == "__main__":
    build_srcnn()
