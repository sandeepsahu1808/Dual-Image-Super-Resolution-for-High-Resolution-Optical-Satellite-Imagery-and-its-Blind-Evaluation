"""Combined training losses for super-resolution."""

import tensorflow as tf

MSE_WEIGHT = 0.7
SSIM_WEIGHT = 0.3
MAX_VAL = 1.0


def perceptual_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combined loss: 70% MSE + 30% (1 - SSIM).

    Expects y_true, y_pred in [0, 1], shape (batch, H, W, channels).
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=MAX_VAL))
    ssim_loss = 1.0 - ssim
    return MSE_WEIGHT * mse + SSIM_WEIGHT * ssim_loss
