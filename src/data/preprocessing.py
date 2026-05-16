"""Preprocess PROBA-V (LR1, LR2, HR) triplets for dual-input super-resolution."""

import numpy as np
import cv2

HR_SIZE = (384, 384)


def upsample_bicubic(image: np.ndarray, size: tuple[int, int] = HR_SIZE) -> np.ndarray:
    """Upsample a 2D float image to `size` (height, width) with bicubic interpolation."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    height, width = size
    return cv2.resize(
        image.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )


def _to_channel_last(image: np.ndarray) -> np.ndarray:
    """Ensure image has shape (H, W, 1)."""
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    if image.ndim == 3 and image.shape[-1] == 1:
        return image
    raise ValueError(f"Expected (H, W) or (H, W, 1), got {image.shape}")


def apply_random_flips(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random horizontal and vertical flips to X and y."""
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=1)
    if rng.random() < 0.5:
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def preprocess_triplet(
    lr1: np.ndarray,
    lr2: np.ndarray,
    hr: np.ndarray,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build model input/target from one scene triplet.

    Parameters
    ----------
    lr1, lr2 : np.ndarray
        Low-resolution frames, shape (128, 128), values in [0, 1].
    hr : np.ndarray
        High-resolution ground truth, shape (384, 384), values in [0, 1].
    augment : bool
        If True, apply random horizontal and vertical flips (training only).
    rng : np.random.Generator, optional
        Random generator for reproducible augmentation.

    Returns
    -------
    X : np.ndarray, shape (384, 384, 2)
        Bicubic-upsampled, stacked LR pair.
    y : np.ndarray, shape (384, 384, 1)
        HR target.
    """
    lr1_up = upsample_bicubic(lr1, HR_SIZE)
    lr2_up = upsample_bicubic(lr2, HR_SIZE)
    x = np.stack([lr1_up, lr2_up], axis=-1).astype(np.float32)
    y = _to_channel_last(hr.astype(np.float32))

    if augment:
        x, y = apply_random_flips(x, y, rng=rng)

    return x, y


if __name__ == "__main__":
    from dataset_loader import load_scene, prepare_dataset

    train_scenes, _, _ = prepare_dataset()
    lr1, lr2, hr = load_scene(train_scenes[0])

    x, y = preprocess_triplet(lr1, lr2, hr, augment=False)
    x_aug, y_aug = preprocess_triplet(lr1, lr2, hr, augment=True, rng=np.random.default_rng(42))

    print(f"LR shapes: {lr1.shape}, {lr2.shape} | HR: {hr.shape}")
    print(f"X {x.shape} [{x.min():.4f}, {x.max():.4f}]")
    print(f"y {y.shape} [{y.min():.4f}, {y.max():.4f}]")
    print(f"Augmented X {x_aug.shape}, y {y_aug.shape}")
