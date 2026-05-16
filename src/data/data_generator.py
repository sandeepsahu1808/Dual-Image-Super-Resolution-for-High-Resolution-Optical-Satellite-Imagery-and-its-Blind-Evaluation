"""Keras Sequence generator for PROBA-V dual-input super-resolution."""

import math

import numpy as np
import tensorflow as tf

from src.data.dataset_loader import load_scene
from src.data.preprocessing import preprocess_triplet

logger = tf.get_logger()


class PROBAVDataGenerator(tf.keras.utils.Sequence):
    """
    Batch loader for (LR1, LR2) -> HR super-resolution training.

    Each sample is preprocessed to X (384, 384, 2) and y (384, 384, 1).
    """

    def __init__(
        self,
        scene_paths: list[str],
        batch_size: int = 16,
        augment: bool = False,
    ):
        self.scene_paths = list(scene_paths)
        self.batch_size = batch_size
        self.augment = augment
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        if not self.scene_paths:
            return 0
        return math.ceil(len(self.scene_paths) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.augment:
            np.random.shuffle(self.scene_paths)

    def _load_sample(self, scene_path: str) -> tuple[np.ndarray, np.ndarray]:
        lr1, lr2, hr = load_scene(scene_path)
        return preprocess_triplet(
            lr1,
            lr2,
            hr,
            augment=self.augment,
            rng=self._rng,
        )

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        batch_paths = self.scene_paths[start : start + self.batch_size]

        x_batch: list[np.ndarray] = []
        y_batch: list[np.ndarray] = []

        for scene_path in batch_paths:
            try:
                x, y = self._load_sample(scene_path)
                x_batch.append(x)
                y_batch.append(y)
            except Exception as exc:
                logger.warning("Skipping scene %s: %s", scene_path, exc)

        if not x_batch:
            raise RuntimeError(
                f"Batch {index} has no valid scenes "
                f"(paths {start}–{start + len(batch_paths) - 1})."
            )

        return np.stack(x_batch, axis=0), np.stack(y_batch, axis=0)


if __name__ == "__main__":
    from src.data.dataset_loader import prepare_dataset

    train_scenes, val_scenes, _ = prepare_dataset()
    train_gen = PROBAVDataGenerator(train_scenes[:32], batch_size=8, augment=True)
    val_gen = PROBAVDataGenerator(val_scenes[:16], batch_size=4, augment=False)

    print(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")
    x, y = train_gen[0]
    print(f"Batch 0 — X: {x.shape}, y: {y.shape}, dtypes: {x.dtype}, {y.dtype}")
    xv, yv = val_gen[0]
    print(f"Val batch 0 — X: {xv.shape}, y: {yv.shape}")
