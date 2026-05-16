"""PROBA-V RED band loader: quality-mask LR pair selection and train/val/test split."""

import os
import random

import numpy as np
from PIL import Image

NUM_LR_FRAMES = 18
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "probav_data",
)
TRAIN_RED_DIR = os.path.join(DEFAULT_DATA_ROOT, "train", "RED")


def discover_scenes(red_dir: str = TRAIN_RED_DIR) -> list[str]:
    """Return sorted absolute paths to scene directories under train/RED/."""
    if not os.path.isdir(red_dir):
        raise FileNotFoundError(f"RED train directory not found: {red_dir}")

    scenes = []
    for name in sorted(os.listdir(red_dir)):
        scene_path = os.path.join(red_dir, name)
        hr_path = os.path.join(scene_path, "HR.png")
        if os.path.isdir(scene_path) and os.path.isfile(hr_path):
            scenes.append(os.path.abspath(scene_path))
    return scenes


def qm_good_pixel_fraction(qm_path: str) -> float:
    """Fraction of QM pixels with value > 0 (good pixels)."""
    qm = np.array(Image.open(qm_path))
    if qm.size == 0:
        return 0.0
    return float(np.count_nonzero(qm > 0) / qm.size)


def select_best_lr_indices(scene_dir: str, count: int = 2) -> list[int]:
    """Pick LR indices with highest QM good-pixel fractions (ties: lower index first)."""
    scored = []
    for idx in range(NUM_LR_FRAMES):
        qm_path = os.path.join(scene_dir, f"QM{idx:03d}.png")
        if not os.path.isfile(qm_path):
            continue
        scored.append((qm_good_pixel_fraction(qm_path), idx))

    if len(scored) < count:
        raise ValueError(f"Scene {scene_dir} has fewer than {count} QM/LR pairs")

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in scored[:count]]


def load_image_normalized(image_path: str) -> np.ndarray:
    """Load a single-band image as float32 in [0, 1]."""
    arr = np.array(Image.open(image_path))
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    max_val = 1.0 if arr.max() <= 1.0 else 65535.0
    return arr / max_val


def load_scene(scene_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (LR1, LR2, HR) for one scene.

    LR1 and LR2 are the two LR frames with the best quality masks.
    All arrays are float32, normalized to [0, 1], shape (H, W).
    """
    best_indices = select_best_lr_indices(scene_dir, count=2)
    lr1 = load_image_normalized(os.path.join(scene_dir, f"LR{best_indices[0]:03d}.png"))
    lr2 = load_image_normalized(os.path.join(scene_dir, f"LR{best_indices[1]:03d}.png"))
    hr = load_image_normalized(os.path.join(scene_dir, "HR.png"))
    return lr1, lr2, hr


def split_scenes(
    scenes: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle scenes and split into train, val, and test path lists."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    scenes = list(scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)

    n = len(scenes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val : n_train + n_val + n_test]
    return train_scenes, val_scenes, test_scenes


def prepare_dataset(
    red_dir: str = TRAIN_RED_DIR,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Discover all RED train scenes and split 70/15/15.

    Returns
    -------
    train_scenes, val_scenes, test_scenes
        Lists of absolute scene directory paths.
    """
    scenes = discover_scenes(red_dir)
    return split_scenes(scenes, seed=seed)


if __name__ == "__main__":
    train_scenes, val_scenes, test_scenes = prepare_dataset()
    print(f"Total scenes: {len(train_scenes) + len(val_scenes) + len(test_scenes)}")
    print(f"Train: {len(train_scenes)}, Val: {len(val_scenes)}, Test: {len(test_scenes)}")

    sample = train_scenes[0]
    lr1, lr2, hr = load_scene(sample)
    print(f"Sample scene: {os.path.basename(sample)}")
    print(f"  LR1 {lr1.shape} [{lr1.min():.4f}, {lr1.max():.4f}]")
    print(f"  LR2 {lr2.shape} [{lr2.min():.4f}, {lr2.max():.4f}]")
    print(f"  HR  {hr.shape} [{hr.min():.4f}, {hr.max():.4f}]")
    print(f"  Best LR indices: {select_best_lr_indices(sample)}")
