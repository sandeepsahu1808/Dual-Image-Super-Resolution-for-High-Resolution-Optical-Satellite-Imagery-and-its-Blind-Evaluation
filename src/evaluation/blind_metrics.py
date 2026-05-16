"""Blind / no-reference quality metrics (NIQE, BRISQUE) for SR outputs."""

import os

import numpy as np
import pandas as pd
import piq
import torch
from piq import brisque

from src.data.dataset_loader import load_scene
from src.data.preprocessing import preprocess_triplet

if not hasattr(piq, "niqe"):
    from src.evaluation._piq_niqe import niqe as _niqe_impl

    piq.niqe = _niqe_impl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BLIND_METRICS_LOG_PATH = os.path.join(RESULTS_DIR, "blind_metrics_log.csv")


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert float32 (H, W, 1) in [0, 1] to tensor (1, 1, H, W)."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3 or arr.shape[-1] != 1:
        raise ValueError(f"Expected (H, W, 1) or (H, W), got {arr.shape}")
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def compute_niqe(img: np.ndarray) -> float:
    """NIQE score via piq.niqe() on tensor (1, 1, H, W). Lower is better."""
    tensor = img_to_tensor(img)
    with torch.no_grad():
        score = piq.niqe(tensor, data_range=1.0, reduction="none")
    return float(score.reshape(-1)[0].item())


def compute_brisque(img: np.ndarray) -> float:
    """BRISQUE score via piq.brisque(). Lower is better."""
    tensor = img_to_tensor(img)
    with torch.no_grad():
        score = brisque(tensor, data_range=1.0, reduction="none")
    return float(score.reshape(-1)[0].item())


def evaluate_blind_all(scene_paths: list[str], model) -> pd.DataFrame:
    """
    Predict SR for test scenes and compute blind metrics on outputs.

    Saves per-scene NIQE and BRISQUE to results/blind_metrics_log.csv.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows: list[dict] = []
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        try:
            lr1, lr2, hr = load_scene(scene_path)
            x, _ = preprocess_triplet(lr1, lr2, hr, augment=False)
            sr = model.predict(x[np.newaxis, ...], verbose=0)[0].astype(np.float32)

            rows.append(
                {
                    "scene": scene_name,
                    "scene_path": scene_path,
                    "niqe": compute_niqe(sr),
                    "brisque": compute_brisque(sr),
                }
            )
        except Exception as exc:
            print(f"Skipping {scene_name}: {exc}")

    if not rows:
        raise RuntimeError("No scenes were evaluated successfully.")

    df = pd.DataFrame(rows)
    df.to_csv(BLIND_METRICS_LOG_PATH, index=False)
    print(f"Saved blind metrics to {BLIND_METRICS_LOG_PATH}")
    print(f"Mean NIQE: {df['niqe'].mean():.4f} (lower is better)")
    print(f"Mean BRISQUE: {df['brisque'].mean():.4f} (lower is better)")
    return df


if __name__ == "__main__":
    from src.data.dataset_loader import prepare_dataset
    from src.models.srcnn import build_srcnn

    _, _, test_scenes = prepare_dataset()
    model = build_srcnn()
    weights_path = os.path.join(PROJECT_ROOT, "weights", "best_model.h5")
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: best_model.h5 not found — using untrained weights.")

    evaluate_blind_all(test_scenes, model)
