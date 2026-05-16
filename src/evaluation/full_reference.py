"""Full-reference image quality metrics (PSNR, SSIM, MSE, RMSE)."""

import os

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.data.dataset_loader import load_scene
from src.data.preprocessing import preprocess_triplet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_LOG_PATH = os.path.join(RESULTS_DIR, "metrics_log.csv")


def _squeeze_channels(image: np.ndarray) -> np.ndarray:
    """Drop singleton channel dim for 2D metrics when shape is (H, W, 1)."""
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    return image


def compute_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    """Peak signal-to-noise ratio (data in [0, 1])."""
    return float(
        peak_signal_noise_ratio(
            _squeeze_channels(hr),
            _squeeze_channels(sr),
            data_range=1.0,
        )
    )


def compute_ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    """Structural similarity index (data in [0, 1])."""
    hr_2d = _squeeze_channels(hr)
    sr_2d = _squeeze_channels(sr)
    return float(structural_similarity(hr_2d, sr_2d, data_range=1.0))


def compute_mse(hr: np.ndarray, sr: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((hr.astype(np.float64) - sr.astype(np.float64)) ** 2))


def compute_rmse(hr: np.ndarray, sr: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(compute_mse(hr, sr)))


def evaluate_scene(hr: np.ndarray, sr: np.ndarray) -> dict[str, float]:
    """Compute PSNR, SSIM, MSE, and RMSE for one HR/SR pair."""
    return {
        "psnr": compute_psnr(hr, sr),
        "ssim": compute_ssim(hr, sr),
        "mse": compute_mse(hr, sr),
        "rmse": compute_rmse(hr, sr),
    }


def evaluate_all(scene_paths: list[str], model) -> pd.DataFrame:
    """
    Run full-reference evaluation on all scenes.

    Loads each scene, preprocesses to model input, predicts SR, and logs metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows: list[dict] = []
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        try:
            lr1, lr2, hr = load_scene(scene_path)
            x, y = preprocess_triplet(lr1, lr2, hr, augment=False)
            sr = model.predict(x[np.newaxis, ...], verbose=0)[0]

            metrics = evaluate_scene(y, sr.astype(np.float32))
            rows.append({"scene": scene_name, "scene_path": scene_path, **metrics})
        except Exception as exc:
            print(f"Skipping {scene_name}: {exc}")

    if not rows:
        raise RuntimeError("No scenes were evaluated successfully.")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_LOG_PATH, index=False)
    print(f"Saved metrics to {METRICS_LOG_PATH}")
    print(f"Mean PSNR: {df['psnr'].mean():.4f} dB")
    print(f"Mean SSIM: {df['ssim'].mean():.4f}")
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

    evaluate_all(test_scenes, model)
