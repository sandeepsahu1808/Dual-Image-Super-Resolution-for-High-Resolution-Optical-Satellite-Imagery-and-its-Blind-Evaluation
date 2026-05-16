"""
PIQ-compatible NIQE (piq 0.8.0 does not ship niqe; registered on piq at import).

Based on BasicSR / official NIQE release parameters.
"""

from __future__ import annotations

import math
import os
from typing import Union

import numpy as np
import torch
from piq.functional import imresize
from piq.utils import _reduce, _validate_input

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "niqe_pris_params.npz")
_PARAMS: dict[str, np.ndarray] | None = None


def _load_params() -> dict[str, np.ndarray]:
    global _PARAMS
    if _PARAMS is None:
        data = np.load(_PARAMS_PATH)
        _PARAMS = {
            "mu_pris_param": data["mu_pris_param"],
            "cov_pris_param": data["cov_pris_param"],
            "gaussian_window": data["gaussian_window"],
        }
    return _PARAMS


def _convolve_nearest(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = torch.from_numpy(image).view(1, 1, *image.shape).double()
    k = torch.from_numpy(kernel).view(1, 1, *kernel.shape).double()
    pad = kernel.shape[0] // 2
    out = torch.nn.functional.conv2d(img, k, padding=pad)
    return out[0, 0].numpy()


def _estimate_aggd_param(block: np.ndarray) -> tuple[float, float, float]:
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(_gamma(gam_reciprocal * 2)) / (
        _gamma(gam_reciprocal) * _gamma(gam_reciprocal * 3)
    )

    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
    alpha = float(gam[np.argmin((r_gam - rhatnorm) ** 2)])
    beta_l = left_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    beta_r = right_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    return alpha, float(beta_l), float(beta_r)


def _gamma(z: np.ndarray | float) -> np.ndarray | float:
    return np.vectorize(math.gamma)(z)


def _compute_feature(block: np.ndarray) -> list[float]:
    feat: list[float] = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted)
        mean = (beta_r - beta_l) * (math.gamma(2 / alpha) / math.gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def _niqe_gray(
    img: np.ndarray,
    mu_pris_param: np.ndarray,
    cov_pris_param: np.ndarray,
    gaussian_window: np.ndarray,
    block_size_h: int = 96,
    block_size_w: int = 96,
) -> float:
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    distparam: list[np.ndarray] = []
    for scale in (1, 2):
        mu = _convolve_nearest(img, gaussian_window)
        sigma = np.sqrt(
            np.abs(_convolve_nearest(np.square(img), gaussian_window) - np.square(mu))
        )
        img_normalized = (img - mu) / (sigma + 1)

        feat_blocks = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat_blocks.append(_compute_feature(block))
        distparam.append(np.array(feat_blocks))

        if scale == 1:
            tensor = torch.from_numpy(img / 255.0).view(1, 1, img.shape[0], img.shape[1])
            resized = imresize(tensor, sizes=(img.shape[0] // 2, img.shape[1] // 2))
            img = resized.numpy()[0, 0] * 255.0

    distparam_arr = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam_arr, axis=0)
    distparam_no_nan = distparam_arr[~np.isnan(distparam_arr).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = mu_pris_param - mu_distparam
    quality = float(np.sqrt(diff @ invcov_param @ diff.T))
    return quality


def niqe(
    x: torch.Tensor,
    data_range: Union[int, float] = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Natural Image Quality Evaluator (PIQ-style API).

    Args:
        x: Tensor (N, C, H, W) in [0, data_range].
    """
    _validate_input([x], dim_range=(4, 4), data_range=(0, data_range))
    params = _load_params()
    scores = []
    for i in range(x.shape[0]):
        img = x[i, 0].detach().cpu().numpy() * float(data_range) * 255.0
        img = np.round(img.astype(np.float64))
        score = _niqe_gray(
            img,
            params["mu_pris_param"],
            params["cov_pris_param"],
            params["gaussian_window"],
        )
        scores.append(score)
    return _reduce(torch.tensor(scores, dtype=x.dtype, device=x.device), reduction)
