"""Spearman rank correlation between blind and full-reference SR metrics."""

import os

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_LOG_PATH = os.path.join(RESULTS_DIR, "metrics_log.csv")
BLIND_METRICS_LOG_PATH = os.path.join(RESULTS_DIR, "blind_metrics_log.csv")
CORRELATION_REPORT_PATH = os.path.join(RESULTS_DIR, "correlation_report.csv")
PS12_CORRELATION_TARGET = 0.7

CORRELATION_PAIRS = [
    ("niqe", "psnr"),
    ("niqe", "ssim"),
    ("niqe", "mse"),
    ("niqe", "rmse"),
    ("brisque", "psnr"),
    ("brisque", "ssim"),
    ("brisque", "mse"),
    ("brisque", "rmse"),
]


def load_merged_metrics(
    metrics_path: str = METRICS_LOG_PATH,
    blind_path: str = BLIND_METRICS_LOG_PATH,
) -> pd.DataFrame:
    """Load full-reference and blind logs and merge on scene."""
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Full-reference log not found: {metrics_path}")
    if not os.path.isfile(blind_path):
        raise FileNotFoundError(f"Blind metrics log not found: {blind_path}")

    full_ref = pd.read_csv(metrics_path)
    blind = pd.read_csv(blind_path)

    merged = pd.merge(full_ref, blind, on="scene", how="inner", suffixes=("_full", "_blind"))
    if merged.empty:
        raise ValueError("No overlapping scenes after merge on 'scene'.")
    return merged


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation for each blind vs reference pair."""
    rows: list[dict] = []
    for blind_col, ref_col in CORRELATION_PAIRS:
        if blind_col not in merged.columns:
            raise KeyError(f"Missing blind metric column: {blind_col}")
        if ref_col not in merged.columns:
            raise KeyError(f"Missing reference metric column: {ref_col}")

        rho, p_value = spearmanr(merged[blind_col], merged[ref_col])
        strength = abs(float(rho))
        rows.append(
            {
                "blind_metric": blind_col.upper(),
                "reference_metric": ref_col.upper(),
                "comparison": f"{blind_col.upper()} vs {ref_col.upper()}",
                "spearman_rho": float(rho),
                "correlation_strength": strength,
                "meets_ps12_target": strength >= PS12_CORRELATION_TARGET,
                "p_value": float(p_value),
                "n_scenes": len(merged),
            }
        )
    return pd.DataFrame(rows)


def run_correlation_analysis(
    metrics_path: str = METRICS_LOG_PATH,
    blind_path: str = BLIND_METRICS_LOG_PATH,
    output_path: str = CORRELATION_REPORT_PATH,
) -> pd.DataFrame:
    """Merge metric logs, compute correlations, print and save report."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    merged = load_merged_metrics(metrics_path, blind_path)
    report = compute_correlations(merged)
    report.to_csv(output_path, index=False)

    print(f"Merged {len(merged)} scenes from metrics logs.\n")
    print(
        f"Spearman rank correlation (raw NIQE/BRISQUE vs full-reference). "
        f"PS-12 target: |ρ| ≥ {PS12_CORRELATION_TARGET}\n"
    )
    print(
        report[
            [
                "comparison",
                "spearman_rho",
                "correlation_strength",
                "meets_ps12_target",
                "p_value",
                "n_scenes",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved report to {output_path}")
    return report


if __name__ == "__main__":
    run_correlation_analysis()
