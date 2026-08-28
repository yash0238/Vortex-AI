"""
Statistical evaluation of TimeGAN synthetic NIFTY-50 data.

Compares:
    Real NIFTY-50 windows
    vs
    TimeGAN generated windows

Input:
    baselines/data/timegan_windows.npy
    baselines/results/timegan_full/generated_samples.npy

Output:
    baselines/results/timegan_full/statistical_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_PATH = PROJECT_ROOT / "baselines" / "data" / "timegan_windows.npy"
FAKE_PATH = (
    PROJECT_ROOT
    / "baselines"
    / "results"
    / "timegan_full"
    / "generated_samples.npy"
)

OUTPUT_DIR = PROJECT_ROOT / "baselines" / "results" / "timegan_full"
OUTPUT_PATH = OUTPUT_DIR / "statistical_metrics.csv"


# ============================================================
# Helper functions
# ============================================================

def validate_data(name, data):
    """Validate the loaded dataset."""

    print(f"{name}: {data.shape}")
    print(f"{name} dtype: {data.dtype}")
    print(f"{name} finite: {np.isfinite(data).all()}")

    if data.ndim != 3:
        raise ValueError(
            f"{name} must be 3-dimensional, got shape {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def flatten_windows(data):
    """
    Convert:
        (samples, sequence_length, features)

    into:
        (samples * sequence_length, features)
    """

    return data.reshape(-1, data.shape[-1])


def safe_kurtosis(data):
    """
    Calculate Fisher kurtosis independently for each stock.
    """

    values = kurtosis(
        data,
        axis=0,
        fisher=True,
        bias=False
    )

    return values


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("TimeGAN Statistical Evaluation")
print("=" * 60)

if not REAL_PATH.exists():
    raise FileNotFoundError(f"Real data not found: {REAL_PATH}")

if not FAKE_PATH.exists():
    raise FileNotFoundError(f"TimeGAN data not found: {FAKE_PATH}")

real = np.load(REAL_PATH)
fake = np.load(FAKE_PATH)

validate_data("Real", real)
validate_data("TimeGAN", fake)


# ============================================================
# Shape validation
# ============================================================

if real.shape != fake.shape:
    raise ValueError(
        f"Shape mismatch:\n"
        f"Real: {real.shape}\n"
        f"TimeGAN: {fake.shape}"
    )

num_samples, seq_len, num_features = real.shape

print("\nDataset configuration:")
print(f"  Samples:         {num_samples}")
print(f"  Sequence length: {seq_len}")
print(f"  Features/stocks: {num_features}")


# ============================================================
# Flatten windows
# ============================================================

real_flat = flatten_windows(real)
fake_flat = flatten_windows(fake)

print("\nFlattened data:")
print(f"  Real:    {real_flat.shape}")
print(f"  TimeGAN: {fake_flat.shape}")


# ============================================================
# Results container
# ============================================================

metrics = []


def add_metric(name, real_value, fake_value, error=None):
    metrics.append(
        {
            "Metric": name,
            "Real": float(real_value),
            "TimeGAN": float(fake_value),
            "Absolute_Error": (
                float(error) if error is not None else np.nan
            ),
        }
    )


# ============================================================
# 1. Mean
# ============================================================

real_mean = real_flat.mean()
fake_mean = fake_flat.mean()

mean_error = abs(real_mean - fake_mean)

add_metric(
    "Mean",
    real_mean,
    fake_mean,
    mean_error
)


# ============================================================
# 2. Standard deviation
# ============================================================

real_std = real_flat.std()
fake_std = fake_flat.std()

std_error = abs(real_std - fake_std)

add_metric(
    "Std",
    real_std,
    fake_std,
    std_error
)


# ============================================================
# 3. Minimum
# ============================================================

real_min = real_flat.min()
fake_min = fake_flat.min()

min_error = abs(real_min - fake_min)

add_metric(
    "Min",
    real_min,
    fake_min,
    min_error
)


# ============================================================
# 4. Maximum
# ============================================================

real_max = real_flat.max()
fake_max = fake_flat.max()

max_error = abs(real_max - fake_max)

add_metric(
    "Max",
    real_max,
    fake_max,
    max_error
)


# ============================================================
# 5. Kurtosis
# ============================================================

real_kurtosis = safe_kurtosis(real_flat)
fake_kurtosis = safe_kurtosis(fake_flat)

valid_kurtosis = (
    np.isfinite(real_kurtosis)
    & np.isfinite(fake_kurtosis)
)

if valid_kurtosis.sum() > 0:

    average_real_kurtosis = np.mean(
        real_kurtosis[valid_kurtosis]
    )

    average_fake_kurtosis = np.mean(
        fake_kurtosis[valid_kurtosis]
    )

    kurtosis_error = np.mean(
        np.abs(
            real_kurtosis[valid_kurtosis]
            - fake_kurtosis[valid_kurtosis]
        )
    )

else:
    average_real_kurtosis = np.nan
    average_fake_kurtosis = np.nan
    kurtosis_error = np.nan

add_metric(
    "Average Kurtosis",
    average_real_kurtosis,
    average_fake_kurtosis,
    kurtosis_error
)


# ============================================================
# 6. Cross-stock correlation
# ============================================================

real_feature_std = np.std(real_flat, axis=0)
fake_feature_std = np.std(fake_flat, axis=0)

valid_features = (
    (real_feature_std > 1e-12)
    & (fake_feature_std > 1e-12)
)

valid_feature_count = int(valid_features.sum())

print(
    f"\nValid features for correlation: "
    f"{valid_feature_count}/{num_features}"
)

if valid_feature_count >= 2:

    real_corr = np.corrcoef(
        real_flat[:, valid_features].T
    )

    fake_corr = np.corrcoef(
        fake_flat[:, valid_features].T
    )

    # Upper triangular portion only.
    # Diagonal is excluded because correlation with itself = 1.
    upper_triangle = np.triu(
        np.ones(real_corr.shape, dtype=bool),
        k=1
    )

    valid_pairs = (
        np.isfinite(real_corr)
        & np.isfinite(fake_corr)
        & upper_triangle
    )

    valid_pair_count = int(valid_pairs.sum())

    if valid_pair_count > 0:

        correlation_error = np.mean(
            np.abs(
                real_corr[valid_pairs]
                - fake_corr[valid_pairs]
            )
        )

    else:
        correlation_error = np.nan

else:

    valid_pair_count = 0
    correlation_error = np.nan


# For the table, Real is shown as 0 because it is the reference.
add_metric(
    "Correlation Error",
    0.0,
    correlation_error,
    correlation_error
)

add_metric(
    "Valid Correlation Pairs",
    valid_pair_count,
    valid_pair_count,
    0.0
)


# ============================================================
# Create DataFrame
# ============================================================

results_df = pd.DataFrame(metrics)


# ============================================================
# Save results
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Print results
# ============================================================

print("\n")
print("=" * 60)
print("TIMEGAN STATISTICAL EVALUATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nCorrelation error:", correlation_error)
print("Valid correlation pairs:", valid_pair_count)

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nEvaluation completed successfully.")