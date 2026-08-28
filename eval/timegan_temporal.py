"""
Temporal evaluation of TimeGAN synthetic NIFTY-50 data.

Compares:
    Real NIFTY-50 windows
    vs
    TimeGAN generated windows

Metrics:
    1. Per-stock volatility
    2. Volatility absolute error
    3. Autocorrelation at lags 1, 5, 10, 20
    4. ACF absolute error

Output:
    baselines/results/timegan_full/temporal_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_PATH = (
    PROJECT_ROOT
    / "baselines"
    / "data"
    / "timegan_windows.npy"
)

FAKE_PATH = (
    PROJECT_ROOT
    / "baselines"
    / "results"
    / "timegan_full"
    / "generated_samples.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "baselines"
    / "results"
    / "timegan_full"
)

OUTPUT_PATH = OUTPUT_DIR / "temporal_metrics.csv"


# ============================================================
# Helper functions
# ============================================================

def autocorrelation(series, lag):
    """
    Calculate Pearson autocorrelation for a 1D series.
    """

    if lag >= len(series):
        return np.nan

    x = series[:-lag]
    y = series[lag:]

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std <= 1e-12 or y_std <= 1e-12:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def calculate_acf(data, lag):
    """
    Calculate average ACF across:
        samples × stocks

    Each sequence is evaluated independently.
    """

    values = []

    for sample in data:
        for stock in range(data.shape[2]):
            value = autocorrelation(
                sample[:, stock],
                lag
            )

            if np.isfinite(value):
                values.append(value)

    if not values:
        return np.nan

    return float(np.mean(values))


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("TimeGAN Temporal Evaluation")
print("=" * 60)

if not REAL_PATH.exists():
    raise FileNotFoundError(
        f"Real data not found: {REAL_PATH}"
    )

if not FAKE_PATH.exists():
    raise FileNotFoundError(
        f"TimeGAN data not found: {FAKE_PATH}"
    )

real = np.load(REAL_PATH)
fake = np.load(FAKE_PATH)


# ============================================================
# Validate
# ============================================================

print("\nReal shape:", real.shape)
print("TimeGAN shape:", fake.shape)

if real.ndim != 3:
    raise ValueError(
        f"Real data must be 3D, got {real.shape}"
    )

if fake.ndim != 3:
    raise ValueError(
        f"TimeGAN data must be 3D, got {fake.shape}"
    )

if real.shape != fake.shape:
    raise ValueError(
        "Real and TimeGAN shapes do not match:\n"
        f"Real: {real.shape}\n"
        f"TimeGAN: {fake.shape}"
    )

if not np.isfinite(real).all():
    raise ValueError("Real data contains NaN or Inf.")

if not np.isfinite(fake).all():
    raise ValueError(
        "TimeGAN data contains NaN or Inf."
    )


# ============================================================
# Configuration
# ============================================================

num_samples = real.shape[0]
sequence_length = real.shape[1]
num_stocks = real.shape[2]

print("\nConfiguration:")
print("  Samples:", num_samples)
print("  Sequence length:", sequence_length)
print("  Stocks:", num_stocks)


# ============================================================
# 1. Volatility
# ============================================================

print("\nCalculating volatility...")

# Standard deviation across time for every
# sample × stock.
real_volatility = np.std(
    real,
    axis=1
)

fake_volatility = np.std(
    fake,
    axis=1
)

# Average volatility across all samples and stocks.
real_avg_volatility = np.mean(
    real_volatility
)

fake_avg_volatility = np.mean(
    fake_volatility
)

volatility_error = np.mean(
    np.abs(
        real_volatility
        - fake_volatility
    )
)

print(
    "Real average volatility:",
    real_avg_volatility
)

print(
    "TimeGAN average volatility:",
    fake_avg_volatility
)

print(
    "Volatility absolute error:",
    volatility_error
)


# ============================================================
# 2. ACF
# ============================================================

lags = [1, 5, 10, 20]

acf_results = []

print("\nCalculating autocorrelation...")

for lag in lags:

    print(f"  Lag {lag}...")

    real_acf = calculate_acf(
        real,
        lag
    )

    fake_acf = calculate_acf(
        fake,
        lag
    )

    acf_error = abs(
        real_acf - fake_acf
    )

    acf_results.append(
        {
            "Metric": f"ACF_Lag_{lag}",
            "Real": real_acf,
            "TimeGAN": fake_acf,
            "Absolute_Error": acf_error
        }
    )


# ============================================================
# 3. Build results
# ============================================================

results = [
    {
        "Metric": "Average Volatility",
        "Real": real_avg_volatility,
        "TimeGAN": fake_avg_volatility,
        "Absolute_Error": volatility_error
    }
]

results.extend(acf_results)

results_df = pd.DataFrame(results)


# ============================================================
# Validate results
# ============================================================

numeric_columns = [
    "Real",
    "TimeGAN",
    "Absolute_Error"
]

for column in numeric_columns:

    if not np.isfinite(
        results_df[column].values
    ).all():

        print(
            f"WARNING: {column} "
            "contains non-finite values."
        )


# ============================================================
# Save
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
print("TIMEGAN TEMPORAL EVALUATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nTemporal evaluation completed.")