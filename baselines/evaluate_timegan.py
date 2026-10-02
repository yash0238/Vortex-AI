"""Evaluate the completed TimeGAN baseline on NIFTY-50 windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_REAL = ROOT / "data" / "timegan_windows.npy"
DEFAULT_SYNTHETIC = ROOT / "results" / "timegan_full" / "generated_samples.npy"
DEFAULT_OUTPUT = ROOT / "results" / "timegan_full"


def load_arrays(real_path: Path, synthetic_path: Path) -> tuple[np.ndarray, np.ndarray]:

    real = np.asarray(np.load(real_path), dtype=np.float64)
    synthetic = np.asarray(np.load(synthetic_path), dtype=np.float64)
    if real.shape != synthetic.shape or real.ndim != 3:
        raise ValueError(
            f"Expected matching 3D arrays, got real={real.shape}, synthetic={synthetic.shape}"
        )
    if not np.isfinite(real).all() or not np.isfinite(synthetic).all():
        raise ValueError("Real or synthetic data contains NaN/Inf values")
    return real, synthetic


def lag_one_acf(data: np.ndarray, squared: bool = False) -> float:
    values = data**2 if squared else data
    left = values[:, :-1, :].reshape(-1)
    right = values[:, 1:, :].reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def correlation_error(real: np.ndarray, synthetic: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        real_corr = np.corrcoef(real.reshape(-1, real.shape[-1]), rowvar=False)
        synthetic_corr = np.corrcoef(synthetic.reshape(-1, synthetic.shape[-1]), rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    synthetic_corr = np.nan_to_num(synthetic_corr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(real_corr - synthetic_corr, ord="fro"))


def discriminative_score(real: np.ndarray, synthetic: np.ndarray, seed: int) -> float:
    features = np.concatenate([real, synthetic], axis=0).reshape(real.shape[0] * 2, -1)
    labels = np.concatenate([
        np.zeros(real.shape[0], dtype=int),
        np.ones(synthetic.shape[0], dtype=int),
    ])
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    classifier = LogisticRegression(max_iter=200, solver="liblinear", random_state=seed)
    classifier.fit(train_x, train_y)
    accuracy = accuracy_score(test_y, classifier.predict(test_x))
    return float(abs(accuracy - 0.5))


def predictive_score(real: np.ndarray, synthetic: np.ndarray) -> float:
    train_x = synthetic[:, :-1, :].reshape(synthetic.shape[0], -1)
    train_y = synthetic[:, -1, :]
    test_x = real[:, :-1, :].reshape(real.shape[0], -1)
    test_y = real[:, -1, :]
    predictor = Ridge(alpha=1.0)
    predictor.fit(train_x, train_y)
    return float(mean_squared_error(test_y, predictor.predict(test_x)))


def evaluate(real: np.ndarray, synthetic: np.ndarray, seed: int) -> dict[str, float | int | list[int]]:
    return {
        "sample_count": int(synthetic.shape[0]),
        "sequence_length": int(synthetic.shape[1]),
        "feature_count": int(synthetic.shape[2]),
        "real_mean": float(real.mean()),
        "synthetic_mean": float(synthetic.mean()),
        "mean_absolute_error": float(abs(real.mean() - synthetic.mean())),
        "real_std": float(real.std()),
        "synthetic_std": float(synthetic.std()),
        "std_absolute_error": float(abs(real.std() - synthetic.std())),
        "acf_lag1_real": lag_one_acf(real),
        "acf_lag1_synthetic": lag_one_acf(synthetic),
        "acf_squared_lag1_real": lag_one_acf(real, squared=True),
        "acf_squared_lag1_synthetic": lag_one_acf(synthetic, squared=True),
        "kurtosis_real": float(kurtosis(real.reshape(-1), fisher=True, bias=False)),
        "kurtosis_synthetic": float(kurtosis(synthetic.reshape(-1), fisher=True, bias=False)),
        "correlation_frobenius_error": correlation_error(real, synthetic),
        "discriminative_score_logistic": discriminative_score(real, synthetic, seed),
        "predictive_score_ridge_mse": predictive_score(real, synthetic),
    }


def write_comparison(path: Path, metrics: dict[str, float | int | list[int]]) -> None:
    rows = [
        {
            "metric": "Discriminative score",
            "paper_reference": "Reported by TimeGAN per dataset; no universal value",
            "paper_direction": "Lower is better; 0 is ideal",
            "nifty50_result": metrics["discriminative_score_logistic"],
            "comparability": "Surrogate: logistic classifier, not the paper's post-hoc RNN",
        },
        {
            "metric": "Predictive score",
            "paper_reference": "Reported by TimeGAN per dataset; no universal value",
            "paper_direction": "Lower is better",
            "nifty50_result": metrics["predictive_score_ridge_mse"],
            "comparability": "Surrogate: Ridge next-step predictor, not the paper's post-hoc RNN",
        },
        {
            "metric": "Mean",
            "paper_reference": "Temporal dynamics and marginal statistics preserved",
            "paper_direction": "Closer to real data is better",
            "nifty50_result": metrics["mean_absolute_error"],
            "comparability": "Direct NIFTY-50 comparison",
        },
        {
            "metric": "Standard deviation",
            "paper_reference": "Temporal dynamics and marginal statistics preserved",
            "paper_direction": "Closer to real data is better",
            "nifty50_result": metrics["std_absolute_error"],
            "comparability": "Direct NIFTY-50 comparison",
        },
        {
            "metric": "ACF lag 1",
            "paper_reference": "Temporal correlation preservation evaluated",
            "paper_direction": "Synthetic should be close to real",
            "nifty50_result": metrics["acf_lag1_synthetic"],
            "comparability": "Direct value; compare with acf_lag1_real in metrics JSON",
        },
        {
            "metric": "Squared-return ACF lag 1",
            "paper_reference": "Volatility dynamics are part of temporal fidelity",
            "paper_direction": "Synthetic should be close to real",
            "nifty50_result": metrics["acf_squared_lag1_synthetic"],
            "comparability": "Additional finance-specific metric",
        },
        {
            "metric": "Correlation Frobenius error",
            "paper_reference": "Cross-variable dynamics evaluated",
            "paper_direction": "Lower is better",
            "nifty50_result": metrics["correlation_frobenius_error"],
            "comparability": "Additional multivariate finance metric",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    real, synthetic = load_arrays(args.real, args.synthetic)
    metrics = evaluate(real, synthetic, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_comparison(args.output_dir / "paper_vs_nifty50_comparison.csv", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()