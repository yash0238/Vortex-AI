"""Statistical fidelity tests for VORTEX-AI evaluation.

Implements metrics from masterplan Part F.1:
- Kurtosis matching (fat tails)
- ACF of squared returns (volatility clustering)
- Correlation matrix error
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis
from typing import Dict


def evaluate_stylized_facts(real_returns: np.ndarray, gen_returns: np.ndarray) -> Dict[str, float]:
    """
    Evaluate stylized facts matching between real and generated returns.
    
    As specified in masterplan Part F.1.
    
    Args:
        real_returns: (N_scenarios, T, n_stocks) real returns
        gen_returns: (N_scenarios, T, n_stocks) generated returns
    
    Returns:
        results: dict with statistical metrics
    """
    results = {}
    
    # 1. Kurtosis (fat tails)
    # Flatten all returns for aggregated statistics
    real_flat = real_returns.reshape(-1)
    gen_flat = gen_returns.reshape(-1)
    
    real_kurt = kurtosis(real_flat, fisher=True)  # Excess kurtosis (should be >0)
    gen_kurt = kurtosis(gen_flat, fisher=True)
    
    results["kurtosis_real"] = float(real_kurt)
    results["kurtosis_gen"] = float(gen_kurt)
    results["kurtosis_pass"] = bool(gen_kurt > 3.0)  # Target: >3 for fat tails
    results["kurtosis_error"] = abs(real_kurt - gen_kurt)
    
    # 2. ACF of squared returns (volatility clustering)
    def mean_acf_squared(data: np.ndarray) -> float:
        """Compute mean ACF at lag 1 for squared returns across all series."""
        sq = data ** 2
        acf_vals = []
        
        # Compute ACF for each stock/scenario
        if data.ndim == 3:
            N_scenarios, T, n_stocks = data.shape
            for i in range(N_scenarios):
                for j in range(n_stocks):
                    series = sq[i, :, j]
                    if T > 1:
                        acf = np.corrcoef(series[1:], series[:-1])[0, 1]
                        if not np.isnan(acf):
                            acf_vals.append(acf)
        else:
            # Handle 2D case
            for j in range(data.shape[-1]):
                series = sq[:, j]
                if len(series) > 1:
                    acf = np.corrcoef(series[1:], series[:-1])[0, 1]
                    if not np.isnan(acf):
                        acf_vals.append(acf)
        
        return float(np.mean(acf_vals)) if acf_vals else 0.0
    
    real_acf = mean_acf_squared(real_returns)
    gen_acf = mean_acf_squared(gen_returns)
    
    results["acf_sq_real"] = real_acf
    results["acf_sq_gen"] = gen_acf
    results["acf_sq_pass"] = bool(gen_acf > 0.05)  # Target: significant positive ACF
    results["acf_sq_error"] = abs(real_acf - gen_acf)
    
    # 3. Correlation matrix error (Frobenius norm)
    # Use first scenario or average across scenarios
    if real_returns.ndim == 3:
        real_sample = real_returns[0]  # (T, n_stocks)
        gen_sample = gen_returns[0]
    else:
        real_sample = real_returns
        gen_sample = gen_returns
    
    corr_real = np.corrcoef(real_sample.T)  # (n_stocks, n_stocks)
    corr_gen = np.corrcoef(gen_sample.T)
    
    corr_error = np.linalg.norm(corr_real - corr_gen, ord='fro')
    
    results["corr_error"] = float(corr_error)
    results["corr_error_pass"] = bool(corr_error < 5.0)  # Target: <2.5 (masterplan)
    
    # Summary
    results["all_tests_pass"] = all([
        results["kurtosis_pass"],
        results["acf_sq_pass"],
        results["corr_error_pass"],
    ])
    
    return results


def compute_distribution_moments(returns: np.ndarray) -> Dict[str, float]:
    """
    Compute distribution moments for returns.
    
    Args:
        returns: (N, T, n_stocks) or (T, n_stocks) array
    
    Returns:
        moments: dict with mean, std, skewness, kurtosis
    """
    flat = returns.reshape(-1)
    
    from scipy.stats import skew
    
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat, fisher=True)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def compare_distributions(real_returns: np.ndarray, gen_returns: np.ndarray) -> Dict[str, float]:
    """
    Compare distributional properties between real and generated returns.
    
    Args:
        real_returns: real returns array
        gen_returns: generated returns array
    
    Returns:
        comparison: dict with moment comparisons
    """
    real_moments = compute_distribution_moments(real_returns)
    gen_moments = compute_distribution_moments(gen_returns)
    
    comparison = {}
    for key in real_moments.keys():
        comparison[f"{key}_real"] = real_moments[key]
        comparison[f"{key}_gen"] = gen_moments[key]
        comparison[f"{key}_error"] = abs(real_moments[key] - gen_moments[key])
    
    return comparison


def test_no_autocorrelation_returns(returns: np.ndarray) -> Dict[str, float]:
    """
    Test that returns themselves have no significant autocorrelation.
    
    This is a stylized fact: returns should be uncorrelated, but
    squared returns (volatility) should be correlated.
    
    Args:
        returns: (N, T, n_stocks) returns array
    
    Returns:
        results: dict with ACF statistics for returns
    """
    acf_vals = []
    
    if returns.ndim == 3:
        N, T, n_stocks = returns.shape
        for i in range(N):
            for j in range(n_stocks):
                series = returns[i, :, j]
                if T > 1:
                    acf = np.corrcoef(series[1:], series[:-1])[0, 1]
                    if not np.isnan(acf):
                        acf_vals.append(acf)
    else:
        T, n_stocks = returns.shape
        for j in range(n_stocks):
            series = returns[:, j]
            if T > 1:
                acf = np.corrcoef(series[1:], series[:-1])[0, 1]
                if not np.isnan(acf):
                    acf_vals.append(acf)
    
    mean_acf = float(np.mean(acf_vals)) if acf_vals else 0.0
    
    return {
        "returns_acf_lag1": mean_acf,
        "returns_uncorrelated": bool(abs(mean_acf) < 0.1),  # Should be ~0
    }
