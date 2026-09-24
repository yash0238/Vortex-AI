"""Contagion propagation test for VORTEX-AI evaluation.

Implements contagion test from masterplan Part F.3:
Measures if crisis periods show increased cross-correlation.
Target: crisis correlation boost > 50% vs normal periods.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def crisis_correlation_boost(
    gen_returns: np.ndarray,
    regimes: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Measure correlation boost during crisis vs normal regimes.
    
    As specified in masterplan Part F.3.
    
    Args:
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) regime labels (0=normal, 1=crisis)
    
    Returns:
        boost: percentage increase in crisis correlation
        avg_normal: average correlation in normal periods
        avg_crisis: average correlation in crisis periods
    """
    # Separate normal and crisis samples
    normal_idx = np.where(regimes == 0)[0]
    crisis_idx = np.where(regimes == 1)[0]
    
    if len(normal_idx) == 0 or len(crisis_idx) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute average correlation matrices
    def avg_correlation(returns: np.ndarray) -> float:
        """Compute average off-diagonal correlation."""
        if len(returns) == 0:
            return 0.0
        
        # Average returns across scenarios
        avg_returns = returns.mean(axis=0)  # (T, n_stocks)
        
        # Correlation matrix
        corr = np.corrcoef(avg_returns.T)  # (n_stocks, n_stocks)
        
        # Average absolute off-diagonal correlation
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.abs(corr[mask]).mean()
        
        return float(avg_corr)
    
    avg_normal = avg_correlation(gen_returns[normal_idx])
    avg_crisis = avg_correlation(gen_returns[crisis_idx])
    
    # Compute boost
    if avg_normal > 0:
        boost = (avg_crisis - avg_normal) / avg_normal
    else:
        boost = 0.0
    
    return float(boost), float(avg_normal), float(avg_crisis)


def evaluate_contagion(
    gen_returns: np.ndarray,
    regimes: np.ndarray,
) -> Dict[str, float]:
    """
    Complete contagion evaluation.
    
    Args:
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) regime labels
    
    Returns:
        results: dict with contagion metrics
    """
    boost, avg_n, avg_c = crisis_correlation_boost(gen_returns, regimes)
    
    results = {
        "crisis_corr_boost": boost,
        "corr_normal": avg_n,
        "corr_crisis": avg_c,
        "contagion_pass": bool(boost > 0.50),  # Target: >50% boost
    }
    
    return results


def sector_contagion_analysis(
    gen_returns: np.ndarray,
    regimes: np.ndarray,
    sector_map: Dict[int, str] | None = None,
) -> Dict[str, float]:
    """
    Analyze sector-level contagion patterns.
    
    Measures if certain sectors lead during crises (e.g., Banking -> IT).
    
    Args:
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) regime labels
        sector_map: optional mapping of stock_idx -> sector name
    
    Returns:
        results: dict with sector contagion patterns
    """
    crisis_idx = np.where(regimes == 1)[0]
    
    if len(crisis_idx) == 0:
        return {"sector_contagion": 0.0}
    
    # For NIFTY-50, we can define rough sectors
    # This is a simplified placeholder
    if sector_map is None:
        # Assume first 10 stocks are Banking, next 10 are IT, etc.
        n_stocks = gen_returns.shape[2]
        sector_map = {i: "Banking" if i < 10 else "IT" if i < 20 else "Other" 
                     for i in range(n_stocks)}
    
    crisis_returns = gen_returns[crisis_idx]
    
    # Compute cross-sector correlations
    avg_returns = crisis_returns.mean(axis=0)  # (T, n_stocks)
    corr_matrix = np.corrcoef(avg_returns.T)
    
    # Banking-IT cross-correlation
    banking_idx = [i for i, s in sector_map.items() if s == "Banking"]
    it_idx = [i for i, s in sector_map.items() if s == "IT"]
    
    if banking_idx and it_idx:
        cross_sector_corr = corr_matrix[np.ix_(banking_idx, it_idx)].mean()
    else:
        cross_sector_corr = 0.0
    
    return {
        "banking_it_corr_crisis": float(cross_sector_corr),
        "cross_sector_contagion": bool(cross_sector_corr > 0.3),
    }


def granger_causality_test(
    returns: np.ndarray,
    max_lag: int = 5,
) -> Dict[str, float]:
    """
    Test for Granger causality between stocks.
    
    Simplified version: measures if past returns of one stock
    predict future returns of another.
    
    Args:
        returns: (T, n_stocks) returns time series
        max_lag: maximum lag to test
    
    Returns:
        results: dict with causality metrics
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    T, n_stocks = returns.shape
    
    # Test a few stock pairs for Granger causality
    causality_count = 0
    total_tests = 0
    
    # Test first 5 stocks against each other (simplified)
    test_stocks = min(5, n_stocks)
    
    for i in range(test_stocks):
        for j in range(test_stocks):
            if i != j:
                try:
                    # Test if stock i Granger-causes stock j
                    data = np.column_stack([returns[:, j], returns[:, i]])
                    result = grangercausalitytests(data, max_lag, verbose=False)
                    
                    # Check p-value at lag 1
                    p_value = result[1][0]['ssr_ftest'][1]
                    if p_value < 0.05:
                        causality_count += 1
                    
                    total_tests += 1
                except:
                    pass
    
    causality_ratio = causality_count / total_tests if total_tests > 0 else 0.0
    
    return {
        "granger_causality_ratio": float(causality_ratio),
        "granger_pairs_significant": causality_count,
        "granger_total_tests": total_tests,
    }


def cvar_regime_ratio(
    gen_returns: np.ndarray,
    regimes: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute CVaR ratio between crisis and normal regimes.
    
    As specified in masterplan Part F.4.
    Target: CVaR ratio > 3x
    
    Args:
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) regime labels
        alpha: quantile for CVaR (default 5%)
    
    Returns:
        results: dict with CVaR metrics
    """
    normal_idx = np.where(regimes == 0)[0]
    crisis_idx = np.where(regimes == 1)[0]
    
    if len(normal_idx) == 0 or len(crisis_idx) == 0:
        return {"cvar_ratio": 0.0}
    
    def compute_cvar(returns: np.ndarray, alpha: float) -> float:
        """Compute Conditional Value at Risk."""
        flat_returns = returns.reshape(-1)
        var = np.quantile(flat_returns, alpha)
        cvar = flat_returns[flat_returns <= var].mean()
        return abs(float(cvar))  # Absolute value for interpretability
    
    cvar_normal = compute_cvar(gen_returns[normal_idx], alpha)
    cvar_crisis = compute_cvar(gen_returns[crisis_idx], alpha)
    
    ratio = cvar_crisis / cvar_normal if cvar_normal > 0 else 0.0
    
    return {
        "cvar_normal": cvar_normal,
        "cvar_crisis": cvar_crisis,
        "cvar_ratio": float(ratio),
        "cvar_pass": bool(ratio > 3.0),  # Target: >3x
    }
