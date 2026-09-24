"""NSE Circuit Filter Enforcement for VORTEX-AI CG-NSDE framework."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# NSE Circuit Breaker Bands as per official documentation
BAND_A_STOCKS = ["SBIN.NS", "TATAMOTORS.NS", "TATASTEEL.NS"]  # ±5%
BAND_B_STOCKS = []  # ±10% (most NIFTY-50 stocks)
BAND_C_STOCKS = []  # ±20% (less liquid stocks)

DEFAULT_BAND = 0.10  # 10% for most NIFTY-50 stocks


def apply_circuit_filter(returns_array: np.ndarray, tickers: list[str], 
                         default_band: float = DEFAULT_BAND) -> np.ndarray:
    """
    Hard clipping of returns to NSE circuit breaker bands.
    
    As specified in masterplan Part D.5.
    
    Args:
        returns_array: (T, N) numpy array of returns
        tickers: list of N stock tickers
        default_band: default circuit limit (10% for Band B)
    
    Returns:
        clipped: (T, N) array with returns clipped to circuit bands
    """
    clipped = returns_array.copy()
    
    for j, tk in enumerate(tickers):
        # Determine band limit for this stock
        if tk in BAND_A_STOCKS:
            limit = 0.05  # Band A: ±5%
        elif tk in BAND_C_STOCKS:
            limit = 0.20  # Band C: ±20%
        else:
            limit = default_band  # Band B: ±10%
        
        # Clip returns to ±limit
        clipped[:, j] = np.clip(returns_array[:, j], -limit, limit)
    
    return clipped


def circuit_filter_loss(r_hat: torch.Tensor, band: float = DEFAULT_BAND) -> torch.Tensor:
    """
    Soft penalty during training for returns outside NSE circuit bands.
    
    As specified in masterplan Part D.5.
    
    This allows the model to learn to respect circuit limits without hard constraints
    during training, which could cause gradient issues.
    
    Args:
        r_hat: (B, T, N) predicted returns
        band: circuit limit threshold (default 10%)
    
    Returns:
        loss: scalar penalty for circuit violations
    """
    # Compute violation: how much returns exceed the band
    violation = torch.clamp(r_hat.abs() - band, min=0.0)
    
    # Square the violation for smooth gradients
    return violation.pow(2).mean()


def get_stock_circuit_band(ticker: str) -> float:
    """
    Get the appropriate circuit band limit for a given stock ticker.
    
    Args:
        ticker: NSE stock ticker (e.g., "SBIN.NS")
    
    Returns:
        limit: circuit breaker limit (0.05, 0.10, or 0.20)
    """
    if ticker in BAND_A_STOCKS:
        return 0.05
    elif ticker in BAND_C_STOCKS:
        return 0.20
    else:
        return 0.10


def apply_circuit_filter_torch(returns: torch.Tensor, tickers: list[str]) -> torch.Tensor:
    """
    PyTorch version of circuit filter for use in model forward pass.
    
    Args:
        returns: (B, T, N) or (T, N) tensor of returns
        tickers: list of N stock tickers
    
    Returns:
        clipped: same shape as input with circuit limits applied
    """
    clipped = returns.clone()
    
    # Handle both (B, T, N) and (T, N) shapes
    if returns.ndim == 3:
        for j, tk in enumerate(tickers):
            limit = get_stock_circuit_band(tk)
            clipped[:, :, j] = torch.clamp(returns[:, :, j], -limit, limit)
    else:
        for j, tk in enumerate(tickers):
            limit = get_stock_circuit_band(tk)
            clipped[:, j] = torch.clamp(returns[:, j], -limit, limit)
    
    return clipped


def check_circuit_violations(returns: np.ndarray, tickers: list[str]) -> dict:
    """
    Analyze circuit breaker violations in generated data.
    
    Args:
        returns: (T, N) array of returns
        tickers: list of N stock tickers
    
    Returns:
        stats: dictionary with violation statistics
    """
    violations = []
    total_observations = returns.size
    
    for j, tk in enumerate(tickers):
        limit = get_stock_circuit_band(tk)
        stock_violations = np.sum(np.abs(returns[:, j]) > limit)
        violations.append(stock_violations)
    
    total_violations = sum(violations)
    violation_rate = total_violations / total_observations
    
    return {
        "total_violations": total_violations,
        "total_observations": total_observations,
        "violation_rate": violation_rate,
        "violations_per_stock": violations,
        "max_violation": np.max(np.abs(returns)),
    }
