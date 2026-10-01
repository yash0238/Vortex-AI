"""Circuit-breaker detection utilities.

Implements NSE-specific circuit filter logic (masterplan Part D.5):
- NSE Band A: +-5% (illiquid/small-cap stocks)
- NSE Band B: +-10% (most NIFTY-50 large-caps, default)
- NSE Band C: +-20% (rare)
Plus a soft training-time penalty for violations.
"""

import numpy as np
import pandas as pd
import torch

BAND_A_STOCKS = ["SBIN.NS", "TATAMOTORS.NS", "TATASTEEL.NS"]
BAND_A_LIMIT = 0.05
BAND_B_LIMIT = 0.10
BAND_C_LIMIT = 0.20


def _normalize_ticker(ticker: str) -> str:
    """Normalize to .NS-suffixed form so Band-A checks work for both
    'SBIN' and 'SBIN.NS' caller conventions (2.7)."""
    tk = str(ticker).upper()
    return tk if tk.endswith(".NS") else tk + ".NS"


def _is_band_a(ticker: str) -> bool:
    return _normalize_ticker(ticker) in BAND_A_STOCKS


def detect_circuit_breakers(returns_df: pd.DataFrame, limit: float = 0.10) -> pd.Series:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return returns_df.abs().gt(limit).any(axis=1)


def circuit_breaker_stats(returns_df: pd.DataFrame, limit: float = 0.10) -> dict:
    mask = detect_circuit_breakers(returns_df, limit)
    return {"num_days": int(mask.sum()), "pct_days": float(mask.mean() * 100), "dates": returns_df.index[mask]}


def apply_circuit_filter(returns_array: np.ndarray, tickers: list[str],
                         default_band: float = 0.10) -> np.ndarray:
    """Hard-clip returns to NSE circuit filter bands (post-processing).

    Args:
        returns_array: (T, N) numpy array of stock returns.
        tickers: List of stock tickers to determine band assignment.
        default_band: Default band limit (10% for most NIFTY-50).

    Returns:
        Clipped returns array with same shape as input.
    """
    clipped = returns_array.copy()
    for j, tk in enumerate(tickers):
        limit = BAND_A_LIMIT if _is_band_a(tk) else default_band
        clipped[:, j] = np.clip(returns_array[:, j], -limit, limit)
    return clipped


def circuit_filter_loss(r_hat: torch.Tensor, tickers=None,
                        band: float = 0.10) -> torch.Tensor:
    """Soft penalty during training for returns exceeding circuit filter bands.

    Penalizes squared magnitude of any return exceeding the band limit,
    encouraging the model to learn within NSE constraints.

    Args:
        r_hat: (..., N) — predicted returns (tensor).
        tickers: Optional list to assign per-stock bands (Band A = 5%).
        band: Default circuit filter limit (default 0.10 = 10%).

    Returns:
        Scalar penalty tensor.
    """
    if tickers is not None and r_hat.shape[-1] == len(tickers):
        band_tensor = torch.full_like(r_hat, band)
        for j, tk in enumerate(tickers):
            if _is_band_a(tk):
                band_tensor[..., j] = BAND_A_LIMIT
        violation = torch.clamp(r_hat.abs() - band_tensor, min=0.0)
    else:
        violation = torch.clamp(r_hat.abs() - band, min=0.0)
    return violation.pow(2).mean()


__all__ = [
    "BAND_A_STOCKS",
    "BAND_A_LIMIT",
    "BAND_B_LIMIT",
    "BAND_C_LIMIT",
    "detect_circuit_breakers",
    "circuit_breaker_stats",
    "apply_circuit_filter",
    "circuit_filter_loss",
]
