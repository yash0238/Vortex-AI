"""Circuit-breaker detection utilities."""

import pandas as pd


def detect_circuit_breakers(returns_df: pd.DataFrame, limit: float = 0.095) -> pd.Series:
	if limit < 0:
		raise ValueError("limit must be non-negative")
	return returns_df.abs().gt(limit).any(axis=1)


def circuit_breaker_stats(returns_df: pd.DataFrame, limit: float = 0.095) -> dict:
	mask = detect_circuit_breakers(returns_df, limit)
	return {"num_days": int(mask.sum()), "pct_days": float(mask.mean() * 100), "dates": returns_df.index[mask]}
