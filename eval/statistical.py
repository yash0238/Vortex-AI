"""Statistical tests and summaries for return data."""

from __future__ import annotations

import numpy as np
from scipy import stats


def jarque_bera_test(returns) -> dict:
	statistic, p_value = stats.jarque_bera(np.asarray(returns).ravel())
	return {"statistic": float(statistic), "p_value": float(p_value), "normal": bool(p_value > 0.05)}


def adf_test(series) -> dict:
	from statsmodels.tsa.stattools import adfuller
	result = adfuller(np.asarray(series).ravel(), autolag="AIC")
	return {"statistic": float(result[0]), "p_value": float(result[1]), "used_lag": int(result[2]), "is_stationary": bool(result[1] < 0.05)}


def summary_statistics(returns) -> dict:
	values = np.asarray(returns, dtype=float)
	return {"mean": float(np.mean(values)), "std": float(np.std(values)), "skew": float(stats.skew(values.ravel())), "kurtosis": float(stats.kurtosis(values.ravel())), "min": float(np.min(values)), "max": float(np.max(values))}


def correlation_summary(corr_matrix: np.ndarray) -> dict:
	correlation = np.asarray(corr_matrix, dtype=float)
	if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
		raise ValueError("corr_matrix must be square")
	off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
	return {"avg_corr": float(np.mean(off_diagonal)), "std_corr": float(np.std(off_diagonal)), "pct_positive": float(np.mean(off_diagonal > 0) * 100), "pct_abs_above_0.3": float(np.mean(np.abs(off_diagonal) > 0.3) * 100)}
