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


def _kurtosis_pearson(x: np.ndarray) -> float:
	values = np.asarray(x, dtype=float).ravel()
	values = values[np.isfinite(values)]
	if values.size < 4:
		return float("nan")
	return float(stats.kurtosis(values, fisher=False))


def _acf_lag1(x: np.ndarray) -> float:
	values = np.asarray(x, dtype=float).ravel()
	values = values[np.isfinite(values)]
	if values.size < 3:
		return float("nan")
	v = values - values.mean()
	denom = float(np.dot(v, v))
	if denom == 0.0:
		return 0.0
	return float(np.dot(v[:-1], v[1:]) / denom)


def _mean_corr(returns_3d: np.ndarray) -> np.ndarray:
	arr = np.asarray(returns_3d, dtype=float)
	if arr.ndim != 3:
		raise ValueError("expected (N, T, S) returns")
	n, t, s = arr.shape
	flat = arr.reshape(n * t, s)
	flat = flat[:, np.isfinite(flat).all(axis=0)]
	if flat.shape[1] < 2:
		return np.eye(s)
	corr = np.corrcoef(flat, rowvar=False)
	return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_stylized_facts(real_returns: np.ndarray, gen_returns: np.ndarray) -> dict:
	"""Masterplan Part F: kurtosis>3.0, ACF(r^2)>0.05, corr error<2.5 Frobenius."""
	real = np.asarray(real_returns, dtype=float)
	gen = np.asarray(gen_returns, dtype=float)
	kurt_real = _kurtosis_pearson(real)
	kurt_gen = _kurtosis_pearson(gen)
	acf_real = _acf_lag1(np.square(real))
	acf_gen = _acf_lag1(np.square(gen))
	corr_error = float(np.linalg.norm(_mean_corr(real) - _mean_corr(gen), ord="fro"))
	kurt_pass = bool(np.isfinite(kurt_gen) and kurt_gen > 3.0
		and abs(kurt_gen - kurt_real) / (abs(kurt_real) + 1e-8) < 0.30)
	acf_pass = bool(np.isfinite(acf_gen) and acf_gen > 0.05)
	corr_pass = bool(np.isfinite(corr_error) and corr_error < 2.5)
	return {
		"kurtosis_real": kurt_real, "kurtosis_gen": kurt_gen, "kurtosis_pass": kurt_pass,
		"acf_sq_real": acf_real, "acf_sq_gen": acf_gen, "acf_sq_pass": acf_pass,
		"corr_error": corr_error, "corr_error_pass": corr_pass,
		"all_tests_pass": bool(kurt_pass and acf_pass and corr_pass),
	}


def compare_distributions(real_returns: np.ndarray, gen_returns: np.ndarray) -> dict:
	real = np.asarray(real_returns, dtype=float).ravel()
	gen = np.asarray(gen_returns, dtype=float).ravel()
	real = real[np.isfinite(real)]
	gen = gen[np.isfinite(gen)]
	return {
		"mean_real": float(real.mean()), "mean_gen": float(gen.mean()),
		"std_real": float(real.std()), "std_gen": float(gen.std()),
		"mean_abs_diff": float(abs(real.mean() - gen.mean())),
		"std_abs_diff": float(abs(real.std() - gen.std())),
		"skew_real": float(stats.skew(real)), "skew_gen": float(stats.skew(gen)),
		"kurtosis_real": _kurtosis_pearson(real), "kurtosis_gen": _kurtosis_pearson(gen),
	}


def test_no_autocorrelation_returns(gen_returns: np.ndarray) -> dict:
	acf = _acf_lag1(np.asarray(gen_returns, dtype=float))
	return {
		"returns_acf_lag1": float(acf),
		"returns_uncorrelated": bool(np.isfinite(acf) and abs(acf) < 0.1),
	}
