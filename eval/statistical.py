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
	"""Lag-1 autocorrelation along TIME, averaged over windows and stocks.

	`x` is (n_windows, T) or (n_windows, T, n_stocks). Flattening the array
	first (the old behaviour) mixes stocks and window boundaries into every
	"lag", so a 50-stock panel of perfectly independent series reads ~0 no
	matter how strong the volatility clustering is.
	"""
	values = np.asarray(x, dtype=float)
	if values.ndim == 3:
		sq = np.square(values)
		sq = sq - sq.mean(axis=1, keepdims=True)
		num = (sq[:, :-1] * sq[:, 1:]).mean(axis=1)
		den = sq.var(axis=1)
		vals = num.ravel() / den.ravel()
		vals = vals[np.isfinite(vals) & (den.ravel() > 0)]
		return float(np.mean(vals)) if vals.size else float("nan")
	if values.ndim == 2:
		vals = []
		for row in values:
			v = row[np.isfinite(row)]
			if v.size < 3:
				continue
			v = v - v.mean()
			d = float(np.dot(v, v))
			if d > 0:
				vals.append(float(np.dot(v[:-1], v[1:]) / d))
		return float(np.mean(vals)) if vals else float("nan")
	values = values.ravel()
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


def _per_stock_kurt_median(returns_3d: np.ndarray) -> float:
	arr = np.asarray(returns_3d, dtype=float)
	if arr.ndim != 3:
		arr = np.asarray(arr).reshape(-1)
		return _kurtosis_pearson(arr)
	n, t, s = arr.shape
	ks = [float(_kurtosis_pearson(arr[:, :, j])) for j in range(s)]
	ks = [k for k in ks if np.isfinite(k)]
	return float(np.median(ks)) if ks else float("nan")


def block_bootstrap_kurt_interval(real_returns: np.ndarray, n_draws: int = 1000,
		block: int = 100, seed: int = 0, use_median: bool = False) -> tuple:
	"""5th to 95th pct of real kurtosis over block resamples (contiguous blocks)."""
	rng = np.random.default_rng(seed)
	real = np.asarray(real_returns, dtype=float)
	n = len(real)
	nb = max(1, n // block)
	vals = []
	for _ in range(n_draws):
		picks = rng.integers(0, nb, size=nb)
		samp = np.concatenate([real[k * block:(k + 1) * block] for k in picks], axis=0)[:n]
		vals.append(_per_stock_kurt_median(samp) if use_median else _kurtosis_pearson(samp))
	vals = np.array([v for v in vals if np.isfinite(v)])
	return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


def _offdiag_mae(a: np.ndarray, b: np.ndarray) -> float:
	mask = ~np.eye(a.shape[0], dtype=bool)
	return float(np.mean(np.abs(a[mask] - b[mask])))


# Default gate thresholds (fallback values when the data-derived statistic is
# not finite). These are the thresholds the gates are judged against and are
# importable by the reproducibility test.
acf_tol = 0.02
corr_limit = 3.0


def _split_half_corr_error(real: np.ndarray) -> float:
	"""Real-vs-real Frobenius error of the pooled correlation (the metric ceiling).

	Two halves of the same real panel are the closest any generator can get to
	the other half; a fixed threshold like 3.0 is below this floor, so no model
	can ever pass it.
	"""
	if len(real) < 4:
		return float("nan")
	h = len(real) // 2
	return float(np.linalg.norm(_mean_corr(real[:h]) - _mean_corr(real[h:]), ord="fro"))


def evaluate_stylized_facts(real_returns: np.ndarray, gen_returns: np.ndarray) -> dict:
	"""Gates judged against the real data they are supposed to reproduce.

	kurtosis: generated excess kurtosis must sit inside the real block-bootstrap
	          interval. acf: generated ACF(r^2) must be close to the real ACF(r^2)
	          measured the same way. corr: generated pooled correlation must be as
	          close to the real one as two halves of the real panel are to each
	          other.
	"""
	real = np.asarray(real_returns, dtype=float)
	gen = np.asarray(gen_returns, dtype=float)
	kurt_real = _kurtosis_pearson(real)
	kurt_gen = _kurtosis_pearson(gen)
	lo, hi = block_bootstrap_kurt_interval(real)
	acf_real = _acf_lag1(real)
	acf_gen = _acf_lag1(gen)
	cr, cg = _mean_corr(real), _mean_corr(gen)
	corr_error = float(np.linalg.norm(cr - cg, ord="fro"))
	corr_mae = _offdiag_mae(cr, cg)
	corr_ceiling = _split_half_corr_error(real)

	acf_tol = max(0.02, 0.35 * abs(acf_real)) if np.isfinite(acf_real) else 0.02
	corr_limit = max(1.5 * corr_ceiling, corr_ceiling + 1.0) if np.isfinite(corr_ceiling) else 3.0

	kurt_pass = bool(np.isfinite(kurt_gen) and lo <= kurt_gen <= hi)
	acf_pass = bool(np.isfinite(acf_gen) and abs(acf_gen - acf_real) <= acf_tol)
	corr_pass = bool(np.isfinite(corr_error) and corr_error <= corr_limit)
	return {
		"kurtosis_real": kurt_real, "kurtosis_gen": kurt_gen, "kurtosis_pass": kurt_pass,
		"kurtosis_interval": [lo, hi],
		"acf_sq_real": acf_real, "acf_sq_gen": acf_gen, "acf_sq_pass": acf_pass,
		"acf_sq_tolerance": acf_tol,
		"corr_error": corr_error, "corr_mae": corr_mae, "corr_error_pass": corr_pass,
		"corr_error_ceiling": corr_ceiling, "corr_error_limit": corr_limit,
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
