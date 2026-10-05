"""Correlation-based connectedness and spillover summaries."""

from __future__ import annotations

import numpy as np


def _off_diagonal(corr_matrix: np.ndarray, threshold: float) -> np.ndarray:
	correlation = np.asarray(corr_matrix, dtype=float).copy()
	if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
		raise ValueError("corr_matrix must be a square matrix")
	np.fill_diagonal(correlation, 0.0)
	correlation[np.abs(correlation) < threshold] = 0.0
	return correlation


def compute_spillover_index(corr_matrix: np.ndarray, threshold: float = 0.0) -> float:
	correlation = _off_diagonal(corr_matrix, threshold)
	num_assets = correlation.shape[0]
	return float(np.abs(correlation).sum() / (num_assets * (num_assets - 1))) if num_assets > 1 else 0.0


def directional_spillovers(corr_matrix: np.ndarray, threshold: float = 0.0) -> dict[str, np.ndarray]:
	correlation = _off_diagonal(corr_matrix, threshold)
	abs_correlation = np.abs(correlation)
	to = abs_correlation.sum(axis=1)
	from_ = abs_correlation.sum(axis=0)
	return {"to": to, "from": from_, "net": to - from_}


def contagion_matrix(corr_matrix: np.ndarray, threshold: float = 0.0) -> np.ndarray:
	return np.abs(_off_diagonal(corr_matrix, threshold))


def _window_corr(window: np.ndarray) -> float:
	"""Mean absolute off-diagonal correlation for one (T, S) window."""
	w = np.asarray(window, dtype=float)
	if w.ndim != 2 or w.shape[1] < 2:
		return 0.0
	corr = np.corrcoef(w, rowvar=False)
	corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
	off = corr[~np.eye(corr.shape[0], dtype=bool)]
	return float(np.mean(np.abs(off)))


def evaluate_contagion(gen_returns: np.ndarray, regimes: np.ndarray) -> dict:
	"""Crisis correlation boost: (crisis-normal)/|normal|, target >50%."""
	gen = np.asarray(gen_returns, dtype=float)
	reg = np.asarray(regimes).ravel()
	normal = [ _window_corr(gen[i]) for i in range(len(gen)) if reg[i] == 0 ]
	crisis = [ _window_corr(gen[i]) for i in range(len(gen)) if reg[i] == 1 ]
	corr_normal = float(np.mean(normal)) if normal else 0.0
	corr_crisis = float(np.mean(crisis)) if crisis else 0.0
	boost = float((corr_crisis - corr_normal) / abs(corr_normal)) if corr_normal != 0.0 else 0.0
	return {
		"corr_normal": corr_normal, "corr_crisis": corr_crisis,
		"crisis_corr_boost": boost,
		"contagion_pass": bool(boost > 0.5 and len(normal) > 0 and len(crisis) > 0),
	}


def cvar_regime_ratio(gen_returns: np.ndarray, regimes: np.ndarray, alpha: float = 0.05) -> dict:
	"""CVaR (expected shortfall) of per-window mean returns, crisis/normal ratio, target >3x."""
	gen = np.asarray(gen_returns, dtype=float)
	reg = np.asarray(regimes).ravel()
	port = gen.mean(axis=(1, 2)) if gen.ndim == 3 else gen.mean(axis=1)

	def _cvar(x: np.ndarray) -> float:
		x = np.asarray(x, dtype=float)
		x = x[np.isfinite(x)]
		if x.size == 0:
			return 0.0
		k = max(1, int(np.ceil(alpha * x.size)))
		tail = np.sort(x)[:k]
		return float(-tail.mean())

	cvar_normal = _cvar(port[reg == 0])
	cvar_crisis = _cvar(port[reg == 1])
	ratio = float(cvar_crisis / cvar_normal) if cvar_normal > 0 else 0.0
	return {
		"cvar_normal": cvar_normal, "cvar_crisis": cvar_crisis,
		"cvar_ratio": ratio, "cvar_pass": bool(ratio > 3.0),
	}


def granger_causality_test(window: np.ndarray, maxlag: int = 1, p_threshold: float = 0.05, max_pairs: int = 50, seed: int = 42) -> dict:
	"""Pairwise Granger test on one (T, S) window; subsamples pairs if S large."""
	from statsmodels.tsa.stattools import grangercausalitytests
	w = np.asarray(window, dtype=float)
	t, s = w.shape
	pairs = [(i, j) for i in range(s) for j in range(s) if i != j]
	rng = np.random.default_rng(seed)
	if len(pairs) > max_pairs:
		pairs = [ pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False) ]
	sig = 0
	tested = 0
	for i, j in pairs:
		try:
			res = grangercausalitytests(w[:, [j, i]], maxlag=maxlag, verbose=False)
			p = float(res[maxlag][0]["ssr_ftest"][1])
			tested += 1
			if np.isfinite(p) and p < p_threshold:
				sig += 1
		except Exception:
			continue
	return {
		"granger_causality_ratio": float(sig / tested) if tested else 0.0,
		"granger_pairs_significant": int(sig),
		"granger_total_tests": int(tested),
	}
