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
