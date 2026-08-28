"""Synthetic two-state regime-switching return generator."""

import numpy as np


def generate_synthetic_returns(n_days=1000, n_assets=50, mu_normal=0.0002, mu_crisis=-0.0005, vol_normal=0.01, vol_crisis=0.03, regime_transition_prob=0.05, seed=42):
	if n_days < 1 or n_assets < 1 or not 0 <= regime_transition_prob <= 1:
		raise ValueError("n_days and n_assets must be positive; transition probability must be in [0, 1]")
	rng = np.random.default_rng(seed)
	regimes = np.zeros(n_days, dtype=np.int64)
	for day in range(1, n_days):
		probability = regime_transition_prob
		if rng.random() < probability:
			regimes[day] = 1 - regimes[day - 1]
		else:
			regimes[day] = regimes[day - 1]
	means = np.where(regimes[:, None] == 0, mu_normal, mu_crisis)
	volatility = np.where(regimes[:, None] == 0, vol_normal, vol_crisis)
	return rng.normal(means, volatility, size=(n_days, n_assets)), regimes
