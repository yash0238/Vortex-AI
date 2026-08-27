"""Portfolio performance metrics."""

import numpy as np


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
	values = np.asarray(returns, dtype=float)
	excess = values - risk_free_rate / periods_per_year
	return float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
	values = np.asarray(returns, dtype=float)
	excess = values - risk_free_rate / periods_per_year
	downside = np.minimum(excess, 0)
	return float(np.mean(excess) / (np.std(downside) + 1e-9) * np.sqrt(periods_per_year))


def max_drawdown(cum_returns):
	values = np.asarray(cum_returns, dtype=float)
	if values.size == 0:
		return 0.0
	return float(np.min(values - np.maximum.accumulate(values)))


def calmar_ratio(cum_returns, periods_per_year=252):
	values = np.asarray(cum_returns, dtype=float)
	if values.size == 0 or values[-1] <= 0:
		return float("nan")
	annual_return = values[-1] ** (periods_per_year / values.size) - 1
	return float(annual_return / (abs(max_drawdown(values)) + 1e-9))
