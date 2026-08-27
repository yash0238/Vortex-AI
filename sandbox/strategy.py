"""Simple long/flat regime-driven backtest."""

import numpy as np


def generate_positions(regime_signal, position_size=1.0):
	return np.where(np.asarray(regime_signal) == 0, position_size, 0.0)


def backtest(returns, positions, initial_capital=1.0):
	returns = np.asarray(returns, dtype=float)
	positions = np.asarray(positions, dtype=float)
	if returns.shape[0] != positions.shape[0]:
		raise ValueError("returns and positions must have the same number of periods")
	market_returns = returns.mean(axis=1) if returns.ndim == 2 else returns
	strategy_daily = market_returns * positions
	strategy_cum = np.cumprod(1 + strategy_daily) * initial_capital
	benchmark_cum = np.cumprod(1 + market_returns) * initial_capital
	from sandbox.metrics import max_drawdown, sharpe_ratio
	return {"strategy_cum": strategy_cum, "benchmark_cum": benchmark_cum, "final_capital": float(strategy_cum[-1]), "max_drawdown": max_drawdown(strategy_cum), "sharpe": sharpe_ratio(strategy_daily)}
