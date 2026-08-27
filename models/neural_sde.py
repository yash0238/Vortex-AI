"""Neural drift and diffusion model with Euler-Maruyama integration."""

from __future__ import annotations

import torch
from torch import nn


class NeuralSDE(nn.Module):
	def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
		super().__init__()
		self.drift = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, state_dim))
		self.diffusion = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, state_dim), nn.Softplus())

	def forward(self, t, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		return self.drift(x), self.diffusion(x)

	def integrate(self, x0: torch.Tensor, dt: float = 0.01, n_steps: int = 100, brownian: torch.Tensor | None = None, return_path: bool = False) -> torch.Tensor:
		if dt <= 0 or n_steps < 1:
			raise ValueError("dt must be positive and n_steps must be at least one")
		x = x0
		if brownian is None:
			brownian = torch.randn(n_steps, *x0.shape, device=x0.device, dtype=x0.dtype) * dt**0.5
		if brownian.shape != (n_steps, *x0.shape):
			raise ValueError("brownian must have shape (n_steps, *x0.shape)")
		path = [x]
		for increment in brownian:
			drift, diffusion = self.forward(None, x)
			x = x + drift * dt + diffusion * increment
			path.append(x)
		return torch.stack(path) if return_path else x
