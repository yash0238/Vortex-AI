"""Decode dense adjacency matrices from node embeddings."""

from __future__ import annotations

import torch
from torch import nn


class GraphDecoder(nn.Module):
	def __init__(self, latent_dim: int, hidden_dim: int = 64, use_bilinear: bool = True) -> None:
		super().__init__()
		self.use_bilinear = use_bilinear
		if use_bilinear:
			self.weight = nn.Parameter(torch.empty(latent_dim, latent_dim))
			nn.init.xavier_uniform_(self.weight)
		else:
			self.fc = nn.Sequential(nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		if z.ndim != 2:
			raise ValueError("z must have shape (nodes, latent_dim)")
		if self.use_bilinear:
			return z @ self.weight @ z.T
		num_nodes = z.shape[0]
		left = z[:, None, :].expand(num_nodes, num_nodes, -1)
		right = z[None, :, :].expand(num_nodes, num_nodes, -1)
		return self.fc(torch.cat([left, right], dim=-1)).squeeze(-1)
