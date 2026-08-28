"""Spatio-temporal encoder and prediction heads for Vortex-AI."""

from __future__ import annotations

import torch
from torch import nn


class SpatialTemporalGNN(nn.Module):
	"""Encode per-asset return histories and predict graph and regime targets."""

	def __init__(
		self,
		num_assets: int,
		hidden_dim: int = 64,
		lstm_layers: int = 2,
		dropout: float = 0.3,
	) -> None:
		super().__init__()
		self.num_assets = num_assets
		self.hidden_dim = hidden_dim
		self.temporal_encoder = nn.LSTM(
			input_size=1,
			hidden_size=hidden_dim,
			num_layers=lstm_layers,
			batch_first=True,
			dropout=dropout if lstm_layers > 1 else 0.0,
		)
		self.edge_score = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
		self.regime_head = nn.Sequential(
			nn.Linear(hidden_dim * 2, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if x.ndim != 3:
			raise ValueError(f"Expected input shape (batch, time, assets), got {tuple(x.shape)}")
		batch_size, time_steps, num_assets = x.shape
		if num_assets != self.num_assets:
			raise ValueError(f"Expected {self.num_assets} assets, got {num_assets}")

		sequences = x.transpose(1, 2).reshape(batch_size * num_assets, time_steps, 1)
		encoded, _ = self.temporal_encoder(sequences)
		asset_features = encoded[:, -1].reshape(batch_size, num_assets, self.hidden_dim)

		left = asset_features.unsqueeze(2).expand(-1, -1, num_assets, -1)
		right = asset_features.unsqueeze(1).expand(-1, num_assets, -1, -1)
		adjacency_logits = self.edge_score(left, right).squeeze(-1)
		adjacency_logits = (adjacency_logits + adjacency_logits.transpose(1, 2)) / 2
		adjacency_probabilities = torch.sigmoid(adjacency_logits)
		adjacency_probabilities = adjacency_probabilities * (
			1 - torch.eye(num_assets, device=x.device, dtype=x.dtype).unsqueeze(0)
		)

		pooled = torch.cat(
			[asset_features.mean(dim=1), asset_features.max(dim=1).values], dim=1
		)
		return adjacency_probabilities, self.regime_head(pooled).squeeze(1)
