"""Contrastive learning components for asset representations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32) -> None:
		super().__init__()
		self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class ContrastiveLoss(nn.Module):
	"""Supervised InfoNCE loss using graph membership as positive grouping."""

	def __init__(self, temperature: float = 0.1) -> None:
		super().__init__()
		if temperature <= 0:
			raise ValueError("temperature must be positive")
		self.temperature = temperature

	def forward(self, embeddings: torch.Tensor, batch_indices: torch.Tensor | None = None) -> torch.Tensor:
		if embeddings.ndim != 2 or embeddings.shape[0] < 2:
			raise ValueError("embeddings must have shape (samples, features) with at least two samples")
		if batch_indices is None:
			batch_indices = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
		batch_indices = batch_indices.to(embeddings.device)
		if batch_indices.numel() != embeddings.shape[0]:
			raise ValueError("batch_indices must contain one entry per embedding")
		similarity = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T / self.temperature
		valid = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
		positive = (batch_indices[:, None] == batch_indices[None, :]) & valid
		if not positive.any(dim=1).all():
			return embeddings.sum() * 0.0
		log_prob = similarity.masked_fill(~valid, float("-inf")) - torch.logsumexp(similarity.masked_fill(~valid, float("-inf")), dim=1, keepdim=True)
		return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1)).mean()


def info_nce_loss_simple(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
	if embeddings.shape[0] < 2:
		raise ValueError("at least two embeddings are required")
	labels = torch.arange(embeddings.shape[0], device=embeddings.device)
	return ContrastiveLoss(temperature)(torch.cat([embeddings, embeddings], dim=0), torch.cat([labels, labels], dim=0))
