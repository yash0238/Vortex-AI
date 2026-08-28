"""Losses for the graph reconstruction and regime classification tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def joint_loss(
	adjacency_pred: torch.Tensor,
	adjacency_true: torch.Tensor,
	regime_logits: torch.Tensor,
	regime_true: torch.Tensor,
	adjacency_weight: float = 1.0,
	binarize_adjacency: bool = False,
) -> torch.Tensor:
	"""Return weighted adjacency MSE plus binary regime cross entropy."""
	if binarize_adjacency:
		adjacency_loss = F.binary_cross_entropy(
			adjacency_pred.clamp(1e-6, 1 - 1e-6), adjacency_true
		)
	else:
		adjacency_loss = F.mse_loss(adjacency_pred, adjacency_true)
	regime_bce = F.binary_cross_entropy_with_logits(
		regime_logits, regime_true.float()
	)
	return adjacency_weight * adjacency_loss + regime_bce
