"""Contrastive learning components for asset representations.

Implements supervised contrastive learning (SupCon, Khosla et al. 2020)
for regime separation, plus a projection head for latent-to-contrastive-space
mapping.

Masterplan reference: Part D.4
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for regime separation.

    Forces same-regime samples to cluster closer in the projection space
    while pushing different-regime samples apart. Uses the exact formulation
    from Khosla et al. 2020 (arXiv:2004.11362).

    Args:
        temperature: Softmax temperature (default 0.07 per Khosla et al.).

    Forward:
        features: (B, proj_dim) — L2-normalized latent representations
        labels: (B,) — integer regime labels (0=normal, 1=crisis)

    Loss formula:
        L = sum_k [-1/|P(k)| * sum_{p in P(k)}
                   log( exp(u_k . u_p / tau) /
                        sum_{a != k} exp(u_k . u_a / tau) )]
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tau = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.tau

        labels = labels.unsqueeze(1)
        mask_same = (labels == labels.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_same - mask_self

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * (1 - mask_self)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1)
        loss = (loss / (n_pos + 1e-8)).mean()

        return loss


class ProjectionHead(nn.Module):
    """Maps latent representations to the contrastive projection space.

    Pools latent paths across time, then projects through a 2-layer MLP
    with ReLU activation.

    Args:
        latent_dim: Input latent dimension (GRU hidden or SDE output dim).
        proj_dim: Output projection dimension.

    Forward:
        z: (B, T, latent_dim) — latent paths
    Returns:
        (B, proj_dim) — projected features for SupCon
    """

    def __init__(self, latent_dim: int = 64, proj_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_pool = z.mean(dim=1)
        return self.net(z_pool)


class NodeEncoder(nn.Module):
    """MLP encoder for graph-structured node features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """Supervised InfoNCE loss using graph membership as positive grouping.

    .. deprecated:: Replaced by :class:`SupConLoss`.
    """

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
            # 2.5: loud failure instead of silent zero-loss (masked regressions).
            # Train path uses StratifiedBatchSampler (train_vortex.py) so every
            # batch holds both regimes; hitting this means the sampler broke.
            raise RuntimeError(
                "ContrastiveLoss: batch lacks positives for some samples "
                "(single-regime batch). Use StratifiedBatchSampler."
            )
        log_prob = similarity.masked_fill(~valid, float("-inf")) - torch.logsumexp(similarity.masked_fill(~valid, float("-inf")), dim=1, keepdim=True)
        return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1)).mean()


def info_nce_loss_simple(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Convenience wrapper around :class:`ContrastiveLoss`."""
    if embeddings.shape[0] < 2:
        raise ValueError("at least two embeddings are required")
    labels = torch.arange(embeddings.shape[0], device=embeddings.device)
    return ContrastiveLoss(temperature)(torch.cat([embeddings, embeddings], dim=0), torch.cat([labels, labels], dim=0))


__all__ = [
    "SupConLoss",
    "ProjectionHead",
    "NodeEncoder",
    "ContrastiveLoss",
    "info_nce_loss_simple",
]


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, proj_dim = 16, 32
    features = torch.randn(batch_size, proj_dim)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    supcon = SupConLoss(temperature=0.07)
    loss = supcon(features, labels)
    print(f"SupConLoss: {loss.item():.6f}")
    assert torch.isfinite(loss)
    assert loss.item() > 0
    print(f"Gradient exists: {loss.requires_grad}")

    proj = ProjectionHead(latent_dim=64, proj_dim=32)
    z = torch.randn(8, 60, 64)
    out = proj(z)
    print(f"ProjectionHead output shape: {out.shape}")
    assert out.shape == (8, 32)
    print("All contrastive tests passed.")
