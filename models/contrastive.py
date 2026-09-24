"""Supervised Contrastive Learning components for VORTEX-AI CG-NSDE framework."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for regime separation.
    
    Implementation of Khosla et al. 2020 formula as specified in masterplan Part D.4.
    
    L_con = sum_k [-1/|P(k)| * sum_{p in P(k)} log( exp(u_k.u_p/tau) / sum_{a!=k} exp(u_k.u_a/tau) )]
    
    Forces same-regime samples to cluster in latent space while pushing different regimes apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tau = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, proj_dim) - L2-normalized latent representations
            labels:   (B,)          - 0=normal, 1=crisis
        
        Returns:
            loss: scalar supervised contrastive loss
        """
        device = features.device
        B = features.shape[0]
        
        if B < 2:
            # Need at least 2 samples for contrastive learning
            return torch.tensor(0.0, device=device)
        
        # L2 normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix: (B, B)
        sim = torch.matmul(features, features.T) / self.tau

        # Create masks
        labels = labels.unsqueeze(1)  # (B, 1)
        mask_same = (labels == labels.T).float()  # Positive pairs (same regime)
        mask_self = torch.eye(B, device=device)
        mask_pos = mask_same - mask_self  # Remove self-similarity

        # Numerical stability: subtract max before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Compute exp(sim) excluding diagonal
        exp_sim = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        n_pos = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1)
        
        # Average over samples with at least one positive pair
        loss = (loss / (n_pos + 1e-8)).mean()
        
        return loss


class ProjectionHead(nn.Module):
    """
    Projection head for mapping latent SDE paths to contrastive learning space.
    
    As specified in masterplan Part D.4.
    
    Maps pooled latent representations to a lower-dimensional projection space
    where supervised contrastive loss is applied.
    """
    def __init__(self, latent_dim: int = 64, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, latent_dim) - latent SDE paths
        
        Returns:
            proj: (B, proj_dim) - projected representations for contrastive loss
        """
        # Temporal pooling: (B, T, latent_dim) -> (B, latent_dim)
        z_pool = z.mean(dim=1)
        
        # Project to contrastive space
        return self.net(z_pool)


# Legacy contrastive components - kept for backward compatibility
class NodeEncoder(nn.Module):
    """Legacy node encoder for asset representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """Legacy supervised InfoNCE loss using graph membership as positive grouping."""

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
        log_prob = similarity.masked_fill(~valid, float("-inf")) - torch.logsumexp(
            similarity.masked_fill(~valid, float("-inf")), dim=1, keepdim=True
        )
        return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1)).mean()


def info_nce_loss_simple(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Legacy simple InfoNCE loss."""
    if embeddings.shape[0] < 2:
        raise ValueError("at least two embeddings are required")
    labels = torch.arange(embeddings.shape[0], device=embeddings.device)
    return ContrastiveLoss(temperature)(
        torch.cat([embeddings, embeddings], dim=0), 
        torch.cat([labels, labels], dim=0)
    )
