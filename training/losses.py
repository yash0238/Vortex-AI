"""Complete loss assembly for VORTEX-AI CG-NSDE framework.

Implements all loss components as specified in masterplan Part D.6:
1. Reconstruction loss (MSE + stylized facts)
2. Graph consistency loss
3. Supervised contrastive loss
4. Total loss assembly
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def stylized_facts_loss(r_real: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """
    Loss to match stylized facts of financial returns.
    
    Enforces:
    1. Kurtosis matching (fat tails)
    2. ACF of squared returns (volatility clustering)
    
    As specified in masterplan Part D.6.
    
    Args:
        r_real: (B, T, N) real returns
        r_hat: (B, T, N) generated returns
    
    Returns:
        loss: scalar stylized facts loss
    """
    # Reshape to (B, T*N) for statistics computation
    r_real_flat = r_real.reshape(r_real.shape[0], -1)
    r_hat_flat = r_hat.reshape(r_hat.shape[0], -1)
    
    # 1. Kurtosis loss: match 4th moment
    def kurtosis(x):
        """Compute kurtosis (4th standardized moment)."""
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + 1e-8
        standardized = (x - mu) / sigma
        return standardized.pow(4).mean(dim=1)
    
    kurt_real = kurtosis(r_real_flat)
    kurt_hat = kurtosis(r_hat_flat)
    kurt_loss = F.mse_loss(kurt_hat, kurt_real)
    
    # 2. ACF of squared returns at lag 1 (volatility clustering)
    def acf_squared_lag1(x):
        """Compute autocorrelation of squared returns at lag 1."""
        sq = x.pow(2)
        sq_centered = sq - sq.mean(dim=1, keepdim=True)
        
        # Correlation between t and t-1
        numerator = (sq_centered[:, 1:] * sq_centered[:, :-1]).mean(dim=1)
        denominator = sq_centered.pow(2).mean(dim=1) + 1e-8
        
        return numerator / denominator
    
    acf_real = acf_squared_lag1(r_real_flat)
    acf_hat = acf_squared_lag1(r_hat_flat)
    acf_loss = F.mse_loss(acf_hat, acf_real)
    
    return kurt_loss + acf_loss


def reconstruction_loss(r_real: torch.Tensor, r_hat: torch.Tensor, 
                       lambda_styl: float = 0.1) -> torch.Tensor:
    """
    Reconstruction loss combining MSE and stylized facts.
    
    As specified in masterplan Part D.6.
    
    Args:
        r_real: (B, T, N) real returns
        r_hat: (B, T, N) generated returns
        lambda_styl: weight for stylized facts loss
    
    Returns:
        loss: scalar reconstruction loss
    """
    # Basic MSE reconstruction
    mse = F.mse_loss(r_hat, r_real)
    
    # Stylized facts matching
    styl = stylized_facts_loss(r_real, r_hat)
    
    return mse + lambda_styl * styl


def graph_consistency_loss(A_learned: torch.Tensor, A_empirical: torch.Tensor) -> torch.Tensor:
    """
    Graph consistency loss (Frobenius norm).
    
    Forces learned GAT attention weights to match empirical adjacency structure.
    
    As specified in masterplan Part D.6.
    
    Args:
        A_learned: (B, N, N) learned attention-based adjacency
        A_empirical: (B, N, N) empirical adjacency (Pearson/Granger)
    
    Returns:
        loss: scalar graph consistency loss
    """
    return (A_learned - A_empirical).pow(2).mean()


def total_loss(
    r_real: torch.Tensor,
    r_hat: torch.Tensor,
    A_learned: torch.Tensor,
    A_emp: torch.Tensor,
    z_proj: torch.Tensor,
    regime_labels: torch.Tensor,
    supcon_loss_fn: nn.Module,
    lam_rec: float = 1.0,
    lam_graph: float = 0.1,
    lam_con: float = 0.1,
    lam_na: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Total loss assembly for VORTEX-AI.
    
    As specified in masterplan Part D.6:
    
    TOTAL LOSS = lam_rec * L_rec + lam_graph * L_graph + lam_con * L_con + lam_na * L_NA
    
    Args:
        r_real: (B, T, N) real returns
        r_hat: (B, T, N) reconstructed returns
        A_learned: (B, N, N) learned adjacency
        A_emp: (B, N, N) empirical adjacency
        z_proj: (B, proj_dim) projected latent features
        regime_labels: (B,) regime labels (0=normal, 1=crisis)
        supcon_loss_fn: SupConLoss module instance
        lam_rec: weight for reconstruction loss
        lam_graph: weight for graph consistency loss
        lam_con: weight for contrastive loss
        lam_na: weight for no-arbitrage loss (future extension)
    
    Returns:
        total: scalar total loss
        components: dict of individual loss components for logging
    """
    # 1. Reconstruction loss (MSE + stylized facts)
    L_rec = reconstruction_loss(r_real, r_hat)
    
    # 2. Graph consistency loss
    L_graph = graph_consistency_loss(A_learned, A_emp)
    
    # 3. Supervised contrastive loss
    L_con = supcon_loss_fn(z_proj, regime_labels)
    
    # 4. No-arbitrage loss (placeholder for future extension)
    L_na = torch.tensor(0.0, device=r_real.device)
    
    # Total weighted loss
    total = (lam_rec * L_rec + 
             lam_graph * L_graph + 
             lam_con * L_con + 
             lam_na * L_na)
    
    # Return components for logging
    components = {
        "loss/total": total,
        "loss/reconstruction": L_rec,
        "loss/graph_consistency": L_graph,
        "loss/contrastive": L_con,
        "loss/no_arbitrage": L_na,
    }
    
    return total, components


# Legacy joint loss - kept for backward compatibility
def joint_loss(
    adjacency_predicted: torch.Tensor,
    adjacency_target: torch.Tensor,
    regime_predicted: torch.Tensor,
    regime_target: torch.Tensor,
    adjacency_weight: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy joint loss for baseline spatio-temporal model."""
    if adjacency_predicted.shape != adjacency_target.shape:
        raise ValueError(
            f"adjacency shape mismatch: predicted {adjacency_predicted.shape} vs target {adjacency_target.shape}"
        )
    if regime_predicted.shape[0] != regime_target.shape[0]:
        raise ValueError(
            f"regime batch size mismatch: predicted {regime_predicted.shape[0]} vs target {regime_target.shape[0]}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")

    adjacency_loss = F.binary_cross_entropy(adjacency_predicted, adjacency_target, reduction=reduction)
    regime_loss = F.binary_cross_entropy_with_logits(regime_predicted, regime_target.float(), reduction=reduction)
    combined = adjacency_weight * adjacency_loss + regime_loss

    return combined, adjacency_loss, regime_loss
