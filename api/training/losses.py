"""Loss functions for the CG-NSDE framework.

Implements both the baseline LSTM losses (joint_loss) and the full
CG-NSDE multi-component loss (reconstruction + stylized facts +
graph consistency + supervised contrastive).

Masterplan reference: Part D.6
"""

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
    """Return weighted adjacency MSE plus binary regime cross entropy.

    Used by the baseline LSTM trainer (training/trainer.py).
    """
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


def stylized_facts_loss(
    r_real: torch.Tensor,
    r_hat: torch.Tensor,
) -> torch.Tensor:
    """Loss matching financial stylized facts: kurtosis and ACF of squared returns.

    Both inputs: (B, T, N) or (B, T). Kurtosis captures fat tails;
    ACF of squared returns captures volatility clustering.

    Scale note (2026-09-26): raw kurtosis MSE explodes because real kurtosis
    is O(100-1000) on outlier windows (e.g. 527 vs 20 -> MSE ~257k), drowning
    MSE/graph/contrastive terms and freezing val progress. The kurtosis term
    is therefore a RELATIVE squared error (~O(1)); the ACF term stays absolute
    since ACF lives in [-1, 1].
    """
    r_real_flat = r_real.reshape(-1)
    r_hat_flat = r_hat.reshape(-1)

    kurt_real = _kurtosis(r_real_flat)
    kurt_hat = _kurtosis(r_hat_flat)
    kurt_loss = ((kurt_hat - kurt_real) / (kurt_real.abs().detach() + 1.0)).pow(2)

    acf_real = _acf_squared(r_real)
    acf_hat = _acf_squared(r_hat)
    acf_loss = F.mse_loss(acf_hat, acf_real)

    return kurt_loss + acf_loss


def _kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis (Fisher) of a 1D tensor."""
    mu = x.mean()
    sigma = x.std() + 1e-8
    return ((x - mu) / sigma).pow(4).mean() - 3.0


def _acf_squared(x: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """Autocorrelation of squared residuals at given lag, averaged over assets.

    Args:
        x: (B, T, N) or (B, T) — returns
        lag: Autocorrelation lag (default 1 = day-to-day volatility clustering)
    """
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B, T, 1)

    sq = x.pow(2)
    sq_centered = sq - sq.mean(dim=1, keepdim=True)

    var = sq_centered.pow(2).mean(dim=1) + 1e-8
    acf = (sq_centered[:, :-lag] * sq_centered[:, lag:]).mean(dim=1) / var
    return acf.mean()


def reconstruction_loss(
    r_real: torch.Tensor,
    r_hat: torch.Tensor,
    lambda_styl: float = 0.1,
) -> torch.Tensor:
    """Combined reconstruction loss: MSE + stylized facts penalty."""
    mse = F.mse_loss(r_hat, r_real)
    styl = stylized_facts_loss(r_real, r_hat)
    return mse + lambda_styl * styl


def graph_consistency_loss(
    a_learned: torch.Tensor,
    a_empirical: torch.Tensor,
) -> torch.Tensor:
    """Frobenius norm between learned and empirical adjacency matrices.

    Args:
        a_learned: (B, N, N) — attention-derived adjacency from GAT
        a_empirical: (B, N, N) — Pearson/Granger adjacency targets
    """
    return (a_learned - a_empirical).pow(2).mean()


def circuit_filter_loss(
    r_hat: torch.Tensor,
    band: float = 0.10,
) -> torch.Tensor:
    """Soft penalty for returns exceeding NSE circuit filter bands.

    Penalizes squared magnitude of any return exceeding the band limit.

    Args:
        r_hat: (..., N) — predicted returns
        band: Circuit filter limit (default 10% for most NIFTY-50 stocks)
    """
    violation = torch.clamp(r_hat.abs() - band, min=0.0)
    return violation.pow(2).mean()


def total_loss(
    r_real: torch.Tensor,
    r_hat: torch.Tensor,
    a_learned: torch.Tensor,
    a_empirical: torch.Tensor,
    z_proj: torch.Tensor,
    regime_labels: torch.Tensor,
    supcon_loss_fn,
    lam_rec: float = 1.0,
    lam_graph: float = 0.1,
    lam_con: float = 0.1,
    lam_na: float = 0.05,
    lambda_styl: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assemble the full CG-NSDE loss.

    L_total = lam_rec * L_rec + lam_graph * L_graph
             + lam_con * L_con + lam_na * L_circuit

    Returns:
        total: weighted sum scalar tensor.
        components: dict with keys total/reconstruction/graph/contrastive/circuit.

    Args:
        r_real: (B, T, N) — true window returns
        r_hat: (B, T, N) — reconstructed returns
        a_learned: (B, N, N) — learned adjacency from GAT attention
        a_empirical: (B, N, N) — empirical adjacency targets
        z_proj: (B, proj_dim) — projection head output for contrastive loss
        regime_labels: (B,) — regime labels (0=normal, 1=crisis)
        supcon_loss_fn: Callable — SupConLoss instance
        lam_rec, lam_graph, lam_con, lam_na: Loss weights
        lambda_styl: Stylized facts weight in reconstruction loss
    """
    l_rec = reconstruction_loss(r_real, r_hat, lambda_styl=lambda_styl)
    l_graph = graph_consistency_loss(a_learned, a_empirical)
    l_con = supcon_loss_fn(z_proj, regime_labels)
    l_na = circuit_filter_loss(r_hat, band=0.10)

    total = (
        lam_rec * l_rec
        + lam_graph * l_graph
        + lam_con * l_con
        + lam_na * l_na
    )
    components = {
        "total": total,
        "reconstruction": l_rec,
        "graph": l_graph,
        "contrastive": l_con,
        "circuit": l_na,
    }
    return total, components


__all__ = [
    "joint_loss",
    "stylized_facts_loss",
    "reconstruction_loss",
    "graph_consistency_loss",
    "circuit_filter_loss",
    "total_loss",
]
