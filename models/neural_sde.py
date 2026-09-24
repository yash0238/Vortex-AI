"""Graph-Conditioned Neural SDE for VORTEX-AI CG-NSDE framework."""

from __future__ import annotations

import torch
from torch import nn
import torchsde


class GraphConditionedSDE(torchsde.SDEIto):
    """
    Graph-conditioned Neural SDE with dynamic graph context.
    
    The SDE: dX_t = mu_theta(X_t, G_t, z_t, t) dt + sigma_theta(X_t, G_t, z_t, t) dW_t
    
    As specified in VORTEX-AI masterplan Part D.3.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_dim: int = 64, graph_emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Drift network: mu_theta(X_t, G_t, t)
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + graph_emb_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Diffusion network: sigma_theta(X_t, G_t, t)
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + graph_emb_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()  # Ensure positive diffusion
        )

        self.graph_context = None  # (B, graph_emb_dim)

    def set_graph_context(self, graph_emb: torch.Tensor):
        """Set the graph embedding context for the current forward pass."""
        self.graph_context = graph_emb

    def _get_input(self, t, y):
        """Concatenate state, graph context, and time."""
        B = y.shape[0]
        t_vec = t.expand(B, 1) if t.dim() == 0 else t.unsqueeze(-1)
        return torch.cat([y, self.graph_context, t_vec], dim=-1)

    def f(self, t, y):
        """Drift function."""
        inp = self._get_input(t, y)
        return self.drift_net(inp)

    def g(self, t, y):
        """Diffusion function."""
        inp = self._get_input(t, y)
        return self.diffusion_net(inp)


class LatentSDEModel(nn.Module):
    """
    Complete Latent SDE Model combining encoder, SDE, and decoder.
    
    As specified in VORTEX-AI masterplan Part D.3.
    
    Architecture:
    1. GRU encoder: encode windowed returns -> latent z0
    2. Graph-conditioned SDE: evolve z0 through time with graph context
    3. Linear decoder: decode latent paths -> reconstructed returns
    """
    def __init__(self, n_stocks: int = 50, latent_dim: int = 64, T: int = 60):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.n_stocks = n_stocks
        
        # Encoder: returns -> latent z0
        self.encoder = nn.GRU(n_stocks, latent_dim, batch_first=True)
        
        # Graph-conditioned SDE
        self.sde = GraphConditionedSDE(latent_dim=latent_dim, graph_emb_dim=latent_dim)
        
        # Decoder: latent -> returns
        self.decoder = nn.Linear(latent_dim, n_stocks)

    def forward(self, x: torch.Tensor, graph_emb: torch.Tensor, ts: torch.Tensor | None = None):
        """
        Args:
            x: (B, T, N) windowed returns
            graph_emb: (B, H) pooled graph embeddings from GAT
            ts: (T,) time points for SDE integration (defaults to linspace(0, 1, T))
        
        Returns:
            r_hat: (B, T, N) reconstructed returns
            zs: (B, T, latent_dim) latent SDE paths
        """
        B, T, N = x.shape
        
        # Default time points
        if ts is None:
            ts = torch.linspace(0, 1, T, device=x.device)

        # Encode: (B, T, N) -> z0 (B, latent_dim)
        _, h_n = self.encoder(x)
        z0 = h_n.squeeze(0)  # (B, latent_dim)

        # Set graph context for SDE
        self.sde.set_graph_context(graph_emb)

        # Integrate SDE: z0 -> zs (T, B, latent_dim)
        zs = torchsde.sdeint(
            self.sde, 
            z0, 
            ts,
            method="euler",
            dt=1.0 / T,
            names={"drift": "f", "diffusion": "g"}
        )

        # Transpose to (B, T, latent_dim)
        zs = zs.permute(1, 0, 2)
        
        # Decode: (B, T, latent_dim) -> (B, T, N)
        r_hat = self.decoder(zs)
        
        return r_hat, zs


# Legacy NeuralSDE - kept for backward compatibility
class NeuralSDE(nn.Module):
    """Legacy neural drift and diffusion model with Euler-Maruyama integration."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.drift = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, state_dim)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, state_dim), 
            nn.Softplus()
        )

    def forward(self, t, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.drift(x), self.diffusion(x)

    def integrate(
        self, 
        x0: torch.Tensor, 
        dt: float = 0.01, 
        n_steps: int = 100, 
        brownian: torch.Tensor | None = None, 
        return_path: bool = False
    ) -> torch.Tensor:
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
