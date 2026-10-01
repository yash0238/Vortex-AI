"""Neural SDE with graph-conditioned drift and diffusion.

Implements Block 3 of the CG-NSDE architecture (masterplan Part D.3):
- Input: latent z0 (B, latent_dim), graph_emb (B, H), ts (T,)
- Output: latent paths (B, T, latent_dim), decoded returns (B, T, N)
- Solver: Euler-Maruyama (fixed step, method='euler')

SDE formulation (Itô):
    dX_t = mu_theta(X_t, G_t, t) dt + sigma_theta(X_t, G_t, t) dW_t

The graph embedding G_t (pooled GAT node embeddings) is injected as static
context via set_graph_context(), making the SDE coefficients graph-conditioned.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchsde


class GraphConditionedSDE(torchsde.SDEIto):
    """Itô SDE with graph-conditional drift and diffusion.

    Args:
        latent_dim: Dimension of the latent state X_t.
        graph_emb_dim: Dimension of the graph context G_t (pooled GAT output).
        hidden_dim: Width of the MLP hidden layers for drift/diffusion networks.

    The drift and diffusion networks receive the concatenated input
    [X_t, G_t, t] and produce (B, latent_dim) outputs for the SDE solver.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_dim: int = 64, graph_emb_dim: int = 64,
                 hidden_dim: int = 128) -> None:
        super().__init__(noise_type="diagonal")
        self.latent_dim = latent_dim
        self.graph_emb_dim = graph_emb_dim
        input_dim = latent_dim + graph_emb_dim + 1

        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),  # 7.2: (0, 1), scaled by sigma_max in g()
        )

        self.graph_context: torch.Tensor | None = None
        self.sigma_max: float = 3.0  # 7.2: diffusion bound in z-space

    def set_graph_context(self, graph_emb: torch.Tensor) -> None:
        """Inject graph embedding from GAT mean pooling.

        Args:
            graph_emb: (B, graph_emb_dim) — pooled GAT node embeddings.
        """
        self.graph_context = graph_emb

    def _get_input(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenate state, graph context, and scalar time.

        Args:
            t: Scalar tensor (0-dim) or (B,) vector.
            y: (B, latent_dim) — current latent state.
        Returns:
            (B, latent_dim + graph_emb_dim + 1) — concatenated input.
        """
        if self.graph_context is None:  # 2.9: loud failure, was silent None deref
            raise RuntimeError(
                "GraphConditionedSDE: graph_context is None. "
                "Call set_graph_context(graph_emb) before sdeint."
            )
        batch_size = y.shape[0]
        t_vec = t.expand(batch_size, 1) if t.dim() == 0 else t.unsqueeze(-1)
        return torch.cat([y, self.graph_context, t_vec], dim=-1)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: tanh-bounded mu_theta(X_t, G_t, t) (7.2)."""
        inp = self._get_input(t, y)
        return torch.tanh(self.drift_net(inp))

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion: sigma_max * sigmoid output (7.2, z-space bound)."""
        inp = self._get_input(t, y)
        return self.sigma_max * torch.sigmoid(self.diffusion_net(inp))


class LatentSDEModel(nn.Module):
    """Encoder + SDE + Decoder for graph-conditioned latent path generation.

    Block 2 (encoder) + Block 3 (SDE) + Block 5 (decoder) per masterplan.

    Args:
        n_stocks: Number of assets (50 for NIFTY-50).
        latent_dim: Dimension of latent space.
        T: Sequence length (60 trading days).
        sde_hidden_dim: Hidden dimension width for SDE drift/diffusion networks.
        graph_emb_dim: Graph context dim (default: latent_dim for GAT pooling).

    Forward:
        x: (B, T, N) — windowed return sequences
        graph_emb: (B, H) — pooled GAT node embeddings
        ts: (T,) — time grid (default: linspace(0, 1, T))
    Returns:
        r_hat: (B, T, N) — decoded return estimates
        zs: (B, T, latent_dim) — latent paths
    """

    def __init__(self, n_stocks: int = 50, latent_dim: int = 64,
                 T: int = 60, sde_hidden_dim: int = 128,
                 graph_emb_dim: int | None = None, sigma_max: float = 3.0,
                 emission: str = "point", rank_K: int = 0) -> None:
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.n_stocks = n_stocks
        self.emission = emission
        self.rank_K = rank_K
        if graph_emb_dim is None:  # 2.9: was hardcoded to latent_dim
            graph_emb_dim = latent_dim
        self.graph_emb_dim = graph_emb_dim

        self.encoder = nn.GRU(n_stocks, latent_dim, batch_first=True)
        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)

        self.sde = GraphConditionedSDE(
            latent_dim=latent_dim,
            graph_emb_dim=graph_emb_dim,
            hidden_dim=sde_hidden_dim,
        )
        self.sde.sigma_max = sigma_max

        self.decoder = nn.Linear(latent_dim, n_stocks)
        if emission == "hetero":
            self.emi_m = nn.Linear(latent_dim, n_stocks)
            self.emi_d = nn.Linear(latent_dim, n_stocks)
            nn.init.constant_(self.emi_d.bias, 0.54)  # softplus(0.54) ~= 1.0
            self.emi_B = nn.Linear(latent_dim, n_stocks * rank_K) if rank_K > 0 else None

    def encode_stats(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GRU hidden state plus variational parameters. Returns (mu, logvar, h0)."""
        _, h_n = self.encoder(x)
        h0 = h_n.squeeze(0)
        return self.to_mu(h0), self.to_logvar(h0), h0

    @staticmethod
    def sample_z0(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_from_z0(
        self,
        graph_emb: torch.Tensor,
        z0: torch.Tensor,
        ts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SDE + decoder from a provided initial state (prior sampling)."""
        if ts is None:
            ts = torch.linspace(0, 1, self.T, device=z0.device)
        self.sde.set_graph_context(graph_emb)
        zs = torchsde.sdeint(
            self.sde, z0, ts,
            method="euler",
            dt=1.0 / self.T,
            names={"drift": "f", "diffusion": "g"},
        )
        zs = zs.permute(1, 0, 2)
        return self.decoder(zs), zs

    def emission_dist(self, zs: torch.Tensor):
        """7.3: per-step heteroscedastic emission. zs (B, T, latent) -> dist over (B, T, N)."""
        from torch.distributions import Independent, LowRankMultivariateNormal, Normal
        m = 0.5 * torch.tanh(self.emi_m(zs))
        d = torch.nn.functional.softplus(self.emi_d(zs)) + 0.01
        if self.rank_K > 0 and self.emi_B is not None:
            Bf = 0.5 * torch.tanh(self.emi_B(zs)).view(zs.shape[0], zs.shape[1], self.n_stocks, self.rank_K)
            return LowRankMultivariateNormal(m, Bf, d)
        return Independent(Normal(m, torch.sqrt(d)), 1)

    def emission_nll(self, zs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mean -log_prob over batch, time (and stocks via the joint event)."""
        return -self.emission_dist(zs).log_prob(x).mean()

    def emission_sample(self, zs: torch.Tensor) -> torch.Tensor:
        return self.emission_dist(zs).rsample()

    def emission_sigma(self, zs: torch.Tensor) -> torch.Tensor:
        """Per-step marginal std (B, T, N) for probes."""
        d = torch.nn.functional.softplus(self.emi_d(zs)) + 0.01
        if self.rank_K > 0 and self.emi_B is not None:
            Bf = 0.5 * torch.tanh(self.emi_B(zs)).view(zs.shape[0], zs.shape[1], self.n_stocks, self.rank_K)
            return torch.sqrt(d + (Bf.pow(2)).sum(-1))
        return torch.sqrt(d)

    def forward(
        self,
        x: torch.Tensor,
        graph_emb: torch.Tensor,
        ts: torch.Tensor | None = None,
        sample: bool = False,
        z0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z0 is None:
            mu, logvar, h0 = self.encode_stats(x)
            if sample:
                z0 = self.sample_z0(mu, logvar)
                self._last_stats = (mu, logvar)
            else:
                z0 = h0
                self._last_stats = None
        if ts is not None and ts.device != z0.device:
            ts = ts.to(z0.device)
        return self.decode_from_z0(graph_emb, z0, ts)


__all__ = ["GraphConditionedSDE", "LatentSDEModel"]


if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1] Single-stock sanity test (B=4, T=60, N=1)")
    model = LatentSDEModel(n_stocks=1, latent_dim=8, T=60).to(device)
    x = torch.randn(4, 60, 1, device=device) * 0.01
    graph_emb = torch.zeros(4, 8, device=device)
    start = time.time()
    r_hat, zs = model(x, graph_emb)
    elapsed = time.time() - start
    print(f"  r_hat shape: {r_hat.shape}, zs shape: {zs.shape}")
    print(f"  r_hat finite: {torch.isfinite(r_hat).all().item()}")
    print(f"  r_hat mean={r_hat.mean():.4f}, std={r_hat.std():.4f}")
    print(f"  zs std={zs.std():.4f}")
    print(f"  Time: {elapsed:.2f}s")
    assert r_hat.shape == (4, 60, 1)
    assert zs.shape == (4, 60, 8)
    assert torch.isfinite(r_hat).all()
    print("  PASSED")

    print("\n[2] Full 50-stock forward pass (B=8, T=60, N=50)")
    model = LatentSDEModel(n_stocks=50, latent_dim=64, T=60).to(device)
    x = torch.randn(8, 60, 50, device=device) * 0.01
    graph_emb = torch.randn(8, 64, device=device)
    start = time.time()
    r_hat, zs = model(x, graph_emb)
    elapsed = time.time() - start
    print(f"  r_hat shape: {r_hat.shape}, zs shape: {zs.shape}")
    print(f"  r_hat finite: {torch.isfinite(r_hat).all().item()}")
    print(f"  r_hat mean={r_hat.mean():.4f}, std={r_hat.std():.4f}")
    print(f"  zs std={zs.std():.4f}")
    print(f"  Time: {elapsed:.2f}s")
    assert r_hat.shape == (8, 60, 50)
    assert zs.shape == (8, 60, 64)
    assert torch.isfinite(r_hat).all()
    assert elapsed < 30.0
    print("  PASSED")

    print("\nAll tests passed.")
