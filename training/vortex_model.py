"""PyTorch Lightning module for VORTEX-AI CG-NSDE end-to-end training.

Integrates all components as specified in masterplan Part D.7.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from typing import Any

from models.gat_encoder import DynamicGATEncoder
from models.neural_sde import LatentSDEModel
from models.contrastive import SupConLoss, ProjectionHead
from training.losses import total_loss


class VORTEXModel(pl.LightningModule):
    """
    Complete VORTEX-AI (CG-NSDE) model as PyTorch Lightning module.
    
    Architecture (as specified in masterplan):
    1. Dynamic GAT Encoder: learns graph structure and node embeddings
    2. Latent SDE Model: generates synthetic paths with graph conditioning
    3. Projection Head: maps latent paths to contrastive space
    4. Multi-term loss: reconstruction + graph + contrastive
    
    Args:
        n_stocks: number of stocks (50 for NIFTY-50)
        T: window length (60 days)
        in_feats: number of node features (6: return, abs_return, volume, sector, circuit_distance)
        latent_dim: latent SDE dimension
        proj_dim: contrastive projection dimension
        hidden_dim: hidden layer dimension for networks
        gat_heads: number of attention heads
        tau: contrastive loss temperature
        lr: learning rate
        lam_rec: reconstruction loss weight
        lam_graph: graph consistency loss weight
        lam_con: contrastive loss weight
        lam_na: no-arbitrage loss weight (future)
    """
    
    def __init__(
        self,
        n_stocks: int = 50,
        T: int = 60,
        in_feats: int = 6,
        latent_dim: int = 64,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        tau: float = 0.07,
        lr: float = 1e-3,
        lam_rec: float = 1.0,
        lam_graph: float = 0.1,
        lam_con: float = 0.1,
        lam_na: float = 0.05,
        beta_max: float = 0.01,  # 7.1: KL weight ceiling, ramped over prior_warmup
        prior_warmup: int = 10,
        lam_var: float = 0.0,  # 7.2: variance-matching loss weight
        lam_corr: float = 0.0,  # Lane 1A: correlation loss weight (OFF default)
        corr_ramp: int = 5,  # Lane 1A: ramp epochs for lam_corr
        vol_dyn: bool = False,  # Lane 1B: market vol state (OFF default)
        emission: str = "point",  # 7.3: "point" (MSE), "hetero"/"t" (NLL emission)
        rank_K: int = 0,  # 7.3: factor rank for emission covariance
        sde_sigma_max: float = 3.0,  # 7.3: slow-state bound (0.5-1 for hetero)
        use_gat: bool = True,  # 4.1: False = No-GAT ablation (MLP fallback)
        static_graph: bool = False,  # 4.1: True = freeze A_learned to input adj
        warmup_epochs: int = 20,  # contrastive ramp: full weight only after this many epochs
        lambda_styl: float = 0.1,  # stylized-facts weight inside reconstruction loss
        gat_dropout: float = 0.1,  # Stage 1: 0.6 per GAT paper
        weight_decay: float = 0.0,  # Stage 1: 0.0005 L2 per GAT paper
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model components
        self.gat = DynamicGATEncoder(
            in_feats=in_feats,
            hidden=hidden_dim,
            out_feats=latent_dim,
            heads=gat_heads,
            dropout=gat_dropout
        )
        self.gat_dropout = gat_dropout
        self.weight_decay = weight_decay
        # No-GAT ablation: per-node MLP replacing attention message passing.
        self.gat_fallback = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.use_gat = use_gat
        self.static_graph = static_graph
        
        self.sde_model = LatentSDEModel(
            n_stocks=n_stocks,
            latent_dim=latent_dim,
            T=T,
            sigma_max=sde_sigma_max,
            emission=emission,
            rank_K=rank_K,
        )
        
        self.proj_head = ProjectionHead(
            latent_dim=latent_dim,
            proj_dim=proj_dim
        )
        
        self.supcon = SupConLoss(temperature=tau)

        # 7.1: regime-conditional prior p(z0 | regime), one Gaussian per regime
        self.prior_mu = nn.Parameter(torch.zeros(2, latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(2, latent_dim))
        self.beta_max = beta_max
        self.prior_warmup = max(1, prior_warmup)
        self.lam_var = lam_var
        self.lam_corr = lam_corr
        self.corr_ramp = max(1, corr_ramp)
        self.vol_dyn = vol_dyn
        self._regime_corr = None  # lazy cache of data/raw/regime_corr.npz
        self.emission = emission
        self._use_nll = emission in ("hetero", "t", "mt")
        self.rank_K = rank_K
        self.sde_sigma_max = sde_sigma_max
        
        # Loss weights
        self.lam_rec = lam_rec
        self.lam_graph = lam_graph
        self.lam_con = lam_con
        self.lam_na = lam_na
        self.warmup_epochs = max(1, warmup_epochs)
        self.lambda_styl = lambda_styl
        self.lr = lr

    def _eff_lam_con(self) -> float:
        """Linear contrastive warmup: reconstruction first, regimes later."""
        progress = (self.current_epoch + 1) / self.warmup_epochs
        return self.lam_con * min(1.0, progress)

    def _eff_lam_corr(self) -> float:
        progress = (self.current_epoch + 1) / self.corr_ramp
        return self.lam_corr * min(1.0, progress)

    def _load_regime_corr(self, device) -> dict:
        if self._regime_corr is None:
            z = np.load("data/raw/regime_corr.npz")
            self._regime_corr = {k: torch.from_numpy(z[k]).float().to(device) for k in ("crisis", "normal")}
        return {k: v.to(device) for k, v in self._regime_corr.items()}

    @staticmethod
    def _pooled_corr(x: torch.Tensor) -> torch.Tensor:
        f = x.reshape(-1, x.shape[-1])
        f = f - f.mean(dim=0, keepdim=True)
        cov = (f.T @ f) / max(1, f.shape[0] - 1)
        sd = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
        return cov / sd.unsqueeze(1) / sd.unsqueeze(0)

    def correlation_loss(self, r_gen: torch.Tensor, regimes: torch.Tensor) -> torch.Tensor:
        """Lane 1A: per-regime pooled-corr Fro^2 vs train targets, regimes with >=4 windows."""
        tg = self._load_regime_corr(r_gen.device)
        n = r_gen.shape[-1]
        mask = ~torch.eye(n, device=r_gen.device, dtype=torch.bool)
        tot, cnt = 0.0, 0
        for rg, nm in ((1, "crisis"), (0, "normal")):
            sel = (regimes.long() == rg).nonzero(as_tuple=True)[0]
            if len(sel) < 4:
                continue
            cg = self._pooled_corr(r_gen[sel])
            tot = tot + ((cg[mask] - tg[nm][mask]).pow(2).sum() / (n * (n - 1)))
            cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=r_gen.device)
        return tot / cnt
        """Linear contrastive warmup: reconstruction first, regimes later.

        Early epochs have random projections and unscaled SDE outputs, so a
        full contrastive weight fights rescaling. Ramp 0 -> lam_con over
        warmup_epochs; 1.0 thereafter.
        """
        progress = (self.current_epoch + 1) / self.warmup_epochs
        return self.lam_con * min(1.0, progress)

    def _eff_beta(self) -> float:
        """KL weight ramp 0 -> beta_max over prior_warmup epochs."""
        progress = (self.current_epoch + 1) / self.prior_warmup
        return self.beta_max * min(1.0, progress)

    @staticmethod
    def variance_loss(r_hat: torch.Tensor, r_real: torch.Tensor) -> torch.Tensor:
        """7.2: mean over stocks of (log std_gen - log std_real)^2, z-space pooled over batch x time."""
        sg = r_hat.std(dim=(0, 1)) + 1e-8
        sr = r_real.std(dim=(0, 1)) + 1e-8
        return ((torch.log(sg) - torch.log(sr)).pow(2)).mean()

    @staticmethod
    def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
        """Mean KL(N(mu_q, var_q) || N(mu_p, var_p)) over batch and dim."""
        return 0.5 * (logvar_p - logvar_q
                      + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
                      - 1.0).mean()

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor, 
        node_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete VORTEX-AI pipeline.
        
        Args:
            x: (B, T, N) windowed returns
            adj: (B, N, N) empirical adjacency matrix
            node_feats: (B, N, F) node features
        
        Returns:
            r_hat: (B, T, N) reconstructed returns
            A_learned: (B, N, N) learned adjacency
            z_proj: (B, proj_dim) projected latent features
        """
        # 1. Graph encoding (GAT or MLP fallback for no-gat ablation)
        if self.use_gat:
            node_embs, A_learned = self.gat(node_feats, adj)  # (B, N, H), (B, N, N)
        else:
            node_embs = self.gat_fallback(node_feats)  # (B, N, H), no message passing
            A_learned = adj.detach()  # no learned structure; graph loss ~ const
        if self.static_graph:
            # Static-graph ablation: freeze learned adjacency to input target
            # (graph-consistency term becomes ~0 by construction).
            A_learned = adj.detach()
        
        # 2. Pool node embeddings to graph-level representation
        graph_emb = node_embs.mean(dim=1)  # (B, H)
        
        # 3. Latent SDE: sample z0 from posterior in training, deterministic h0 in eval
        r_hat, zs = self.sde_model(x, graph_emb, sample=self.training)
        self._last_zs = zs

        # 4. Project to contrastive space
        z_proj = self.proj_head(zs)  # (B, proj_dim)

        return r_hat, A_learned, z_proj

    def _kl_to_prior(self, regimes: torch.Tensor) -> torch.Tensor:
        """KL(q(z0|x) || p(z0|regime)) using stats stashed by the SDE forward."""
        stats = getattr(self.sde_model, "_last_stats", None)
        if stats is None:
            return torch.tensor(0.0, device=regimes.device)
        mu_q, logvar_q = stats
        r = regimes.long().clamp(0, 1)
        return self.gaussian_kl(mu_q, logvar_q, self.prior_mu[r], self.prior_logvar[r])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with complete loss assembly."""
        x = batch["returns"]  # (B, T, N)
        adj = batch["adj_matrix"]  # (B, N, N)
        node_feats = batch["node_features"]  # (B, N, F)
        A_emp = batch["adj_empirical"]  # (B, N, N)
        regimes = batch["regimes"]  # (B,)
        
        # Forward pass
        r_hat, A_learned, z_proj = self(x, adj, node_feats)

        if self._use_nll:
            # 7.3: NLL replaces MSE recon and L_var; MSE kept as diagnostic only
            zs_tail = self._last_zs
            if self.vol_dyn:
                nll = self.sde_model.emission_nll_vol(zs_tail, x)
                rho, kappa = self.sde_model._rho_kappa(x)
                self.log("train/rho", rho.detach(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/kappa", kappa.detach(), on_step=False, on_epoch=True, prog_bar=False)
            else:
                nll = self.sde_model.emission_nll(zs_tail, x)
            mse_diag = torch.nn.functional.mse_loss(r_hat, x).detach()
            kl = self._kl_to_prior(regimes)
            loss = nll + self._eff_beta() * kl
            self.log("train/nll", nll, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/mse_diag", mse_diag, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/kl", kl, on_step=True, on_epoch=True, prog_bar=True)
            if self.lam_corr > 0:
                # Lane 1A: correlation loss on rsampled scenarios (nu detached inside)
                r_samp = self.sde_model.emission_sample_corr(self._last_zs)
                lcorr = self.correlation_loss(r_samp, regimes)
                wcorr = self._eff_lam_corr() * lcorr
                loss = loss + wcorr
                self.log("train/corr_raw", lcorr.detach(), on_step=True, on_epoch=True, prog_bar=True)
                self.log("train/corr_w", wcorr.detach(), on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/total", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        # Compute total loss
        loss, components = total_loss(
            r_real=x,
            r_hat=r_hat,
            a_learned=A_learned,
            a_empirical=A_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lam_rec=self.lam_rec,
            lam_graph=self.lam_graph,
            lam_con=self._eff_lam_con(),
            lam_na=self.lam_na,
            lambda_styl=self.lambda_styl,
        )

        # 7.1: KL to regime-conditional prior (beta ramped)
        beta = self._eff_beta()
        kl = self._kl_to_prior(regimes)
        loss = loss + beta * kl
        components["kl"] = kl.detach()
        # 7.2: variance matching (z-space pooled std per stock)
        lvar = self.variance_loss(r_hat, x)
        loss = loss + self.lam_var * lvar
        components["var"] = lvar.detach()
        components["total"] = loss.detach()

        # Log all components
        for key, value in components.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch["returns"]
        adj = batch["adj_matrix"]
        node_feats = batch["node_features"]
        A_emp = batch["adj_empirical"]
        regimes = batch["regimes"]
        
        # Forward pass
        r_hat, A_learned, z_proj = self(x, adj, node_feats)

        if self._use_nll:
            if self.vol_dyn:
                nll = self.sde_model.emission_nll_vol(self._last_zs, x)
            else:
                nll = self.sde_model.emission_nll(self._last_zs, x)
            mse_diag = torch.nn.functional.mse_loss(r_hat, x).detach()
            self.log("val/nll", nll, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/mse_diag", mse_diag, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/total", nll, on_step=False, on_epoch=True, prog_bar=True)
            with torch.no_grad():
                mu_q, logvar_q, _ = self.sde_model.encode_stats(x)
                rv = regimes.long().clamp(0, 1)
                self.log("val/kl", self.gaussian_kl(mu_q, logvar_q, self.prior_mu[rv],
                                                    self.prior_logvar[rv]),
                         on_step=False, on_epoch=True, prog_bar=True)
            return nll

        # Compute total loss
        loss, components = total_loss(
            r_real=x,
            r_hat=r_hat,
            a_learned=A_learned,
            a_empirical=A_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lam_rec=self.lam_rec,
            lam_graph=self.lam_graph,
            lam_con=self._eff_lam_con(),
            lam_na=self.lam_na,
            lambda_styl=self.lambda_styl,
        )

        # Log all components
        for key, value in components.items():
            key_val = key.replace("loss/", "")
            self.log(f"val/{key_val}", value, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            mu_q, logvar_q, _ = self.sde_model.encode_stats(x)
            r = regimes.long().clamp(0, 1)
            kl_val = self.gaussian_kl(mu_q, logvar_q, self.prior_mu[r], self.prior_logvar[r])
        self.log("val/kl", kl_val, on_step=False, on_epoch=True, prog_bar=True)
        lvar = self.variance_loss(r_hat, x)
        self.log("val/var", lvar, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/total_incl", loss + self.lam_var * lvar + self._eff_beta() * kl_val,
                 on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step (mirrors validation, logs under test/)."""
        x = batch["returns"]
        adj = batch["adj_matrix"]
        node_feats = batch["node_features"]
        A_emp = batch["adj_empirical"]
        regimes = batch["regimes"]
        r_hat, A_learned, z_proj = self(x, adj, node_feats)
        loss, components = total_loss(
            r_real=x,
            r_hat=r_hat,
            a_learned=A_learned,
            a_empirical=A_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lam_rec=self.lam_rec,
            lam_graph=self.lam_graph,
            lam_con=self._eff_lam_con(),
            lam_na=self.lam_na,
            lambda_styl=self.lambda_styl,
        )
        for key, value in components.items():
            key_val = key.replace("loss/", "")
            self.log(f"test/{key_val}", value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with cosine annealing schedule."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    def generate_scenarios(
        self, 
        x: torch.Tensor,
        adj: torch.Tensor,
        node_feats: torch.Tensor,
        n_scenarios: int = 100
    ) -> torch.Tensor:
        """
        Generate multiple synthetic scenarios for stress testing.
        
        Args:
            x: (B, T, N) conditioning returns
            adj: (B, N, N) adjacency matrix
            node_feats: (B, N, F) node features
            n_scenarios: number of scenarios to generate
        
        Returns:
            scenarios: (n_scenarios, T, N) synthetic return paths
        """
        self.eval()
        scenarios = []

        with torch.no_grad():
            # Get graph embedding (GAT or MLP fallback, mirroring forward)
            if self.use_gat:
                node_embs, _ = self.gat(node_feats, adj)
            else:
                node_embs = self.gat_fallback(node_feats)
            graph_emb = node_embs.mean(dim=1)

            # 2.8: retain every Monte Carlo draw and cycle through all
            # conditioning batch elements (was: r_hat[0] every time, which
            # discarded batch elements 1..B).
            batch_size = x.shape[0]
            for i in range(n_scenarios):
                r_mean, zs_i = self.sde_model(x, graph_emb)
                if self._use_nll:
                    if self.vol_dyn:
                        scenarios.append(self.sde_model.emission_sample_seq(zs_i)[i % batch_size])
                    else:
                        scenarios.append(self.sde_model.emission_sample(zs_i)[i % batch_size])
                else:
                    scenarios.append(r_mean[i % batch_size])  # (T, N)

        return torch.stack(scenarios)

    @torch.no_grad()
    def sample_prior_scenarios(
        self,
        adj: torch.Tensor,
        node_feats: torch.Tensor,
        regimes: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one scenario per input row with z0 from the regime prior.

        Graph context comes from the real row (same-period conditioning);
        the initial state comes from p(z0 | regime). Returns (B, T, N).
        """
        self.eval()
        if self.use_gat:
            node_embs, _ = self.gat(node_feats, adj)
        else:
            node_embs = self.gat_fallback(node_feats)
        graph_emb = node_embs.mean(dim=1)
        r = regimes.long().clamp(0, 1)
        eps = torch.randn(graph_emb.shape[0], self.sde_model.latent_dim, device=graph_emb.device)
        z0 = self.prior_mu[r] + eps * torch.exp(0.5 * self.prior_logvar[r])
        r_mean, zs = self.sde_model.decode_from_z0(graph_emb, z0)
        if self._use_nll:
            # 7.3: generation SAMPLES the emission, never the mean
            if self.vol_dyn:
                return self.sde_model.emission_sample_seq(zs)
            return self.sde_model.emission_sample(zs)
        return r_mean
