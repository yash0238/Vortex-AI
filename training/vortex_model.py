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
from training.losses import (
    total_loss,
    graph_consistency_loss,
    emission_correlation_loss,
    stylized_facts_loss,
)


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
        lam_emission_corr: float = 0.0,  # emission correlation loss weight (OFF default)
        vol_dyn: bool = False,  # Lane 1B: market vol state (OFF default)
        emission: str = "point",  # 7.3: "point" (MSE), "hetero"/"t" (NLL emission)
        rank_K: int = 0,  # 7.3: factor rank for emission covariance
        sde_sigma_max: float = 3.0,  # 7.3: slow-state bound (0.5-1 for hetero)
        use_gat: bool = True,  # 4.1: False = No-GAT ablation (MLP fallback)
        static_graph: bool = False,  # 4.1: True = freeze A_learned to input adj
        warmup_epochs: int = 20,  # contrastive ramp: full weight only after this many epochs
        lambda_styl: float = 0.1,  # stylized-facts weight inside reconstruction loss
        stylized_sampler: str = "seq",  # Defect 2: "seq" (generation's sampler) or "single" (legacy)
        gat_dropout: float = 0.1,  # Stage 1: 0.6 per GAT paper
        weight_decay: float = 0.0,  # Stage 1: 0.0005 L2 per GAT paper
        scheduler_epochs: int = 0,  # cosine T_max; 0 = 100
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
        # Initialize with small random values; will be trained via KL to encoder posterior
        self.prior_mu = nn.Parameter(torch.randn(2, latent_dim) * 0.1)
        self.prior_logvar = nn.Parameter(torch.zeros(2, latent_dim))
        # Generative prior per regime, fitted after training on the aggregate
        # posterior (fit_generation_priors). Together with prior_mu/prior_logvar
        # these are the ONLY inputs the generator gets: nothing about the window
        # it is asked to produce.
        self.regime_graph_emb = nn.Parameter(torch.zeros(2, latent_dim))
        self.regime_log_vol = nn.Parameter(torch.zeros(2))
        # Spread of the per-window log-volatility within each regime. Without it
        # every generated window of a regime shares a single volatility level,
        # which is a trivial structural tell for the discriminative test.
        self.regime_log_vol_std = nn.Parameter(torch.zeros(2))
        self.beta_max = beta_max
        self.prior_warmup = max(1, prior_warmup)
        self.lam_var = lam_var
        self.lam_corr = lam_corr
        self.lam_emission_corr = lam_emission_corr
        self.corr_ramp = max(1, corr_ramp)
        self.vol_dyn = vol_dyn
        self._regime_corr = None  # lazy cache of data/raw/regime_corr.npz
        self.emission = emission
        # "smt"/"mt"/"t"/"hetero" are likelihood emissions. Leaving "smt" out
        # trained the emission heads with an MSE-on-the-mean objective, so the
        # heads never left their initialisation and evaluation returned the
        # decoder mean instead of a sample.
        self._use_nll = emission in ("hetero", "t", "mt", "smt")
        self.rank_K = rank_K
        self.sde_sigma_max = sde_sigma_max
        
        # Loss weights
        self.lam_rec = lam_rec
        self.lam_graph = lam_graph
        self.lam_con = lam_con
        self.lam_na = lam_na
        self.warmup_epochs = max(1, warmup_epochs)
        self.lambda_styl = lambda_styl
        if stylized_sampler not in ("single", "seq"):
            raise ValueError(
                f"stylized_sampler must be 'single' or 'seq', got {stylized_sampler!r}"
            )
        # Defect 2: training/validation used emission_sample (one joint draw)
        # while generation used emission_sample_seq (vol-feedback recursion), so
        # the objective never trained the deployed sampler.
        self.stylized_sampler = stylized_sampler
        self.lr = lr
        self.scheduler_epochs = max(1, int(scheduler_epochs)) if scheduler_epochs else 100

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

    def set_regime_corr(self, corr: dict) -> None:
        """Install per-regime correlation targets computed from the aligned train split."""
        self._regime_corr = {
            k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
            for k, v in corr.items()
        }
        print("Regime correlation targets installed from the training split.")

    @staticmethod
    def _pooled_corr(x: torch.Tensor) -> torch.Tensor:
        f = x.reshape(-1, x.shape[-1])
        f = f - f.mean(dim=0, keepdim=True)
        cov = (f.T @ f) / max(1, f.shape[0] - 1)
        sd = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
        return cov / sd.unsqueeze(1) / sd.unsqueeze(0)

    def correlation_loss(self, r_gen: torch.Tensor, regimes: torch.Tensor) -> torch.Tensor:
        """Lane 1A: per-regime pooled-corr Fro^2 vs train targets, regimes with >=2 windows (batch_size=4, 50/50 split gives 2 per regime)."""
        tg = self._load_regime_corr(r_gen.device)
        n = r_gen.shape[-1]
        mask = ~torch.eye(n, device=r_gen.device, dtype=torch.bool)
        tot, cnt = 0.0, 0
        for rg, nm in ((1, "crisis"), (0, "normal")):
            sel = (regimes.long() == rg).nonzero(as_tuple=True)[0]
            if len(sel) < 2:  # Reduced from 4 to 2 for batch_size=4 with 50/50 split
                continue
            cg = self._pooled_corr(r_gen[sel])
            tot = tot + ((cg[mask] - tg[nm][mask]).pow(2).sum() / (n * (n - 1)))
            cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=r_gen.device)
        return tot / cnt

    def _eff_lambda_styl(self) -> float:
        """Ramp stylized facts weight 0 -> lambda_styl over warmup_epochs."""
        progress = (self.current_epoch + 1) / self.warmup_epochs
        return self.lambda_styl * min(1.0, progress)

    def _eff_beta(self) -> float:
        """KL weight ramp 0 -> beta_max over prior_warmup epochs."""
        progress = (self.current_epoch + 1) / self.prior_warmup
        return self.beta_max * min(1.0, progress)

    def _nll_per_asset(self, nll: torch.Tensor) -> torch.Tensor:
        """Likelihood per asset, so auxiliary moment terms are comparable.

        The raw NLL is a joint log-density over 50 assets (~65). Divided by the
        asset count it is ~1.3, which is the same order as the stylized-fact,
        correlation and graph terms. Without this the likelihood is 99% of the
        loss and every fidelity term is decoration.
        """
        return nll / max(1, self.sde_model.n_stocks)

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
            lv = self.sde_model.log_vol(x)
            if self.vol_dyn:
                nll = self.sde_model.emission_nll_vol(zs_tail, x, regimes, lv)
                rho, kappa = self.sde_model._rho_kappa(x)
                self.log("train/rho", rho.detach(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/kappa", kappa.detach(), on_step=False, on_epoch=True, prog_bar=False)
            else:
                nll = self.sde_model.emission_nll(zs_tail, x, regimes, lv)
            mse_diag = torch.nn.functional.mse_loss(r_hat, x).detach()
            kl = self._kl_to_prior(regimes)
            loss = self._nll_per_asset(nll) + self._eff_beta() * kl
            self.log("train/nll", nll, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/nll_per_asset", loss.detach(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/mse_diag", mse_diag, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/kl", kl, on_step=True, on_epoch=True, prog_bar=True)
            if self.lam_corr > 0:
                # Lane 1A: correlation loss on rsampled scenarios (nu detached inside)
                r_samp = self.sde_model.emission_sample_corr(self._last_zs, regimes, lv)
                lcorr = self.correlation_loss(r_samp, regimes)
                wcorr = self._eff_lam_corr() * lcorr
                loss = loss + wcorr
                self.log("train/corr_raw", lcorr.detach(), on_step=True, on_epoch=True, prog_bar=True)
                self.log("train/corr_w", wcorr.detach(), on_step=True, on_epoch=True, prog_bar=True)
            if self._eff_lambda_styl() > 0:
                # Stylized facts must be optimised explicitly: a joint likelihood
                # is nearly indifferent to 4% squared-return autocorrelation and
                # to the tails that make pooled kurtosis ~12.
                # Defect 2: optimise the sampler generation actually uses. The
                # sequential draw feeds the realised return back into the vol
                # state, which is the only path by which ACF(r^2) can be learnt.
                if self.stylized_sampler == "seq":
                    r_styl = self.sde_model.emission_sample_seq(self._last_zs, regimes, lv)
                else:
                    r_styl = self.sde_model.emission_sample(self._last_zs, regimes, lv)
                lstyl = stylized_facts_loss(x, r_styl)
                wstyl = self._eff_lambda_styl() * lstyl
                loss = loss + wstyl
                self.log("train/styl", lstyl.detach(), on_step=True, on_epoch=True, prog_bar=True)
                self.log("train/styl_w", wstyl.detach(), on_step=True, on_epoch=True, prog_bar=False)
            if self.lam_emission_corr > 0:
                # Correlation target on the emission covariance itself: this is
                # what makes crisis draws move together, and it was unreachable
                # because the NLL branch ignored lam_emission_corr.
                lemiss = emission_correlation_loss(
                    self.sde_model, self._last_zs, regimes, self._load_regime_corr(x.device),
                    log_vol=lv,
                )
                wemiss = self.lam_emission_corr * lemiss
                loss = loss + wemiss
                self.log("train/emission_corr_raw", lemiss.detach(), on_step=True, on_epoch=True, prog_bar=True)
                self.log("train/emission_corr_w", wemiss.detach(), on_step=True, on_epoch=True, prog_bar=True)
            if self.lam_graph > 0:
                lgraph = graph_consistency_loss(A_learned, A_emp)
                loss = loss + self.lam_graph * lgraph
                self.log("train/graph", lgraph.detach(), on_step=True, on_epoch=True, prog_bar=True)
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
            lam_corr=self._eff_lam_corr(),
            lam_emission_corr=self.lam_emission_corr,
            lambda_styl=self._eff_lambda_styl(),
            regime_corr=self._load_regime_corr(x.device),
            sde_model=self.sde_model,
            zs=self._last_zs,
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

        # Gradient norm logging + anomaly detection (first 5 epochs)
        if self.trainer is not None and self.global_step % 50 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True)
            
            # Check for NaN/Inf gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        self.log(f"train/grad_nan/{name}", 1.0, on_step=True)
                    if torch.isinf(param.grad).any():
                        self.log(f"train/grad_inf/{name}", 1.0, on_step=True)

        return loss

    def generate_from_prior(self, adj, node_feats, regimes):
        """Generate scenarios from regime-conditional prior (matches evaluation protocol)."""
        if self.use_gat:
            node_embs, _ = self.gat(node_feats, adj)
        else:
            node_embs = self.gat_fallback(node_feats)
        graph_emb = node_embs.mean(dim=1)  # (B, H)
        r = regimes.long().clamp(0, 1)
        eps = torch.randn_like(self.prior_mu[r])
        z0 = self.prior_mu[r] + eps * torch.exp(0.5 * self.prior_logvar[r])
        r_gen, _ = self.sde_model.decode_from_z0(graph_emb, z0)
        return r_gen, z0

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
            lv = self.sde_model.log_vol(x)
            if self.vol_dyn:
                nll = self.sde_model.emission_nll_vol(self._last_zs, x, regimes, lv)
            else:
                nll = self.sde_model.emission_nll(self._last_zs, x, regimes, lv)
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
                emiss = torch.tensor(0.0, device=x.device)
                if self.lam_emission_corr > 0:
                    emiss = emission_correlation_loss(self.sde_model, self._last_zs, regimes,
                                                      self._load_regime_corr(x.device),
                                                      log_vol=lv)
                self.log("val/emission_corr", emiss, on_step=False, on_epoch=True, prog_bar=True)
                # Defect 2: val/fidelity is the checkpoint metric, so it must be
                # the generation sampler, not a cheaper surrogate.
                if self.stylized_sampler == "seq":
                    r_val = self.sde_model.emission_sample_seq(self._last_zs, regimes, lv)
                else:
                    r_val = self.sde_model.emission_sample(self._last_zs, regimes, lv)
                styl = stylized_facts_loss(x, r_val)
                self.log("val/styl", styl, on_step=False, on_epoch=True, prog_bar=True)
                # The objective that matters for generation, and the metric to
                # checkpoint on: it does not depend on how well the encoder fits
                # the training windows.
                self.log("val/fidelity", emiss + styl, on_step=False, on_epoch=True,
                         prog_bar=True)
            if batch_idx == 0 and self.current_epoch % 5 == 0:
                self._log_gate_metrics(x, regimes, adj, node_feats)
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
            lam_corr=self._eff_lam_corr(),
            lambda_styl=self._eff_lambda_styl(),
            regime_corr=self._load_regime_corr(x.device),
            sde_model=self.sde_model,
            zs=self._last_zs,
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

        # Masterplan gate metrics (every 5 epochs, batch_idx=0 to avoid duplicate compute)
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_gate_metrics(x, regimes, adj, node_feats)

        return loss

    @staticmethod
    def _gate_metrics(real: torch.Tensor, gen: torch.Tensor, regimes: torch.Tensor) -> dict:
        """Real, side-by-side gate statistics (gen vs the same real windows).

        These used to be placeholders (`0.5`, `1.0`, a hardcoded 0.3 target),
        which is why the training log reported perfect-looking gates while the
        evaluation failed every one of them.
        """
        from eval.contagion import cvar_regime_ratio, evaluate_contagion
        from eval.statistical import _acf_lag1

        r = real.detach().cpu().numpy().astype(np.float64)
        g = gen.detach().cpu().numpy().astype(np.float64)
        reg = regimes.detach().cpu().numpy().astype(np.int64)

        def kurt(x):
            flat = x.reshape(-1)
            flat = (flat - flat.mean()) / (flat.std() + 1e-8)
            return float((flat ** 4).mean() - 3.0)

        def pooled_corr(x):
            f = x.reshape(-1, x.shape[-1])
            f = f - f.mean(0, keepdims=True)
            cov = f.T @ f / max(1, len(f) - 1)
            sd = np.sqrt(np.diag(cov)).clip(1e-12)
            return cov / sd[:, None] / sd[None, :]

        out = {
            "kurt_gen": kurt(g), "kurt_real": kurt(r),
            "acf_r2_gen": _acf_lag1(g), "acf_r2_real": _acf_lag1(r),
            "corr_fro": float(np.linalg.norm(pooled_corr(r) - pooled_corr(g), "fro")),
        }
        rc, gc = evaluate_contagion(r, reg), evaluate_contagion(g, reg)
        out["crisis_boost_gen"] = gc["crisis_corr_boost"]
        out["crisis_boost_real"] = rc["crisis_corr_boost"]
        rv, gv = cvar_regime_ratio(r, reg), cvar_regime_ratio(g, reg)
        out["cvar_ratio_gen"] = gv["cvar_ratio"]
        out["cvar_ratio_real"] = rv["cvar_ratio"]
        return out

    def _log_gate_metrics(self, x, regimes, adj, node_feats) -> None:
        """Generate from the regime prior and log the real gates.

        Uses `sample_regime_scenarios`, i.e. the same entry point the evaluator
        uses, so the logged gates describe the honest generator rather than a
        reconstruction of the batch.
        """
        if (regimes == 0).sum() == 0 or (regimes == 1).sum() == 0:
            return  # both regimes are needed for the contagion/CVaR statistics
        with torch.no_grad():
            gen = self.sample_regime_scenarios(regimes)
        metrics = self._gate_metrics(x, gen, regimes)
        prog = {
            "kurt_gen": "val/kurtosis",
            "acf_r2_gen": "val/acf_r2",
            "corr_fro": "val/corr_fro",
            "crisis_boost_gen": "val/crisis_boost",
            "cvar_ratio_gen": "val/cvar_ratio",
        }
        for key, value in metrics.items():
            name = prog.get(key, f"val/{key}")
            self.log(name, float(value), on_step=False, on_epoch=True,
                     prog_bar=name in prog.values())
        for key, real_key in (("kurt_real", "val/kurtosis_real"),
                              ("acf_r2_real", "val/acf_r2_real"),
                              ("crisis_boost_real", "val/crisis_boost_real"),
                              ("cvar_ratio_real", "val/cvar_ratio_real")):
            self.log(real_key, float(metrics[key]), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step (mirrors validation, logs under test/)."""
        x = batch["returns"]
        adj = batch["adj_matrix"]
        node_feats = batch["node_features"]
        A_emp = batch["adj_empirical"]
        regimes = batch["regimes"]
        r_hat, A_learned, z_proj = self(x, adj, node_feats)
        if self._use_nll:
            lv = self.sde_model.log_vol(x)
            if self.vol_dyn:
                nll = self.sde_model.emission_nll_vol(self._last_zs, x, regimes, lv)
            else:
                nll = self.sde_model.emission_nll(self._last_zs, x, regimes, lv)
            self.log("test/nll", nll, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/mse_diag", torch.nn.functional.mse_loss(r_hat, x),
                     on_step=False, on_epoch=True, prog_bar=False)
            return nll
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
            lam_corr=self._eff_lam_corr(),
            lambda_styl=self._eff_lambda_styl(),
            regime_corr=self._load_regime_corr(x.device),
            sde_model=self.sde_model,
            zs=self._last_zs,
        )
        for key, value in components.items():
            key_val = key.replace("loss/", "")
            self.log(f"test/{key_val}", value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Adam with a faster group for the dynamics scalars and a full-run cosine.

        rho/kappa reach the likelihood through a 60-step recursion, so their
        gradients are tiny next to the emission heads. They get their own group
        (the same trick as the CG-NSDE v2 generator's 2e-2 dynamics lr).
        """
        fast_names = {"sde_model.rho_raw", "sde_model.kappa_raw", "sde_model.raw_nu",
                      "sde_model.vol_coupling",
                      "sde_model.regime_emb.weight", "prior_mu", "prior_logvar",
                      "regime_graph_emb", "regime_log_vol", "regime_log_vol_std"}
        fast, slow = [], []
        for name, param in self.named_parameters():
            (fast if name in fast_names else slow).append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": slow, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": fast, "lr": self.lr * 10.0, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_epochs
        )
        
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
                    lv = self.sde_model.log_vol(x)
                    if self.vol_dyn:
                        scenarios.append(self.sde_model.emission_sample_seq(zs_i, regimes, lv)[i % batch_size])
                    else:
                        scenarios.append(self.sde_model.emission_sample(zs_i, regimes, lv)[i % batch_size])
                else:
                    scenarios.append(r_mean[i % batch_size])  # (T, N)

        return torch.stack(scenarios)

    @torch.no_grad()
    def fit_generation_priors(self, batches, device=None) -> None:
        """Fit the generative prior per regime from the training windows.

        Empirical Bayes: the prior p(z0 | regime) is set to the moments of the
        aggregate posterior q(z0 | x) over that regime's windows, and the graph
        context and volatility level to their per-regime means. Sampling from
        these reproduces what the decoder was trained on, without any access to
        the window being generated.
        """
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        mus = {0: [], 1: []}
        graphs = {0: [], 1: []}
        vols = {0: [], 1: []}
        for batch in batches:
            x = batch["returns"].to(device)
            adj = batch["adj_matrix"].to(device)
            node_feats = batch["node_features"].to(device)
            regimes = batch["regimes"].to(device).long().clamp(0, 1)
            mu, _, _ = self.sde_model.encode_stats(x)
            if self.use_gat:
                embs, _ = self.gat(node_feats, adj)
            else:
                embs = self.gat_fallback(node_feats)
            graph = embs.mean(dim=1)
            vol = self.sde_model.log_vol(x)
            for r in (0, 1):
                sel = regimes == r
                if sel.any():
                    mus[r].append(mu[sel].cpu())
                    graphs[r].append(graph[sel].cpu())
                    vols[r].append(vol[sel].cpu())

        with torch.no_grad():
            for r in (0, 1):
                if not mus[r]:
                    print(f"Warning: no regime-{r} windows; prior left at its initial value")
                    continue
                m = torch.cat(mus[r])
                self.prior_mu.data[r] = m.mean(0).to(self.prior_mu.device)
                var = m.var(0, unbiased=False).clamp_min(1e-4)
                self.prior_logvar.data[r] = var.log().to(self.prior_logvar.device)
                self.regime_graph_emb.data[r] = torch.cat(graphs[r]).mean(0).to(self.regime_graph_emb.device)
                self.regime_log_vol.data[r] = torch.cat(vols[r]).mean().to(self.regime_log_vol.device)
                self.regime_log_vol_std.data[r] = torch.cat(vols[r]).std(unbiased=False).clamp_min(
                    1e-3).to(self.regime_log_vol_std.device)
                print(f"Prior regime {r}: {len(m)} windows, "
                      f"log_vol {self.regime_log_vol[r].item():+.3f} "
                      f"(std {self.regime_log_vol_std[r].item():.3f}), "
                      f"prior std {var.sqrt().mean().item():.3f}")
        self.train(was_training)

    @torch.no_grad()
    def sample_regime_scenarios(self, regimes, device=None, chunk: int = 128) -> torch.Tensor:
        """Generate one scenario per requested regime, using only that regime.

        This is the generative entry point: no returns, adjacency, node features
        or posterior of the target window are available here.
        """
        self.eval()
        device = device or next(self.parameters()).device
        r_all = torch.as_tensor(regimes, dtype=torch.long, device=device).reshape(-1).clamp(0, 1)
        outs = []
        for start in range(0, len(r_all), chunk):
            r = r_all[start:start + chunk]
            z0 = self.prior_mu[r] + torch.randn(
                len(r), self.sde_model.latent_dim, device=device
            ) * torch.exp(0.5 * self.prior_logvar[r])
            graph_emb = self.regime_graph_emb[r]
            # Per-window volatility level, drawn from the regime's own log-vol
            # distribution: real windows vary in volatility, so the generator must
            # too, otherwise every draw sits at the regime mean and is trivially
            # separable from real data.
            log_vol = self.regime_log_vol[r] + self.regime_log_vol_std[r] * torch.randn(
                len(r), device=device)
            _, zs = self.sde_model.decode_from_z0(graph_emb, z0)
            if self._use_nll:
                if self.vol_dyn:
                    outs.append(self.sde_model.emission_sample_seq(zs, r, log_vol))
                else:
                    outs.append(self.sde_model.emission_sample(zs, r, log_vol))
            else:
                outs.append(self.sde_model.decoder(zs))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def sample_prior_scenarios(
        self,
        adj: torch.Tensor,
        node_feats: torch.Tensor,
        regimes: torch.Tensor,
        x: torch.Tensor | None = None,
        z0_source: str = "aggregate_posterior",
    ) -> torch.Tensor:
        """Generate one scenario per input row, conditioned on the row's regime.

        Graph context comes from the real row (same-period conditioning). z0 is
        drawn from either the learned regime prior p(z0 | regime) or the
        aggregate posterior q(z0 | x), which is the distribution the decoder
        was actually trained on. Returns (B, T, N).
        """
        self.eval()
        if self.use_gat:
            node_embs, _ = self.gat(node_feats, adj)
        else:
            node_embs = self.gat_fallback(node_feats)
        graph_emb = node_embs.mean(dim=1)
        r = regimes.long().clamp(0, 1)
        if z0_source == "aggregate_posterior" and x is not None:
            mu, logvar, _ = self.sde_model.encode_stats(x)
            z0 = self.sde_model.sample_z0(mu, logvar)
        elif z0_source == "posterior" and x is not None:
            mu, logvar, _ = self.sde_model.encode_stats(x)
            z0 = mu
        else:
            eps = torch.randn(graph_emb.shape[0], self.sde_model.latent_dim, device=graph_emb.device)
            z0 = self.prior_mu[r] + eps * torch.exp(0.5 * self.prior_logvar[r])
        r_mean, zs = self.sde_model.decode_from_z0(graph_emb, z0)
        if self._use_nll:
            # 7.3: generation SAMPLES the emission, never the mean
            lv = self.sde_model.log_vol(x) if x is not None else None
            if self.vol_dyn:
                return self.sde_model.emission_sample_seq(zs, regimes, lv)
            return self.sde_model.emission_sample(zs, regimes, lv)
        return r_mean
