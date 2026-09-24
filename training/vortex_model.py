"""PyTorch Lightning module for VORTEX-AI CG-NSDE end-to-end training.

Integrates all components as specified in masterplan Part D.7.
"""

from __future__ import annotations

import torch
from torch import nn
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
        lam_na: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.gat = DynamicGATEncoder(
            in_feats=in_feats,
            hidden=hidden_dim,
            out_feats=latent_dim,
            heads=gat_heads,
            dropout=0.1
        )
        
        self.sde_model = LatentSDEModel(
            n_stocks=n_stocks,
            latent_dim=latent_dim,
            T=T
        )
        
        self.proj_head = ProjectionHead(
            latent_dim=latent_dim,
            proj_dim=proj_dim
        )
        
        self.supcon = SupConLoss(temperature=tau)
        
        # Loss weights
        self.lam_rec = lam_rec
        self.lam_graph = lam_graph
        self.lam_con = lam_con
        self.lam_na = lam_na
        self.lr = lr

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
        # 1. Dynamic GAT: encode graph structure
        node_embs, A_learned = self.gat(node_feats, adj)  # (B, N, H), (B, N, N)
        
        # 2. Pool node embeddings to graph-level representation
        graph_emb = node_embs.mean(dim=1)  # (B, H)
        
        # 3. Latent SDE: generate synthetic paths
        r_hat, zs = self.sde_model(x, graph_emb)  # (B, T, N), (B, T, latent_dim)
        
        # 4. Project to contrastive space
        z_proj = self.proj_head(zs)  # (B, proj_dim)
        
        return r_hat, A_learned, z_proj

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with complete loss assembly."""
        x = batch["returns"]  # (B, T, N)
        adj = batch["adj_matrix"]  # (B, N, N)
        node_feats = batch["node_features"]  # (B, N, F)
        A_emp = batch["adj_empirical"]  # (B, N, N)
        regimes = batch["regimes"]  # (B,)
        
        # Forward pass
        r_hat, A_learned, z_proj = self(x, adj, node_feats)
        
        # Compute total loss
        loss, components = total_loss(
            r_real=x,
            r_hat=r_hat,
            A_learned=A_learned,
            A_emp=A_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lam_rec=self.lam_rec,
            lam_graph=self.lam_graph,
            lam_con=self.lam_con,
            lam_na=self.lam_na,
        )
        
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
        
        # Compute total loss
        loss, components = total_loss(
            r_real=x,
            r_hat=r_hat,
            A_learned=A_learned,
            A_emp=A_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lam_rec=self.lam_rec,
            lam_graph=self.lam_graph,
            lam_con=self.lam_con,
            lam_na=self.lam_na,
        )
        
        # Log all components
        for key, value in components.items():
            key_val = key.replace("loss/", "")
            self.log(f"val/{key_val}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with cosine annealing schedule."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
            # Get graph embedding
            node_embs, _ = self.gat(node_feats, adj)
            graph_emb = node_embs.mean(dim=1)
            
            # Generate multiple scenarios
            for _ in range(n_scenarios):
                r_hat, _ = self.sde_model(x, graph_emb)
                scenarios.append(r_hat[0])  # Take first batch element
        
        return torch.stack(scenarios)
