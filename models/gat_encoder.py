"""Dynamic Graph Attention Network encoder for VORTEX-AI CG-NSDE framework."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class DynamicGATEncoder(nn.Module):
    """
    Dynamic GAT Encoder as specified in VORTEX-AI masterplan.
    
    Input:  node_feats  (B, N, F) - node features per stock
            adj_matrix  (B, N, N) - empirical adjacency (used for edge_index)
    Output: node_embs   (B, N, H) - node embeddings
            learned_adj (B, N, N) - attention weights as learned adjacency
    """
    def __init__(self, in_feats: int = 6, hidden: int = 64, out_feats: int = 64, 
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads, out_feats, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(out_feats)
        self.act = nn.ELU()

    def forward(self, node_feats: torch.Tensor, adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats: (B, N, F) node features
            adj_matrix: (B, N, N) empirical adjacency matrix
        
        Returns:
            node_embs: (B, N, H) node embeddings
            learned_adj: (B, N, N) learned attention-based adjacency
        """
        B, N, F = node_feats.shape
        all_embs, all_attn = [], []

        for b in range(B):
            x = node_feats[b]  # (N, F)
            edge_index, _ = dense_to_sparse(adj_matrix[b])
            
            # First GAT layer with multi-head attention
            x1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x1 = self.act(self.norm1(x1))
            
            # Second GAT layer with single head
            x2, attn2 = self.gat2(x1, edge_index, return_attention_weights=True)
            x2 = self.act(self.norm2(x2))

            # Construct learned adjacency matrix from attention weights
            attn_matrix = torch.zeros(N, N, device=x.device)
            ei, aw = attn2
            attn_matrix[ei[0], ei[1]] = aw.squeeze(-1)

            all_embs.append(x2)
            all_attn.append(attn_matrix)

        return torch.stack(all_embs), torch.stack(all_attn)


# Legacy spatio-temporal baseline - kept for backward compatibility
class SpatialTemporalGNN(nn.Module):
    """Encode per-asset return histories and predict graph and regime targets."""

    def __init__(
        self,
        num_assets: int,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.temporal_encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.edge_score = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, time, assets), got {tuple(x.shape)}")
        batch_size, time_steps, num_assets = x.shape
        if num_assets != self.num_assets:
            raise ValueError(f"Expected {self.num_assets} assets, got {num_assets}")

        sequences = x.transpose(1, 2).reshape(batch_size * num_assets, time_steps, 1)
        encoded, _ = self.temporal_encoder(sequences)
        asset_features = encoded[:, -1].reshape(batch_size, num_assets, self.hidden_dim)

        left = asset_features.unsqueeze(2).expand(-1, -1, num_assets, -1)
        right = asset_features.unsqueeze(1).expand(-1, num_assets, -1, -1)
        adjacency_logits = self.edge_score(left, right).squeeze(-1)
        adjacency_logits = (adjacency_logits + adjacency_logits.transpose(1, 2)) / 2
        adjacency_probabilities = torch.sigmoid(adjacency_logits)
        adjacency_probabilities = adjacency_probabilities * (
            1 - torch.eye(num_assets, device=x.device, dtype=x.dtype).unsqueeze(0)
        )

        pooled = torch.cat(
            [asset_features.mean(dim=1), asset_features.max(dim=1).values], dim=1
        )
        return adjacency_probabilities, self.regime_head(pooled).squeeze(1)
