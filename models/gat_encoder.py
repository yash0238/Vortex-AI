"""Dynamic Graph Attention Encoder for CG-NSDE.

Implements the masterplan D.2 DynamicGATEncoder:
- Input: node_feats (B, N, F), adj_matrix (B, N, N)
- Output: node_embs (B, N, H), learned_adj (B, N, N)
- Two-layer GAT with multi-head attention
- Learned adjacency from attention weights
- Graph embedding pooling for SDE conditioning
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class SpatialTemporalGNN(nn.Module):
    """Legacy spatio-temporal encoder (kept for backward compatibility)."""

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


class DynamicGATEncoder(nn.Module):
    """Dynamic Graph Attention Encoder per masterplan D.2.

    Encodes node features using two GAT layers over a dense-to-sparse
    empirical adjacency graph. The attention weights from the second layer
    are extracted to form a learned adjacency matrix.

    Input:
        node_feats: (B, N, F) — per-node feature vectors
        adj_matrix: (B, N, N) — empirical adjacency (used for edge_index)
    Output:
        node_embs: (B, N, H) — learned node embeddings
        learned_adj: (B, N, N) — attention weights as learned adjacency
    """

    def __init__(
        self,
        in_feats: int = 6,
        hidden: int = 64,
        out_feats: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.hidden = hidden
        self.out_feats = out_feats
        self.heads = heads

        self.gat1 = GATConv(
            in_feats, hidden, heads=heads, dropout=dropout, concat=True
        )
        self.gat2 = GATConv(
            hidden * heads, out_feats, heads=1, dropout=dropout, concat=False
        )
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(out_feats)
        self.act = nn.ELU()

    def forward(
        self, node_feats: torch.Tensor, adj_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_ndim = node_feats.ndim
        T = None

        if orig_ndim == 4:
            B, T, N, F = node_feats.shape
            node_feats = node_feats.reshape(B * T, N, F)
            if adj_matrix.ndim == 3:
                adj_matrix = adj_matrix.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
            elif adj_matrix.ndim == 4:
                adj_matrix = adj_matrix.reshape(B * T, N, N)
        else:
            B, N, F = node_feats.shape

        all_embs = []
        all_attn = []

        for b in range(node_feats.shape[0]):
            x = node_feats[b]
            edge_index, _ = dense_to_sparse(adj_matrix[b])

            x1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x1 = self.act(self.norm1(x1))

            x2, attn2 = self.gat2(x1, edge_index, return_attention_weights=True)
            x2 = self.act(self.norm2(x2))

            attn_matrix = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            ei, aw = attn2
            attn_matrix[ei[0], ei[1]] = aw.squeeze(-1)

            all_embs.append(x2)
            all_attn.append(attn_matrix)

        node_embs_out = torch.stack(all_embs)
        learned_adj_out = torch.stack(all_attn)

        if orig_ndim == 4 and T is not None:
            node_embs_out = node_embs_out.reshape(B, T, N, -1)
            learned_adj_out = learned_adj_out.reshape(B, T, N, N)

        return node_embs_out, learned_adj_out


def pool_graph_embedding(node_embs: torch.Tensor) -> torch.Tensor:
    """Pool node embeddings to graph-level embedding for SDE conditioning.

    Args:
        node_embs: (B, N, H) — node embeddings from DynamicGATEncoder
    Returns:
        graph_emb: (B, H) — graph-level embedding
    """
    return node_embs.mean(dim=1)


def graph_consistency_loss(
    A_learned: torch.Tensor, A_empirical: torch.Tensor
) -> torch.Tensor:
    """Frobenius norm loss between learned and empirical adjacency.

    Args:
        A_learned: (B, N, N) — attention-based learned adjacency
        A_empirical: (B, N, N) — empirical target adjacency
    Returns:
        loss: scalar tensor
    """
    return (A_learned - A_empirical).pow(2).mean()
