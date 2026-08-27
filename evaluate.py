"""Evaluate a trained GAT checkpoint on the deterministic held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from training.gat_trainer import GATRegimeModel, load_data

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "checkpoints" / "best_gat_model.pt"


def evaluate(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    graphs = load_data()
    labels = np.asarray([int(graph.y_regime.item()) for graph in graphs])
    indices = np.arange(len(graphs))
    _, test_indices = train_test_split(
        indices, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    loader = DataLoader(
        [graphs[index] for index in test_indices], batch_size=args.batch_size, shuffle=False
    )
    model = GATRegimeModel(
        graphs[0].x.shape[1], graphs[0].x.shape[0], args.hidden_dim, args.heads
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    adjacency_errors, probabilities, truth = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            adjacency_pred, regime_logits = model(batch)
            target = batch.y_adj.view(-1, model.num_assets, model.num_assets)
            adjacency_errors.append((adjacency_pred - target).square().mean().item())
            probabilities.extend(torch.sigmoid(regime_logits).cpu().tolist())
            truth.extend(batch.y_regime.view(-1).cpu().tolist())
    probabilities, truth = np.asarray(probabilities), np.asarray(truth, dtype=int)
    auc = roc_auc_score(truth, probabilities) if np.unique(truth).size > 1 else float("nan")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Adjacency MSE: {np.mean(adjacency_errors):.4f}")
    print(f"Regime ROC-AUC: {auc:.4f}")
    print(f"Regime F1: {f1_score(truth, probabilities >= 0.5, zero_division=0):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
