"""Optional PyTorch Geometric trainer for graph-level regime prediction."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

try:
	from torch_geometric.data import Data
	from torch_geometric.loader import DataLoader
	from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as error:
	Data = DataLoader = GATConv = global_mean_pool = None
	_PYG_IMPORT_ERROR = error

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"


class GATRegimeModel(nn.Module):
	"""Use each asset's return history as node features."""

	def __init__(self, in_dim: int, num_assets: int, hidden_dim: int = 64,
				 num_heads: int = 4) -> None:
		super().__init__()
		self.num_assets = num_assets
		self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
		self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
		self.regime_head = nn.Linear(hidden_dim, 1)

	def forward(self, data):
		nodes = torch.relu(self.conv1(data.x, data.edge_index))
		nodes = torch.dropout(nodes, p=0.2, train=self.training)
		nodes = torch.relu(self.conv2(nodes, data.edge_index))
		graph_count = int(data.batch.max().item()) + 1
		nodes = nodes.view(graph_count, self.num_assets, -1)
		adjacency_logits = torch.bmm(nodes, nodes.transpose(1, 2))
		pooled = global_mean_pool(nodes.reshape(-1, nodes.shape[-1]), data.batch)
		return adjacency_logits, self.regime_head(pooled).squeeze(1)


def build_edge_index(num_assets: int) -> torch.Tensor:
	rows, columns = [], []
	for source in range(num_assets):
		for target in range(num_assets):
			if source != target:
				rows.append(source)
				columns.append(target)
	return torch.tensor([rows, columns], dtype=torch.long)


def load_data():
	if Data is None:
		raise ImportError(
			"GAT training requires torch-geometric. Install it with "
			"'py -3 -m pip install torch-geometric'."
		) from _PYG_IMPORT_ERROR
	windows = np.load(RAW_DATA_DIR / "windows.npy")
	adjacency = np.load(RAW_DATA_DIR / "adj_matrices.npy")
	regimes = np.load(RAW_DATA_DIR / "window_regimes.npy")
	edge_index = build_edge_index(windows.shape[2])
	return [
		Data(
			x=torch.from_numpy(window.T).float(),
			edge_index=edge_index,
			y_adj=torch.from_numpy(graph).float(),
			y_regime=torch.tensor([label], dtype=torch.float32),
		)
		for window, graph, label in zip(windows, adjacency, regimes)
	]


def train(args: argparse.Namespace) -> None:
	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")
	graphs = load_data()
	indices = np.arange(len(graphs))
	labels = [int(graphs[index].y_regime.item()) for index in indices]
	train_idx, test_idx = train_test_split(
		indices, test_size=args.test_size, stratify=labels, random_state=args.seed
	)
	train_labels = [labels[index] for index in train_idx]
	train_idx, val_idx = train_test_split(
		train_idx,
		test_size=args.validation_size / (1 - args.test_size),
		stratify=train_labels,
		random_state=args.seed,
	)
	loaders = [
		DataLoader([graphs[index] for index in selected], batch_size=args.batch_size, shuffle=shuffle)
		for selected, shuffle in ((train_idx, True), (val_idx, False), (test_idx, False))
	]
	model = GATRegimeModel(
		graphs[0].x.shape[1], graphs[0].x.shape[0], args.hidden_dim, args.heads
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
	)
	best_loss, best_state, stale = float("inf"), None, 0
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	for epoch in range(1, args.epochs + 1):
		model.train()
		for batch in loaders[0]:
			batch = batch.to(device)
			optimizer.zero_grad()
			adjacency_pred, regime_logits = model(batch)
			true_adjacency = batch.y_adj.view(-1, model.num_assets, model.num_assets)
			loss = nn.functional.mse_loss(adjacency_pred, true_adjacency)
			loss = loss + nn.functional.binary_cross_entropy_with_logits(
				regime_logits, batch.y_regime.view(-1)
			)
			loss.backward()
			optimizer.step()
		model.eval()
		validation = 0.0
		with torch.no_grad():
			for batch in loaders[1]:
				batch = batch.to(device)
				adjacency_pred, regime_logits = model(batch)
				true_adjacency = batch.y_adj.view(-1, model.num_assets, model.num_assets)
				validation += (
					nn.functional.mse_loss(adjacency_pred, true_adjacency)
					+ nn.functional.binary_cross_entropy_with_logits(
						regime_logits, batch.y_regime.view(-1)
					)
				).item()
		validation /= len(loaders[1])
		print(f"Epoch {epoch:03d} | validation={validation:.4f}")
		if validation < best_loss:
			best_loss, best_state, stale = validation, copy.deepcopy(model.state_dict()), 0
			torch.save(best_state, CHECKPOINT_DIR / "best_gat_model.pt")
		else:
			stale += 1
			if stale >= args.patience:
				break
	model.load_state_dict(best_state)
	model.eval()
	probabilities, labels = [], []
	with torch.no_grad():
		for batch in loaders[2]:
			labels.extend(batch.y_regime.view(-1).tolist())
			_, logits = model(batch.to(device))
			probabilities.extend(torch.sigmoid(logits).cpu().tolist())
	truth, probabilities = np.asarray(labels), np.asarray(probabilities)
	auc = roc_auc_score(truth, probabilities) if np.unique(truth).size > 1 else float("nan")
	print(f"Test regime ROC-AUC: {auc:.4f}")
	print(f"Test regime F1: {f1_score(truth, probabilities >= 0.5, zero_division=0):.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--hidden-dim", type=int, default=64)
	parser.add_argument("--heads", type=int, default=4)
	parser.add_argument("--patience", type=int, default=10)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--validation-size", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())