"""Train and evaluate the Vortex-AI spatio-temporal graph model."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.gat_encoder import SpatialTemporalGNN
from training.losses import joint_loss

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Load preprocessed windows, empirical graphs, and regime labels."""
	windows = np.load(RAW_DATA_DIR / "windows.npy")
	regimes = np.load(RAW_DATA_DIR / "window_regimes.npy")
	adjacency = np.load(RAW_DATA_DIR / "adj_matrices.npy")
	if len(windows) != len(regimes) or len(windows) != len(adjacency):
		raise ValueError("windows, window_regimes, and adj_matrices must have equal length")
	return (
		torch.from_numpy(windows).float(),
		torch.from_numpy(adjacency).float(),
		torch.from_numpy(regimes).float(),
	)


def run_epoch(model, loader, optimizer, adjacency_weight: float,
			  device: torch.device, binarize_adjacency: bool) -> float:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	for windows, adjacency, regimes in loader:
		windows = windows.to(device, non_blocking=True)
		adjacency = adjacency.to(device, non_blocking=True)
		regimes = regimes.to(device, non_blocking=True)
		if training:
			optimizer.zero_grad()
		adjacency_pred, regime_logits = model(windows)
		loss = joint_loss(
			adjacency_pred, adjacency, regime_logits, regimes, adjacency_weight,
			binarize_adjacency,
		)
		if training:
			loss.backward()
			optimizer.step()
		total_loss += loss.item() * windows.size(0)
	return total_loss / len(loader.dataset)


def evaluate(model, loader, adjacency_weight: float, device: torch.device) -> tuple[float, float, float]:
	model.eval()
	probabilities, labels = [], []
	adjacency_errors = []
	with torch.no_grad():
		for windows, adjacency, regimes in loader:
			windows = windows.to(device, non_blocking=True)
			adjacency = adjacency.to(device, non_blocking=True)
			regimes = regimes.to(device, non_blocking=True)
			adjacency_pred, regime_logits = model(windows)
			adjacency_errors.append((adjacency_pred - adjacency).square().mean().item())
			probabilities.extend(torch.sigmoid(regime_logits).tolist())
			labels.extend(regimes.tolist())
	probs = np.asarray(probabilities)
	truth = np.asarray(labels, dtype=int)
	auc = roc_auc_score(truth, probs) if np.unique(truth).size > 1 else float("nan")
	f1 = f1_score(truth, probs >= 0.5, zero_division=0)
	return float(np.mean(adjacency_errors)), float(auc), float(f1)


def train(args: argparse.Namespace) -> None:
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
	device = torch.device(args.device)
	print(f"Using device: {device}")
	windows, adjacency, regimes = load_data()
	indices = np.arange(len(windows))
	train_indices, test_indices = train_test_split(
		indices, test_size=args.test_size, stratify=regimes.numpy(), random_state=args.seed
	)
	train_indices, validation_indices = train_test_split(
		train_indices,
		test_size=args.validation_size / (1 - args.test_size),
		stratify=regimes[train_indices].numpy(),
		random_state=args.seed,
	)

	def make_loader(selected, shuffle):
		dataset = TensorDataset(windows[selected], adjacency[selected], regimes[selected])
		return DataLoader(
			dataset,
			batch_size=args.batch_size,
			shuffle=shuffle,
			pin_memory=device.type == "cuda",
		)

	train_loader = make_loader(train_indices, True)
	validation_loader = make_loader(validation_indices, False)
	test_loader = make_loader(test_indices, False)
	model = SpatialTemporalGNN(
		windows.shape[2], args.hidden_dim, args.lstm_layers, args.dropout
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=5
	)
	best_loss, best_state, stale_epochs = float("inf"), None, 0
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

	for epoch in range(1, args.epochs + 1):
		train_loss = run_epoch(
			model, train_loader, optimizer, args.adjacency_weight, device, args.binarize_adjacency
		)
		with torch.no_grad():
			validation_loss = run_epoch(
				model, validation_loader, None, args.adjacency_weight, device,
				args.binarize_adjacency,
			)
		scheduler.step(validation_loss)
		print(f"Epoch {epoch:03d} | train={train_loss:.4f} | validation={validation_loss:.4f}")
		if validation_loss < best_loss:
			best_loss = validation_loss
			best_state = copy.deepcopy(model.state_dict())
			torch.save(best_state, CHECKPOINT_DIR / "best_model.pt")
			stale_epochs = 0
		else:
			stale_epochs += 1
			if stale_epochs >= args.patience:
				break

	if best_state is None:
		raise RuntimeError("Training produced no checkpoint")
	model.load_state_dict(best_state)
	adjacency_mse, auc, f1 = evaluate(model, test_loader, args.adjacency_weight, device)
	print(f"Test adjacency MSE: {adjacency_mse:.4f}")
	print(f"Test regime ROC-AUC: {auc:.4f}")
	print(f"Test regime F1: {f1:.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--hidden-dim", type=int, default=64)
	parser.add_argument("--lstm-layers", type=int, default=2)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--adjacency-weight", type=float, default=1.0)
	parser.add_argument(
		"--binarize-adj", "--binarize_adj", dest="binarize_adjacency",
		action="store_true", help="Train graph reconstruction against binary edge targets.",
	)
	parser.add_argument("--patience", type=int, default=10)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--validation-size", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--device",
		choices=("auto", "cuda", "cpu"),
		default="auto",
		help="Training device; auto selects CUDA when available.",
	)
	args = parser.parse_args()
	if args.device == "auto":
		args.device = "cuda" if torch.cuda.is_available() else "cpu"
	return args


if __name__ == "__main__":
	train(parse_args())
