"""Train and evaluate the Vortex-AI spatio-temporal graph model."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.gat_encoder import DynamicGATEncoder, SpatialTemporalGNN, pool_graph_embedding
from models.neural_sde import LatentSDEModel
from models.contrastive import ProjectionHead, SupConLoss
from training.losses import joint_loss, total_loss

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
    parser.add_argument(
        "--mode",
        choices=("baseline", "cgnsde"),
        default="baseline",
        help="Training mode: baseline LSTM or full CG-NSDE Lightning model.",
    )
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
    # CG-NSDE specific arguments
    parser.add_argument("--n-stocks", type=int, default=47)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--in-feats", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--sde-hidden-dim", type=int, default=128)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lam-rec", type=float, default=1.0)
    parser.add_argument("--lam-graph", type=float, default=0.1)
    parser.add_argument("--lam-con", type=float, default=0.1)
    parser.add_argument("--lam-na", type=float, default=0.05)
    parser.add_argument("--lambda-styl", type=float, default=0.1)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "cgnsde":
        cfg = CGNSDEConfig(
            n_stocks=args.n_stocks,
            T=args.window_size,
            latent_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            in_feats=args.in_feats,
            gat_heads=args.heads,
            gat_dropout=args.dropout,
            tau=args.tau,
            lr=args.learning_rate,
            sde_hidden_dim=args.sde_hidden_dim,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
            gradient_clip=args.gradient_clip,
            loss_weights={
                "lam_rec": args.lam_rec,
                "lam_graph": args.lam_graph,
                "lam_con": args.lam_con,
                "lam_na": args.lam_na,
            },
            lambda_styl=args.lambda_styl,
        )
        train_cgnsde(cfg, args)
    else:
        train(args)


# ============================================================================
# CG-NSDE Lightning Module (Phase 3+)
# Implements masterplan Part D.7: VORTEXModel(pl.LightningModule)
# ============================================================================

@dataclass
class CGNSDEConfig:
    """Configuration for the CG-NSDE Lightning model."""

    n_stocks: int = 47
    T: int = 60
    latent_dim: int = 64
    proj_dim: int = 32
    in_feats: int = 6
    gat_heads: int = 4
    gat_dropout: float = 0.1
    tau: float = 0.07
    lr: float = 1e-3
    sde_hidden_dim: int = 128
    max_epochs: int = 200
    batch_size: int = 32
    test_size: float = 0.2
    validation_size: float = 0.15
    seed: int = 42
    gradient_clip: float = 1.0
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "lam_rec": 1.0,
        "lam_graph": 0.1,
        "lam_con": 0.1,
        "lam_na": 0.05,
    })
    lambda_styl: float = 0.1


def load_cgnsde_data(data_dir: Path | None = None) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Load all 5 data tensors needed by the VORTEXModel.

    Returns:
        windows: (N, T, n_stocks) — return windows
        adj: (N, n_stocks, n_stocks) — empirical adjacency (for GAT edge_index)
        node_feats: (N, n_stocks, F) — last-timestep node features
        a_emp: (N, n_stocks, n_stocks) — empirical adjacency targets
        regimes: (N,) — per-window regime labels (0=normal, 1=crisis)
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    windows = np.load(data_dir / "windows.npy")
    regimes = np.load(data_dir / "window_regimes_v2.npy")
    adj_matrices = np.load(data_dir / "adj_matrices.npy")
    node_feats_full = np.load(data_dir / "window_node_features.npy")

    n = min(len(windows), len(regimes), len(adj_matrices), len(node_feats_full))
    windows = windows[:n]
    regimes = regimes[:n]
    adj_matrices = adj_matrices[:n]
    node_feats_full = node_feats_full[:n]

    node_feats = node_feats_full[:, -1, :, :]

    windows_t = torch.from_numpy(windows).float()
    adj_t = torch.from_numpy(adj_matrices).float()
    node_feats_t = torch.from_numpy(node_feats).float()
    regimes_t = torch.from_numpy(regimes).float()

    node_feats_t = torch.nan_to_num(node_feats_t, nan=0.0, posinf=0.0, neginf=0.0)
    adj_t = torch.nan_to_num(adj_t, nan=0.0, posinf=1.0, neginf=0.0)

    return windows_t, adj_t, node_feats_t, adj_t, regimes_t


class VORTEXDataModule(pl.LightningDataModule):
    """Lightning data module for CG-NSDE training data."""

    def __init__(self, cfg: CGNSDEConfig, data_dir: Path | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir

    def setup(self, stage: str | None = None) -> None:
        windows, adj, node_feats, a_emp, regimes = load_cgnsde_data(self.data_dir)
        indices = np.arange(len(windows))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.cfg.test_size,
            stratify=regimes.numpy(),
            random_state=self.cfg.seed,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.cfg.validation_size / (1 - self.cfg.test_size),
            stratify=regimes[train_idx].numpy(),
            random_state=self.cfg.seed,
        )

        self.train_data = TensorDataset(
            windows[train_idx], adj[train_idx], node_feats[train_idx],
            a_emp[train_idx], regimes[train_idx],
        )
        self.val_data = TensorDataset(
            windows[val_idx], adj[val_idx], node_feats[val_idx],
            a_emp[val_idx], regimes[val_idx],
        )
        self.test_data = TensorDataset(
            windows[test_idx], adj[test_idx], node_feats[test_idx],
            a_emp[test_idx], regimes[test_idx],
        )
        self._input_dims = windows.shape

    def _loader(self, data: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            data,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.trainer and self.trainer.accelerator == "gpu",
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_data, shuffle=False)


class LegacyTupleVORTEXModel(pl.LightningModule):
    """Full CG-NSDE model: GAT encoder + Neural SDE + SupCon regime head.

    .. deprecated::
        Divergent duplicate of the canonical
        :class:`training.vortex_model.VORTEXModel` (dict-batch API,
        ``train/*`` logs). Kept only for the ``--mode cgnsde`` CLI path.
        Uses a tuple-batch API and ``train_loss``/``val_loss`` log names,
        so its checkpoints are NOT interchangeable with the canonical model.
        Do not use for new code.

    Masterplan Part D.7.

    Architecture:
        x (B, T, N) ──────► LatentSDEModel ──► r_hat (B, T, N), zs (B, T, latent)
        node_feats (B, N, F) ──► DynamicGATEncoder ──► node_embs (B, N, H), A_learned (B, N, N)
        A_learned ──► graph_consistency_loss vs A_emp
        zs ──► ProjectionHead ──► z_proj (B, proj_dim)
        z_proj, regimes ──► SupConLoss
    """

    def __init__(self, cfg: CGNSDEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gat = DynamicGATEncoder(
            in_feats=cfg.in_feats,
            hidden=cfg.latent_dim,
            out_feats=cfg.latent_dim,
            heads=cfg.gat_heads,
            dropout=cfg.gat_dropout,
        )
        self.sde_model = LatentSDEModel(
            n_stocks=cfg.n_stocks,
            latent_dim=cfg.latent_dim,
            T=cfg.T,
            sde_hidden_dim=cfg.sde_hidden_dim,
        )
        self.proj_head = ProjectionHead(cfg.latent_dim, cfg.proj_dim)
        self.supcon = SupConLoss(temperature=cfg.tau)
        self.save_hyperparameters()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, node_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the full model.

        Args:
            x: (B, T, n_stocks) — return windows
            adj: (B, n_stocks, n_stocks) — empirical adjacency (for edge_index)
            node_feats: (B, n_stocks, F) — node features for GAT
        Returns:
            r_hat: (B, T, n_stocks) — decoded return estimates
            a_learned: (B, n_stocks, n_stocks) — learned adjacency from attention
            z_proj: (B, proj_dim) — projected latent for SupCon
        """
        node_embs, a_learned = self.gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)
        r_hat, zs = self.sde_model(x, graph_emb)
        z_proj = self.proj_head(zs)
        return r_hat, a_learned, z_proj

    def _compute_loss(
        self, batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        x, adj, node_feats, a_emp, regimes = batch
        r_hat, a_learned, z_proj = self(x, adj, node_feats)
        loss, _ = total_loss(
            r_real=x,
            r_hat=r_hat,
            a_learned=a_learned,
            a_empirical=a_emp,
            z_proj=z_proj,
            regime_labels=regimes,
            supcon_loss_fn=self.supcon,
            lambda_styl=self.cfg.lambda_styl,
            **self.cfg.loss_weights,
        )
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]


def train_cgnsde(
    cfg: CGNSDEConfig | None = None,
    args: argparse.Namespace | None = None,
) -> LegacyTupleVORTEXModel:
    """Train the CG-NSDE Lightning model.

    Uses the deprecated :class:`LegacyTupleVORTEXModel` tuple-batch path.
    For the canonical dict-batch model, use train_vortex.py instead.

    Args:
        cfg: Configuration dataclass. If None, uses CGNSDEConfig defaults.
        args: Optional CLI namespace (for checkpoint path override).

    Returns:
        Trained VORTEXModel.
    """
    if cfg is None:
        cfg = CGNSDEConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = LegacyTupleVORTEXModel(cfg)
    dm = VORTEXDataModule(cfg)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="best_cgnsde_tuple_model",
        save_top_one=True,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        gradient_clip_val=cfg.gradient_clip,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_checkpointing=True,
    )

    trainer.fit(model, dm)

    if trainer.checkpoint_callback is not None:
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint: {best_path}")

    return model
