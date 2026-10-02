"""
End-to-end training script for VORTEX-AI (CG-NSDE).

Usage:
    python train_vortex.py --config training/config.yaml --device cuda
    python train_vortex.py --epochs 100 --batch_size 16 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from training.vortex_model import VORTEXModel


class VortexDataset(Dataset):
    """
    Dataset for VORTEX-AI training.
    
    Loads preprocessed data and constructs batches with:
    - Windowed returns
    - Node features (6-dimensional)
    - Adjacency matrices (empirical Pearson)
    - Regime labels
    """
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        """Load all preprocessed arrays."""
        self.data_dir = Path(data_dir)
        raw_windows = np.load(self.data_dir / "windows.npy")  # (N, T, stocks) raw log returns
        # Stage 3d: train in z-space. Standardize with train-fit scaler; keep raw for inverse.
        sc_path = self.data_dir / "scaler.npz"
        if sc_path.exists():
            sc = np.load(sc_path)
            self.scaler_mean = sc["mean"].astype(np.float32)
            self.scaler_std = sc["std"].astype(np.float32)
            self.windows_raw = raw_windows.astype(np.float32)
            self.windows = ((raw_windows - self.scaler_mean) / self.scaler_std).astype(np.float32)
            print(f"Standardized windows with {sc_path.name} (z-space train)")
        else:
            self.scaler_mean = None
            self.scaler_std = None
            self.windows_raw = None
            self.windows = raw_windows.astype(np.float32)
            print("Warning: scaler.npz missing, using raw returns")
        # 2.3: canonical corrected labels (v2, last-day labeling ~15% crisis),
        # fallback to v1 if v2 missing.
        regimes_path = data_dir / "window_regimes_v2.npy"
        if not regimes_path.exists():
            regimes_path = data_dir / "window_regimes.npy"
        self.regimes = np.load(regimes_path)  # (N,)
        print(f"Regime labels from: {regimes_path.name}")
        self.adj_matrices = np.load(data_dir / "adj_matrices.npy")  # (N, stocks, stocks)

        # Align to shortest length (data artifacts can drift by one window)
        n = min(len(self.windows), len(self.regimes), len(self.adj_matrices))
        if not (len(self.windows) == len(self.regimes) == len(self.adj_matrices)):
            print(f"Aligning dataset lengths to {n} (windows={len(self.windows)}, regimes={len(self.regimes)}, adj={len(self.adj_matrices)})")
        self.windows = self.windows[:n]
        self.regimes = self.regimes[:n]
        self.adj_matrices = self.adj_matrices[:n]

        # Real node-feature sources (2.1): precomputed per-timestep features,
        # raw volume windows, and ticker order for sector/band lookup.
        # Ticker order is assumed to match the stock axis of windows.npy
        # (both derive from nifty50_close.csv column order).
        from data.node_features import load_ticker_list
        self.tickers = load_ticker_list()
        self.volume_windows = None
        vol_path = data_dir / "volume_windows.npy"
        if vol_path.exists():
            try:
                self.volume_windows = np.load(vol_path)[:n]
            except Exception as e:
                print(f"Warning: could not load volume_windows.npy: {e}")
        self.window_node_features = None
        wnf_path = data_dir / "window_node_features.npy"
        if wnf_path.exists():
            try:
                wnf = np.load(wnf_path)
                if wnf.shape[1:] == (self.windows.shape[1], self.windows.shape[2], 6):
                    self.window_node_features = wnf[:n]
                    print(f"Loaded precomputed window_node_features: {wnf.shape}")
                else:
                    print(f"Warning: window_node_features shape {wnf.shape} incompatible, ignoring")
            except Exception as e:
                print(f"Warning: could not load window_node_features.npy: {e}")

        # Basic validation
        assert len(self.windows) == len(self.regimes) == len(self.adj_matrices)
        
        self.n_samples, self.T, self.n_stocks = self.windows.shape
        print(f"Loaded dataset: {self.n_samples} samples, T={self.T}, N={self.n_stocks}")
        print(f"Crisis samples: {self.regimes.sum()} ({self.regimes.mean():.1%})")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dict with:
        - returns: (T, N) windowed returns
        - adj_matrix: (N, N) adjacency for edge construction
        - node_features: (N, F) node features
        - adj_empirical: (N, N) empirical adjacency target
        - regimes: scalar regime label
        """
        returns = torch.tensor(self.windows[idx], dtype=torch.float32)  # (T, N)
        adj = torch.tensor(self.adj_matrices[idx], dtype=torch.float32)  # (N, N)
        regime = torch.tensor(self.regimes[idx], dtype=torch.long)

        # Real 6-dim node features (2.1): precomputed (T, N, 6) averaged over
        # time, else fallback with real sector/band/volume (no placeholders).
        node_features = self._build_node_features(returns, idx)  # (N, 6)
        
        return {
            "returns": returns,
            "adj_matrix": adj,
            "node_features": node_features,
            # 2.2: adj_empirical shares the tensor with adj_matrix by design.
            # They serve different roles: adj_matrix is used only for edge_index
            # structure in DynamicGATEncoder (gat_encoder.py:136), while
            # adj_empirical is the Frobenius value target. The loss is not
            # trivial: GAT attention rows are softmax-normalized (sum to 1)
            # whereas the empirical Pearson matrix is not, so the model cannot
            # match it exactly. Future (masterplan Block 1): K-NN sparsified
            # graph for edges + dense matrix reserved for the loss target.
            "adj_empirical": adj,
            "regimes": regime,
        }
    
    def _build_node_features(self, returns: torch.Tensor, idx: int | None = None) -> torch.Tensor:
        """
        Build 6-dimensional node features per stock.

        Primary: mean over time of precomputed window_node_features (T, N, 6)
        from data/node_features.compute_node_features_compact:
        [r, |r|, vol_z, sector_norm, band, dist_to_band].
        Fallback (same layout as masterplan Part D.2):
        1. Mean return over window
        2. Absolute mean return (volatility)
        3. Real volume activity (std of volume window) or returns std
        4. Real sector id (normalized) from NIFTY50_SECTORS
        5-6. Distance to per-ticker circuit bands via _band_limit
        """
        from data.node_features import NIFTY50_SECTORS, NUM_SECTORS, _band_limit
        T, N = returns.shape

        if idx is not None and self.window_node_features is not None and idx < len(self.window_node_features):
            wnf = torch.tensor(self.window_node_features[idx], dtype=torch.float32)  # (T, N, 6)
            if wnf.shape == (T, N, 6):
                # NaN-safe mean (volume column can hold NaNs from data gaps)
                return torch.nan_to_num(wnf, nan=0.0).mean(dim=0)  # (N, 6)

        # 1. Mean return
        mean_return = returns.mean(dim=0)  # (N,)

        # 2. Absolute mean return (volatility)
        abs_return = returns.abs().mean(dim=0)  # (N,)

        # 3. Real volume activity (std of volume window) or returns std
        if self.volume_windows is not None and idx is not None and idx < len(self.volume_windows):
            vol = torch.tensor(self.volume_windows[idx], dtype=torch.float32)  # (T, N)
            volume_feat = vol.std(dim=0)  # (N,)
        else:
            volume_feat = returns.std(dim=0)  # (N,)

        # 4. Real sector encoding (normalized id) + 5-6. per-ticker bands
        tickers = self.tickers if len(self.tickers) == N else [f"STOCK_{i}" for i in range(N)]
        sector = torch.tensor([NIFTY50_SECTORS.get(tk, 0) / NUM_SECTORS for tk in tickers], dtype=torch.float32)
        bands = torch.tensor([_band_limit(tk) for tk in tickers], dtype=torch.float32)
        max_return = returns.max(dim=0).values
        min_return = returns.min(dim=0).values
        dist_upper = bands - max_return  # (N,)
        dist_lower = bands + min_return  # (N,)

        # Stack all features: (N, 6)
        features = torch.stack([
            mean_return,
            abs_return,
            volume_feat,
            sector,
            dist_upper,
            dist_lower
        ], dim=1)

        return features

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Map z-space returns back to raw log returns for eval/plots (Stage 3d)."""
        if self.scaler_mean is None:
            return z
        mean = torch.from_numpy(self.scaler_mean).to(z.device, dtype=z.dtype)
        std = torch.from_numpy(self.scaler_std).to(z.device, dtype=z.dtype)
        return z * std + mean


def create_dataloaders(
    dataset: VortexDataset,
    batch_size: int,
    val_size: float = 0.15,
    test_size: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    min_per_class: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset and create dataloaders (Stage 3c: chronological split.npz, train sampler only)."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    n_total = len(dataset)
    regimes = np.asarray(dataset.regimes)
    split_path = Path("data/raw/split.npz")
    if split_path.exists():
        sp = np.load(split_path, allow_pickle=True)
        train_idx = np.asarray(sp["train_idx"])
        val_idx = np.asarray(sp["val_idx"])
        test_idx = np.asarray(sp["test_idx"])
        print(f"Chronological split from {split_path.name}: {sp['scheme'][0]}")
    else:
        indices = np.arange(n_total)
        test_frac = test_size
        val_frac_of_rest = val_size / (1 - test_size)
        train_idx, temp_idx = train_test_split(
            indices, test_size=(test_size + val_size),
            stratify=regimes, random_state=seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_size / (test_size + val_size),
            stratify=regimes[temp_idx], random_state=seed,
        )
        print("Warning: split.npz missing, using stratified random split (has leakage)")
    train_dataset, val_dataset, test_dataset = (
        Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx),
    )
    n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)

    print(f"Split: train={n_train}, val={n_val}, test={n_test}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        print(f"  {name} crisis rate: {regimes[idx].mean():.1%}")

    from torch.utils.data import Sampler

    class StratifiedBatchSampler(Sampler[list[int]]):
        """Yield batches with >= min_per_class samples of each regime.

        Masterplan: each batch must contain at least 4 crisis + 4 normal
        samples so SupCon always has positives and negatives.
        Minority class is cycled (oversampled) to fill batches; drop_last.
        """

        def __init__(self, labels: np.ndarray, indices: np.ndarray, batch_size: int,
                     min_per_class: int = 4, seed: int = 42) -> None:
            self.labels = np.asarray(labels)
            self.pool = np.asarray(indices)
            self.batch_size = batch_size
            self.min_per_class = min_per_class
            self.seed = seed
            self.epoch = 0
            pool_labels = self.labels[self.pool]
            crisis_frac = float(pool_labels.mean())
            n_crisis = max(min_per_class, int(round(batch_size * crisis_frac)))
            n_crisis = min(n_crisis, batch_size - min_per_class)
            self.n_crisis = n_crisis
            self.n_normal = batch_size - n_crisis

        def __iter__(self):
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            self.epoch += 1
            pool = self.pool
            crisis = pool[self.labels[pool] == 1]
            normal = pool[self.labels[pool] == 0]
            crisis = crisis[torch.randperm(len(crisis), generator=g).tolist()]
            normal = normal[torch.randperm(len(normal), generator=g).tolist()]
            n_batches = min(len(crisis) // self.n_crisis, len(normal) // self.n_normal)
            ci, ni = 0, 0
            for _ in range(n_batches):
                batch = np.concatenate([
                    crisis[ci:ci + self.n_crisis], normal[ni:ni + self.n_normal],
                ])
                ci += self.n_crisis
                ni += self.n_normal
                if ci + self.n_crisis > len(crisis):  # cycle minority/majority
                    extra = torch.randperm(len(crisis), generator=g).tolist()
                    crisis = np.concatenate([crisis, crisis[extra]])
                if ni + self.n_normal > len(normal):
                    extra = torch.randperm(len(normal), generator=g).tolist()
                    normal = np.concatenate([normal, normal[extra]])
                perm = torch.randperm(len(batch), generator=g).tolist()
                yield [int(batch[i]) for i in perm]

        def __len__(self) -> int:
            pool_labels = self.labels[self.pool]
            n_crisis = int((pool_labels == 1).sum())
            n_normal = int((pool_labels == 0).sum())
            return min(n_crisis // self.n_crisis, n_normal // self.n_normal)

    # Create dataloaders (train uses stratified batches over the full dataset
    # with a global-index pool; val/test use sequential subsets)
    train_loader = DataLoader(
        dataset,
        batch_sampler=StratifiedBatchSampler(regimes, train_idx, batch_size, min_per_class, seed),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader, test_loader


def train(args: argparse.Namespace):
    """Main training function."""
    # Load config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 2.4: flatten nested cgnsde block (+ loss_weights) so flat lookups below
    # pick up masterplan hyperparameters; explicit flat keys and CLI win.
    _cgnsde = config.get("cgnsde") or {}
    config = {**_cgnsde.get("loss_weights", {}), **_cgnsde, **config}

    # Override with CLI arguments
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "resume"]:
            config[key] = value
    
    # 4.1: ablation overrides (applied after config + CLI merge)
    ablation = config.get("ablation", "none")
    if ablation == "no-supcon":
        config["lam_con"] = 0.0
    elif ablation == "no-circuit":
        config["lam_na"] = 0.0
    use_gat = ablation != "no-gat"
    static_graph = ablation == "static-graph"
    if ablation != "none":
        print(f"Ablation variant: {ablation}")

    # Set seed
    pl.seed_everything(config.get("seed", 42))
    
    # Load dataset
    dataset = VortexDataset(Path("data/raw"))
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.get("batch_size", 32),
        val_size=config.get("validation_size", 0.15),
        test_size=config.get("test_size", 0.2),
        num_workers=config.get("num_workers", 4),
        seed=config.get("seed", 42),
    )
    
    # Initialize model
    model = VORTEXModel(
        n_stocks=config.get("n_stocks", 50),
        T=config.get("T", 60),
        in_feats=config.get("in_feats", 6),
        latent_dim=config.get("latent_dim", 64),
        proj_dim=config.get("proj_dim", 32),
        hidden_dim=config.get("hidden_dim", 64),
        gat_heads=config.get("gat_heads", 4),
        tau=config.get("tau", 0.07),
        lr=config.get("learning_rate", 1e-3),
        lam_rec=config.get("lam_rec", 1.0),
        lam_graph=config.get("lam_graph", 0.1),
        lam_con=config.get("lam_con", 0.1),
        lam_na=config.get("lam_na", 0.05),
        beta_max=config.get("beta_max", 0.01),
        prior_warmup=config.get("prior_warmup", 10),
        lam_var=config.get("lam_var", 0.0),
        emission=config.get("emission", "point"),
        rank_K=config.get("rank_K", 0),
        sde_sigma_max=config.get("sde_sigma_max", 3.0),
        lambda_styl=config.get("lambda_styl", 0.1),
        warmup_epochs=config.get("warmup_epochs", 20),
        use_gat=use_gat,
        static_graph=static_graph,
    )
    
    # Setup wandb logger
    run_id = None
    if config.get("wandb_project"):
        wandb_logger = WandbLogger(
            project=config["wandb_project"],
            entity=config.get("wandb_entity"),
            config=config,
            log_model=True,
        )
    else:
        wandb_logger = None
        print("Warning: wandb logging disabled. Set 'wandb_project' in config to enable.")
        from datetime import datetime
        from pytorch_lightning.loggers import CSVLogger
        run_id = datetime.now().strftime("%Y%m%d-%H%M") + "-" + (config.get("run_tag") or "run")
        wandb_logger = CSVLogger(save_dir="runs", name=run_id)
        print(f"Run dir: runs/{run_id}")
    
    # Setup callbacks
    ckpt_tag = "vortex" if ablation == "none" else f"vortex-abl-{ablation}"
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename=ckpt_tag + "-{epoch:02d}-{val/total:.4f}",
        monitor="val/total",
        mode="min",
        save_top_k=config.get("save_top_k", 3),
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/total",
        patience=config.get("patience", 20),
        mode="min",
        verbose=True,
    )
    print(f"EarlyStopping patience: {config.get('patience', 20)} (best_val monitor val/total)")
    
    # Setup trainer (logger=False when wandb off: env tensorboard/protobuf is broken)
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 200),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        logger=wandb_logger if wandb_logger else False,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        log_every_n_steps=config.get("log_every_n_steps", 10),
        deterministic=True,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting VORTEX-AI (CG-NSDE) Training")
    print("="*80 + "\n")
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    
    # Test
    print("\n" + "="*80)
    print("Testing best model")
    print("="*80 + "\n")
    
    trainer.test(model, test_loader, ckpt_path="best")
    
    # Finish wandb run
    if wandb_logger:
        wandb.finish()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train VORTEX-AI (CG-NSDE)")
    
    parser.add_argument("--config", type=str, default="training/config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Override config parameters
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, help="Maximum epochs")
    parser.add_argument("--accelerator", type=str, choices=["auto", "cpu", "cuda", "mps"],
                       help="Accelerator type")
    parser.add_argument("--devices", type=int, help="Number of devices")
    parser.add_argument("--lam_con", type=float, help="Contrastive loss weight")
    parser.add_argument("--lam_graph", type=float, help="Graph loss weight")
    parser.add_argument("--lam_na", type=float, help="Circuit-filter loss weight")
    parser.add_argument("--beta_max", type=float, help="KL weight ceiling (7.1)")
    parser.add_argument("--prior_warmup", type=int, help="KL ramp epochs (7.1)")
    parser.add_argument("--lam_var", type=float, help="Variance loss weight (7.2)")
    parser.add_argument("--emission", type=str, choices=["point", "hetero", "t", "mt"], help="Emission head (7.3)")
    parser.add_argument("--rank_K", type=int, help="Emission factor rank (7.3b)")
    parser.add_argument("--sde_sigma_max", type=float, help="SDE diffusion bound (7.3)")
    parser.add_argument("--run_tag", type=str, help="Run dir suffix under runs/")
    parser.add_argument("--num_workers", type=int, help="Dataloader workers (0 = safest on Windows)")
    parser.add_argument("--patience", type=int, help="Early-stopping patience in epochs")
    parser.add_argument("--lambda_styl", type=float, help="Stylized-facts loss weight")
    parser.add_argument("--warmup_epochs", type=int, help="Contrastive warmup epochs")
    # 4.1: ablation selector (none = full model)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no-gat", "no-supcon", "static-graph", "no-circuit"],
                        help="Ablation variant to train")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
