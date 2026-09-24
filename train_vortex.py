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
        self.windows = np.load(data_dir / "windows.npy")  # (N, T, stocks)
        self.regimes = np.load(data_dir / "window_regimes.npy")  # (N,)
        self.adj_matrices = np.load(data_dir / "adj_matrices.npy")  # (N, stocks, stocks)
        
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
        
        # Construct node features (6-dimensional as per masterplan)
        node_features = self._build_node_features(returns)  # (N, 6)
        
        return {
            "returns": returns,
            "adj_matrix": adj,
            "node_features": node_features,
            "adj_empirical": adj,  # Use same adjacency as target
            "regimes": regime,
        }
    
    def _build_node_features(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Build 6-dimensional node features per stock.
        
        Features (as per masterplan Part D.2):
        1. Mean return over window
        2. Absolute mean return (volatility proxy)
        3. Normalized volume (placeholder: use std as proxy)
        4. Sector one-hot (placeholder: uniform)
        5. Distance to upper circuit band
        6. Distance to lower circuit band
        """
        T, N = returns.shape
        
        # 1. Mean return
        mean_return = returns.mean(dim=0)  # (N,)
        
        # 2. Absolute mean return (volatility)
        abs_return = returns.abs().mean(dim=0)  # (N,)
        
        # 3. Volume proxy (use std dev as placeholder)
        volume_proxy = returns.std(dim=0)  # (N,)
        
        # 4. Sector encoding (placeholder: uniform for now)
        sector = torch.zeros(N)  # (N,)
        
        # 5-6. Distance to circuit bands (assume 10% default)
        circuit_band = 0.10
        max_return = returns.max(dim=0).values
        min_return = returns.min(dim=0).values
        dist_upper = circuit_band - max_return  # (N,)
        dist_lower = circuit_band + min_return  # (N,)
        
        # Stack all features: (N, 6)
        features = torch.stack([
            mean_return,
            abs_return,
            volume_proxy,
            sector,
            dist_upper,
            dist_lower
        ], dim=1)
        
        return features


def create_dataloaders(
    dataset: VortexDataset,
    batch_size: int,
    val_size: float = 0.15,
    test_size: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset and create dataloaders."""
    n_total = len(dataset)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    # Override with CLI arguments
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "resume"]:
            config[key] = value
    
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
        lam_na=config.get("lam_na", 0.0),
    )
    
    # Setup wandb logger
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
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="vortex-{epoch:02d}-{val/total:.4f}",
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
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 200),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        logger=wandb_logger,
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
