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
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from training.vortex_model import VORTEXModel
from training.data_validation import validate_training_data

# Bumped whenever the training/eval contract changes. The Kaggle notebook asserts
# this marker before training: a stale dataset silently re-ran an old experiment
# once, which cost a full hour of GPU quota.
CODE_VERSION = "v19-regime-only-balanced"


def _drop_padded_leading_windows(windows: np.ndarray, adj: np.ndarray):
    """Drop leading windows whose first timestep is padding (all zeros).

    `windows.npy` carries one such window at the front: its first row is zeros
    because the first log-return is undefined. Regimes and node features start
    at the second window, so keeping it shifts every sample by one day.
    """
    k = 0
    while k < len(windows) - 1 and np.all(windows[k, 0] == 0.0):
        k += 1
    if k:
        print(f"Dropped {k} padded leading window(s) from windows.npy / adj_matrices.npy")
        windows, adj = windows[k:], adj[k:]
    return windows, adj


def _feature_offset(windows: np.ndarray, feats: np.ndarray, max_shift: int = 4):
    """Find the shift that pairs window i with node-feature window i.

    Node-feature channel 0 is the return itself, so the correct shift is the
    one that reproduces `windows` most closely. Returns (offset, mean_abs_err).
    """
    best = (0, float("inf"))
    for cand in range(max_shift):
        if cand >= len(feats):
            break
        n = min(len(windows), len(feats) - cand)
        if n <= 0:
            break
        err = float(np.mean(np.abs(windows[:n] - feats[cand:cand + n, :, :, 0])))
        if err < best[1]:
            best = (cand, err)
    return best


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
        self.adj_matrices = np.load(data_dir / "adj_matrices.npy")  # (N, stocks, stocks)
        # Padding must be detected on RAW returns: standardising turns the zero
        # fill into (0 - mean)/std, which is no longer zero.
        raw_windows, self.adj_matrices = _drop_padded_leading_windows(
            raw_windows, self.adj_matrices
        )
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
        # derive them from daily labels when the generated v2 file is missing.
        regimes_path = data_dir / "window_regimes_v2.npy"
        if regimes_path.exists():
            self.regimes = np.load(regimes_path)
            print(f"Regime labels from: {regimes_path.name}")
        else:
            daily_regimes_path = data_dir / "regimes.npy"
            if daily_regimes_path.exists():
                daily_regimes = np.load(daily_regimes_path)
                first_window_end = self.windows.shape[1] - 1
                self.regimes = daily_regimes[
                    first_window_end:first_window_end + len(self.windows)
                ]
                print(f"Derived final-day regime labels from: {daily_regimes_path.name}")
            else:
                regimes_path = data_dir / "window_regimes.npy"
                self.regimes = np.load(regimes_path)
                print(f"Regime labels from legacy {regimes_path.name}")

        # Align to shortest length, then align node features by content: the
        # old code truncated windows.npy at the tail while node features started
        # one window later, so every sample paired returns[i] with features[i+1].
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
        self.feature_offset = 0
        vol_raw = None
        vol_path = data_dir / "volume_windows.npy"
        if vol_path.exists():
            try:
                vol_raw = np.load(vol_path)
            except Exception as e:
                print(f"Warning: could not load volume_windows.npy: {e}")
        self.window_node_features = None
        wnf_path = data_dir / "window_node_features.npy"
        if wnf_path.exists():
            try:
                wnf = np.load(wnf_path)
                if wnf.shape[1:] == (self.windows.shape[1], self.windows.shape[2], 6):
                    off, err = _feature_offset(self.windows, wnf)
                    if off:
                        print(f"Node features shifted by {off} window(s) to align (err {err:.2e})")
                    if err > 1e-2:
                        raise ValueError(
                            f"window_node_features does not align with returns "
                            f"(best shift {off}, mean abs err {err:.3f})"
                        )
                    self.feature_offset = off
                    self.window_node_features = wnf[off:off + n]
                    if len(self.window_node_features) < n:
                        print("Warning: node features shorter than windows after alignment; "
                              "falling back to computed features for the tail")
                    print(f"Loaded precomputed window_node_features: {wnf.shape} (offset {off})")
                else:
                    print(f"Warning: window_node_features shape {wnf.shape} incompatible, ignoring")
            except Exception as e:
                print(f"Warning: could not load window_node_features.npy: {e}")

        if vol_raw is not None and vol_raw.shape[1:] == self.windows.shape[1:]:
            shifted = vol_raw[self.feature_offset:self.feature_offset + n]
            if len(shifted) == n:
                self.volume_windows = shifted

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


class StratifiedBatchSampler(Sampler[list[int]]):
    """Emit regime-stratified batches that visit the whole training pool.

    Each batch holds `n_crisis` crisis windows and the rest normal ones, so the
    regime-conditional losses always see both classes. An epoch runs long enough
    for *both* classes to be visited at least once; the smaller class (crisis)
    is cycled. The old version stopped at `min(crisis//k, normal//k)`, so one
    epoch used only 664 of 2,234 training windows.

    DDP-aware: shards indices across ranks so each rank sees disjoint data.
    """

    def __init__(self, labels: np.ndarray, indices: np.ndarray, batch_size: int,
                 min_per_class: int = 2, seed: int = 42,
                 rank: int = 0, world_size: int = 1,
                 crisis_fraction: float | None = None,
                 max_batches: int | None = None) -> None:
        self.labels = np.asarray(labels)
        self.pool = np.asarray(indices)
        self.batch_size = int(batch_size)
        self.min_per_class = max(1, int(min_per_class))
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.max_batches = max_batches if max_batches and max_batches > 0 else None

        self.pool = self.pool[rank::world_size]
        pool_labels = self.labels[self.pool]
        frac = float(pool_labels.mean()) if crisis_fraction is None else float(crisis_fraction)
        per_rank = max(1, self.batch_size // world_size)
        self.n_crisis = max(self.min_per_class, int(round(per_rank * frac)))
        self.n_crisis = min(self.n_crisis, per_rank - 1) if per_rank > 1 else 1
        self.n_normal = max(1, per_rank - self.n_crisis)

    def _batches(self) -> int:
        pool_labels = self.labels[self.pool]
        n_crisis = int((pool_labels == 1).sum())
        n_normal = int((pool_labels == 0).sum())
        total = max(
            -(-n_crisis // self.n_crisis) if n_crisis else 0,
            -(-n_normal // self.n_normal) if n_normal else 0,
        )
        if self.max_batches:
            total = min(total, self.max_batches)
        return total

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        pool = self.pool
        crisis = pool[self.labels[pool] == 1]
        normal = pool[self.labels[pool] == 0]
        crisis = crisis[torch.randperm(len(crisis), generator=g).tolist()]
        normal = normal[torch.randperm(len(normal), generator=g).tolist()]
        ci = ni = 0
        for _ in range(self._batches()):
            if len(crisis) == 0:
                take_c = np.empty(0, dtype=crisis.dtype)
            elif ci + self.n_crisis <= len(crisis):
                take_c = crisis[ci:ci + self.n_crisis]
            else:  # cycle the minority stream
                idx = [(ci + k) % len(crisis) for k in range(self.n_crisis)]
                take_c = crisis[np.array(idx)]
            ci += self.n_crisis
            if len(normal) == 0:
                take_n = np.empty(0, dtype=normal.dtype)
            elif ni + self.n_normal <= len(normal):
                take_n = normal[ni:ni + self.n_normal]
            else:  # cycle the majority stream
                idx = [(ni + k) % len(normal) for k in range(self.n_normal)]
                take_n = normal[np.array(idx)]
            ni += self.n_normal
            batch = np.concatenate([take_c, take_n])
            perm = torch.randperm(len(batch), generator=g).tolist()
            yield [int(batch[i]) for i in perm]

    def __len__(self) -> int:
        return self._batches()


def merge_config(config: dict) -> dict:
    """Flatten the nested `cgnsde` block over the legacy top-level keys.

    Precedence, lowest to highest: legacy top-level keys, `cgnsde.loss_weights`,
    `cgnsde.*`. The `cgnsde` block is the CG-NSDE configuration; the top-level
    keys belong to the old LSTM baseline and used to silently win because they
    were merged last (`batch_size: 32` overrode `cgnsde.batch_size: 16`).
    """
    nested = config.get("cgnsde") or {}
    legacy = {k: v for k, v in config.items() if k != "cgnsde"}
    return {**legacy, **(nested.get("loss_weights") or {}), **nested}


def create_dataloaders(
    dataset: VortexDataset,
    batch_size: int,
    val_size: float = 0.15,
    test_size: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    min_per_class: int = 2,
    crisis_fraction: float | None = None,
    max_batches_per_epoch: int | None = None,
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

    # Get DDP rank/world_size for distributed sampling
    import os
    ddp_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

    effective_min_per_class = max(1, min(min_per_class, batch_size // 2))

    train_loader = DataLoader(
        dataset,
        batch_sampler=StratifiedBatchSampler(
            regimes, train_idx, batch_size, effective_min_per_class, seed,
            rank=ddp_rank, world_size=ddp_world_size,
            crisis_fraction=crisis_fraction,
            max_batches=max_batches_per_epoch,
        ),
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

    # 2.4: flatten the nested cgnsde block (+ loss_weights) so flat lookups below
    # pick up masterplan hyperparameters. The nested block wins over the legacy
    # top-level keys; explicit CLI values win over both.
    config = merge_config(config)

    # Override with CLI arguments
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "resume"]:
            config[key] = value

    if args.split is not None:
        print(f"Note: --split {args.split} is recorded for evaluation only; "
              "training always fits on train and validates on val.")
    
    # 4.1: ablation overrides (applied after config + CLI merge)
    ablation = config.get("ablation") or "none"
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
    
    # Validate training data before loading
    print("Validating training data...")
    validate_training_data(Path("data/raw"))
    print("Data validation passed.")
    
    # Load dataset
    dataset = VortexDataset(Path("data/raw"))
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.get("batch_size", 32),
        val_size=config.get("validation_size", 0.15),
        test_size=config.get("test_size", 0.2),
        num_workers=config.get("num_workers", 4),
        seed=config.get("seed", 42),
        min_per_class=config.get("min_per_class", 2),
        crisis_fraction=config.get("crisis_fraction"),
        max_batches_per_epoch=config.get("max_batches_per_epoch"),
    )
    print(f"Train batches per epoch: {len(train_loader)} "
          f"(crisis per batch: {train_loader.batch_sampler.n_crisis})")
    
    # Initialize model
    model = VORTEXModel(
        n_stocks=dataset.n_stocks,
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
        gat_dropout=config.get("gat_dropout", 0.1),
        weight_decay=config.get("weight_decay", 0.0),
        lam_corr=config.get("lam_corr", 0.0),
        lam_emission_corr=config.get("lam_emission_corr", 0.0),
        vol_dyn=config.get("vol_dyn", False),
        emission=config.get("emission", "point"),
        rank_K=config.get("rank_K", 0),
        sde_sigma_max=config.get("sde_sigma_max", 3.0),
        lambda_styl=config.get("lambda_styl", 0.1),
        stylized_sampler=config.get("stylized_sampler", "seq"),
        warmup_epochs=config.get("warmup_epochs", 20),
        corr_ramp=config.get("corr_ramp", 5),
        use_gat=use_gat,
        static_graph=static_graph,
        scheduler_epochs=config.get("max_epochs", 200),
    )

    # Per-regime correlation targets from the aligned training split. Computed
    # here instead of trusting data/raw/regime_corr.npz, which was built on the
    # pre-alignment indexing.
    sp = np.load("data/raw/split.npz", allow_pickle=True) if Path("data/raw/split.npz").exists() else None
    if sp is not None:
        tr_idx = np.asarray(sp["train_idx"])
        tr_windows = dataset.windows[tr_idx].astype(np.float64)
        tr_regimes = dataset.regimes[tr_idx]
        # Columns that are constant across the panel (LTI.NS and TATAMOTORS.NS
        # are empty in the vendor CSV) give corrcoef NaN and poison the loss.
        constant = np.flatnonzero(tr_windows.std(axis=(0, 1)) < 1e-8)
        if len(constant):
            print(f"Warning: {len(constant)} constant asset columns in the panel {constant.tolist()}; "
                  "their correlation rows are zeroed")
        targets = {}
        for rg, nm in ((1, "crisis"), (0, "normal")):
            sel = tr_windows[tr_regimes == rg]
            if len(sel) < 2:
                print(f"Warning: only {len(sel)} {nm} training windows; skipping target")
                continue
            flat = sel.reshape(-1, sel.shape[-1])
            corr = np.corrcoef(flat.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 1.0)
            targets[nm] = corr
            off = corr[~np.eye(corr.shape[0], dtype=bool)]
            print(f"Regime target {nm}: {len(sel)} windows, mean|offdiag|={np.abs(off).mean():.3f}")
        if len(targets) == 2:
            model.set_regime_corr(targets)
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
    
    # W&B Sweep configuration (if sweep enabled)
    sweep_config = config.get("wandb_sweep")
    if sweep_config and config.get("wandb_project"):
        sweep_yaml = sweep_config.get("config", {})
        sweep_id = wandb.sweep(
            sweep=sweep_yaml,
            project=config["wandb_project"],
            entity=config.get("wandb_entity"),
        )
        print(f"Created W&B sweep: {sweep_id}")
        
        # Run sweep agent
        def train_sweep():
            train(args)  # Recursive call with sweep config
        
        wandb.agent(sweep_id, function=train_sweep, count=sweep_config.get("count", 50))
        wandb.finish()
        return  # Exit after sweep
    
    # Setup callbacks
    default_monitor = "val/nll" if model._use_nll else "val/total"
    monitor_metric = config.get("monitor_metric") or default_monitor
    if monitor_metric == "val/reconstruction" and model._use_nll:
        print("Ignoring monitor_metric=val/reconstruction: this emission is trained by NLL.")
        monitor_metric = "val/nll"
    ckpt_tag = "vortex" if ablation == "none" else f"vortex-abl-{ablation}"
    # Use full metric name for filename (Lightning replaces / with _ automatically)
    # The epoch-only filename stays flat; a {metric} placeholder would create a
    # subdirectory because "val/nll" contains a slash.
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename=ckpt_tag + "-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=config.get("save_top_k", 3),
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=config.get("patience", 20),
        mode="min",
        verbose=True,
    )
    print(f"EarlyStopping patience: {config.get('patience', 20)} (best_val monitor {monitor_metric})")
    
    # Setup trainer (logger=False when wandb off: env tensorboard/protobuf is broken)
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 200),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        logger=wandb_logger if wandb_logger else False,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        log_every_n_steps=config.get("log_every_n_steps", 10),
        deterministic=True,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting VORTEX-AI (CG-NSDE) Training")
    print("="*80 + "\n")
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    # Record which checkpoint actually won, so evaluation cannot accidentally
    # score `last.ckpt` (the epoch training stopped on) instead of the best.
    best_path = checkpoint_callback.best_model_path
    best_file = Path("models/checkpoints/best_model_path.txt")
    best_file.parent.mkdir(parents=True, exist_ok=True)
    best_file.write_text(str(best_path), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (val/nll {checkpoint_callback.best_model_score})")
    print(f"Recorded in {best_file}")

    # Fit the generative prior on the training windows and persist it beside the
    # checkpoint. Generation uses only this prior and the requested regime, so
    # evaluation never sees the window it is asked to produce.
    print("\nFitting generative priors (per-regime aggregate posterior)...")
    model.fit_generation_priors(train_loader)
    priors_path = Path("models/checkpoints/generation_priors.npz")
    np.savez(
        priors_path,
        prior_mu=model.prior_mu.detach().cpu().numpy(),
        prior_logvar=model.prior_logvar.detach().cpu().numpy(),
        regime_graph_emb=model.regime_graph_emb.detach().cpu().numpy(),
        regime_log_vol=model.regime_log_vol.detach().cpu().numpy(),
        regime_log_vol_std=model.regime_log_vol_std.detach().cpu().numpy(),
    )
    print(f"Generative priors saved to {priors_path}")
    
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Every override defaults to None so that a flag the user did not pass cannot
    overwrite the YAML. `--vol_dyn` used to be `store_true` with default False,
    and the merge loop copied that False over `vol_dyn: true` in config.yaml.
    """
    parser = argparse.ArgumentParser(description="Train VORTEX-AI (CG-NSDE)")
    
    parser.add_argument("--config", type=str, default="training/config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Override config parameters
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum epochs")
    parser.add_argument("--accelerator", type=str, default=None,
                       choices=["auto", "cpu", "cuda", "mps"], help="Accelerator type")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices")
    parser.add_argument("--lam_con", type=float, default=None, help="Contrastive loss weight")
    parser.add_argument("--lam_graph", type=float, default=None, help="Graph loss weight")
    parser.add_argument("--lam_na", type=float, default=None, help="Circuit-filter loss weight")
    parser.add_argument("--beta_max", type=float, default=None, help="KL weight ceiling (7.1)")
    parser.add_argument("--prior_warmup", type=int, default=None, help="KL ramp epochs (7.1)")
    parser.add_argument("--lam_var", type=float, default=None, help="Variance loss weight (7.2)")
    parser.add_argument("--gat_dropout", type=float, default=None, help="GAT dropout (Stage 1: 0.6)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Adam L2 (Stage 1: 0.0005)")
    parser.add_argument("--lam_corr", type=float, default=None, help="Correlation loss target (Lane 1A)")
    parser.add_argument("--lam_emission_corr", type=float, default=None,
                       help="Emission covariance correlation weight")
    parser.add_argument("--vol_dyn", action="store_true", default=None,
                       help="Market vol state (Lane 1B)")
    parser.add_argument("--emission", type=str, default=None,
                       choices=["point", "hetero", "t", "mt", "smt"], help="Emission head (7.3)")
    parser.add_argument("--rank_K", type=int, default=None, help="Emission factor rank (7.3b)")
    parser.add_argument("--sde_sigma_max", type=float, default=None, help="SDE diffusion bound (7.3)")
    parser.add_argument("--run_tag", type=str, default=None, help="Run dir suffix under runs/")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Dataloader workers (0 = safest on Windows)")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience in epochs")
    parser.add_argument("--lambda_styl", type=float, default=None, help="Stylized-facts loss weight")
    parser.add_argument("--stylized_sampler", type=str, default=None,
                       choices=["single", "seq"],
                       help="Sampler for the stylized loss: 'single' (legacy one-shot) "
                            "or 'seq' (sequential with vol feedback, matches generation)")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "val", "test"],
                       help="Data split for downstream evaluation (recorded only; "
                            "training always fits on train and validates on val)")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Contrastive warmup epochs")
    parser.add_argument("--monitor_metric", type=str, default=None,
                       help="Validation metric for checkpointing/early stopping")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                       help="Cap batches per epoch (0 = cover the whole pool)")
    parser.add_argument("--crisis_fraction", type=float, default=None,
                       help="Crisis share per batch (default: empirical train rate)")
    # 4.1: ablation selector (none = full model)
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["none", "no-gat", "no-supcon", "static-graph", "no-circuit"],
                        help="Ablation variant to train")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args)
