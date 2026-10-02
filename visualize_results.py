"""
Visualization generation for VORTEX-AI (CG-NSDE) results.

Generates visualizations as specified in masterplan Part H (Figure generation):
1. Fan charts (normal vs crisis scenarios)
2. Adjacency heatmap animations/comparisons
3. P&L distribution under strategy sandbox

Usage:
    python visualize_results.py --checkpoint models/checkpoints/best_vortex.ckpt
    python visualize_results.py --checkpoint models/checkpoints/best_vortex.ckpt --output_dir results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import json

from train_vortex import VortexDataset
from training.vortex_model import VORTEXModel


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def generate_fan_chart(
    model: VORTEXModel,
    dataloader: DataLoader,
    n_scenarios: int = 100,
    device: str = "cuda",
    output_path: Path | None = None,
):
    """
    Generate fan chart showing distribution of scenario paths.
    
    As specified in masterplan Part H - Figure 1: Fan chart (normal vs crisis scenarios).
    
    Args:
        model: trained VORTEX model
        dataloader: data loader
        n_scenarios: number of scenarios to generate
        device: computation device
        output_path: where to save figure
    """
    model.eval()
    model.to(device)
    
    normal_scenarios = []
    crisis_scenarios = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(normal_scenarios) >= n_scenarios // 2 and len(crisis_scenarios) >= n_scenarios // 2:
                break
            
            x = batch["returns"].to(device)
            adj = batch["adj_matrix"].to(device)
            node_feats = batch["node_features"].to(device)
            regimes = batch["regimes"]
            
            # Generate
            r_hat, _, _ = model(x, adj, node_feats)
            
            # Separate by regime
            for i, regime in enumerate(regimes):
                if regime == 0 and len(normal_scenarios) < n_scenarios // 2:
                    normal_scenarios.append(r_hat[i].cpu().numpy())
                elif regime == 1 and len(crisis_scenarios) < n_scenarios // 2:
                    crisis_scenarios.append(r_hat[i].cpu().numpy())
    
    normal_scenarios = np.array(normal_scenarios)  # (N, T, n_stocks)
    crisis_scenarios = np.array(crisis_scenarios)
    
    # Compute cumulative returns for visualization
    def cumulative_returns(scenarios):
        """Convert returns to cumulative price paths."""
        return np.exp(np.cumsum(scenarios, axis=1))
    
    normal_cum = cumulative_returns(normal_scenarios).mean(axis=2)  # Average across stocks
    crisis_cum = cumulative_returns(crisis_scenarios).mean(axis=2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    T = normal_cum.shape[1]
    time_axis = np.arange(T)
    
    # Plot normal scenarios
    ax1.set_title("Normal Regime Scenarios", fontsize=14, fontweight='bold')
    for scenario in normal_cum:
        ax1.plot(time_axis, scenario, alpha=0.1, color='steelblue')
    
    # Percentiles
    p5 = np.percentile(normal_cum, 5, axis=0)
    p25 = np.percentile(normal_cum, 25, axis=0)
    p50 = np.percentile(normal_cum, 50, axis=0)
    p75 = np.percentile(normal_cum, 75, axis=0)
    p95 = np.percentile(normal_cum, 95, axis=0)
    
    ax1.fill_between(time_axis, p5, p95, alpha=0.3, color='steelblue', label='5th-95th percentile')
    ax1.fill_between(time_axis, p25, p75, alpha=0.5, color='steelblue', label='25th-75th percentile')
    ax1.plot(time_axis, p50, color='darkblue', linewidth=2, label='Median')
    
    ax1.set_xlabel("Time Steps", fontsize=12)
    ax1.set_ylabel("Cumulative Return", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot crisis scenarios
    ax2.set_title("Crisis Regime Scenarios", fontsize=14, fontweight='bold')
    for scenario in crisis_cum:
        ax2.plot(time_axis, scenario, alpha=0.1, color='crimson')
    
    # Percentiles
    p5 = np.percentile(crisis_cum, 5, axis=0)
    p25 = np.percentile(crisis_cum, 25, axis=0)
    p50 = np.percentile(crisis_cum, 50, axis=0)
    p75 = np.percentile(crisis_cum, 75, axis=0)
    p95 = np.percentile(crisis_cum, 95, axis=0)
    
    ax2.fill_between(time_axis, p5, p95, alpha=0.3, color='crimson', label='5th-95th percentile')
    ax2.fill_between(time_axis, p25, p75, alpha=0.5, color='crimson', label='25th-75th percentile')
    ax2.plot(time_axis, p50, color='darkred', linewidth=2, label='Median')
    
    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_ylabel("Cumulative Return", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Fan chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_adjacency_heatmap(
    model: VORTEXModel,
    dataloader: DataLoader,
    device: str = "cuda",
    output_path: Path | None = None,
    stock_names: list[str] | None = None,
):
    """
    Generate adjacency matrix heatmap comparison.
    
    As specified in masterplan Part H - Figure 2: Adjacency heatmap animation.
    Shows empirical vs learned adjacency matrices.
    
    Args:
        model: trained VORTEX model
        dataloader: data loader
        device: computation device
        output_path: where to save figure
        stock_names: optional stock names for labels
    """
    model.eval()
    model.to(device)
    
    # Get one batch
    batch = next(iter(dataloader))
    x = batch["returns"].to(device)
    adj_emp = batch["adj_empirical"].to(device)
    node_feats = batch["node_features"].to(device)
    
    with torch.no_grad():
        _, A_learned, _ = model(x, adj_emp, node_feats)
    
    # Take first sample
    adj_empirical_np = adj_emp[0].cpu().numpy()
    adj_learned_np = A_learned[0].cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Empirical adjacency
    im1 = ax1.imshow(adj_empirical_np, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax1.set_title("Empirical Adjacency (Pearson)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Stock Index")
    ax1.set_ylabel("Stock Index")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Learned adjacency
    im2 = ax2.imshow(adj_learned_np, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title("Learned Adjacency (GAT Attention)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Stock Index")
    ax2.set_ylabel("Stock Index")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Difference
    diff = adj_learned_np - adj_empirical_np
    im3 = ax3.imshow(diff, cmap='seismic', vmin=-0.5, vmax=0.5, aspect='auto')
    ax3.set_title("Difference (Learned - Empirical)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Stock Index")
    ax3.set_ylabel("Stock Index")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Adjacency heatmap saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_pnl_distribution(
    model: VORTEXModel,
    dataloader: DataLoader,
    n_scenarios: int = 500,
    device: str = "cuda",
    output_path: Path | None = None,
):
    """
    Generate P&L distribution under simple momentum strategy.
    
    As specified in masterplan Part H - Figure 3: P&L distribution (momentum strategy).
    
    Args:
        model: trained VORTEX model
        dataloader: data loader
        n_scenarios: number of scenarios
        device: computation device
        output_path: where to save figure
    """
    model.eval()
    model.to(device)
    
    pnl_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(pnl_list) >= n_scenarios:
                break
            
            x = batch["returns"].to(device)
            adj = batch["adj_matrix"].to(device)
            node_feats = batch["node_features"].to(device)
            
            # Generate
            r_hat, _, _ = model(x, adj, node_feats)
            returns = r_hat.cpu().numpy()
            
            # Simple momentum strategy: buy if positive momentum, sell if negative
            for scenario in returns:
                # Compute momentum (sign of past 5-day return)
                momentum_window = 5
                positions = np.zeros_like(scenario)
                
                for t in range(momentum_window, scenario.shape[0]):
                    past_returns = scenario[t-momentum_window:t].sum(axis=0)
                    positions[t] = np.sign(past_returns)  # +1 long, -1 short, 0 neutral
                
                # Compute P&L: position * next return
                pnl = (positions[:-1] * scenario[1:]).sum(axis=1).sum()
                pnl_list.append(pnl)
    
    pnl_array = np.array(pnl_list)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(pnl_array, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero P&L')
    ax1.axvline(pnl_array.mean(), color='darkblue', linestyle='-', linewidth=2, 
                label=f'Mean: {pnl_array.mean():.4f}')
    ax1.set_title("P&L Distribution (Momentum Strategy)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Cumulative P&L", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot (check for normality)
    from scipy import stats
    stats.probplot(pnl_array, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normality Check)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("\nP&L Statistics:")
    print(f"Mean: {pnl_array.mean():.6f}")
    print(f"Std:  {pnl_array.std():.6f}")
    print(f"Min:  {pnl_array.min():.6f}")
    print(f"Max:  {pnl_array.max():.6f}")
    print(f"Sharpe (annualized): {(pnl_array.mean() / pnl_array.std()) * np.sqrt(252):.3f}")
    print(f"Win Rate: {(pnl_array > 0).mean():.2%}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nP&L distribution saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_loss_curves(
    training_log: str,
    output_path: Path | None = None,
):
    """
    Generate training/validation loss curves from wandb or tensorboard logs.
    
    Args:
        training_log: path to training log JSON
        output_path: where to save figure
    """
    # This is a placeholder - actual implementation would parse wandb/tensorboard logs
    print("Loss curves generation requires wandb API or tensorboard log parsing.")
    print("Use wandb web interface or tensorboard for interactive loss visualization.")


def visualize_all(args: argparse.Namespace):
    """Generate all visualizations."""
    print("\n" + "="*80)
    print("VORTEX-AI (CG-NSDE) Visualization Generation")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    # Load model (strict=False: see evaluate_vortex.py note on 4.1 fallback keys)
    model = VORTEXModel.load_from_checkpoint(args.checkpoint, strict=False)
    
    # Load dataset
    print("Loading dataset...")
    dataset = VortexDataset(Path("data/raw"))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 1. Fan chart
    print("\n" + "-"*80)
    print("Generating Fan Chart...")
    print("-"*80)
    generate_fan_chart(
        model, dataloader, n_scenarios=args.n_scenarios,
        device=args.device,
        output_path=output_dir / "fan_chart.png"
    )
    
    # 2. Adjacency heatmap
    print("\n" + "-"*80)
    print("Generating Adjacency Heatmap...")
    print("-"*80)
    generate_adjacency_heatmap(
        model, dataloader,
        device=args.device,
        output_path=output_dir / "adjacency_heatmap.png"
    )
    
    # 3. P&L distribution
    print("\n" + "-"*80)
    print("Generating P&L Distribution...")
    print("-"*80)
    generate_pnl_distribution(
        model, dataloader, n_scenarios=args.n_scenarios,
        device=args.device,
        output_path=output_dir / "pnl_distribution.png"
    )
    
    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate VORTEX-AI visualizations")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Output directory for figures")
    parser.add_argument("--n_scenarios", type=int, default=100,
                       help="Number of scenarios for fan chart and P&L")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_all(args)
