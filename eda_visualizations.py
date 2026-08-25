from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set global plotting style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 300


def load_data():
    returns = np.load(RAW_DIR / "returns.npy")
    regimes = np.load(RAW_DIR / "regimes.npy")
    adj_matrices = np.load(RAW_DIR / "adj_matrices.npy")
    close_df = pd.read_csv(RAW_DIR / "nifty50_close.csv", index_col=0, parse_dates=True)
    dates = close_df.index[1:]  # Match log-returns length
    return returns, regimes, adj_matrices, dates


def plot_eda():
    returns, regimes, adj_matrices, dates = load_data()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1. Market Index Return with Crisis Regimes Shaded
    ax1 = fig.add_subplot(gs[0, :])
    market_cum_return = np.cumsum(returns.mean(axis=1))
    ax1.plot(dates, market_cum_return, color="#1f77b4", linewidth=1.2, label="Equal-Weighted NIFTY-50 Log Return")
    
    # Shade Crisis periods (regime == 1)
    crisis_mask = regimes == 1
    ax1.fill_between(dates, ax1.get_ylim()[0], ax1.get_ylim()[1], where=crisis_mask, color="crimson", alpha=0.35, label="Crisis Regime (15% Stress)")
    ax1.set_title("NIFTY-50 Cumulative Returns & Identified Crisis Regimes (2010–2024)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left")

    # 2. Return Distributions: Normal vs Crisis Regime
    ax2 = fig.add_subplot(gs[1, 0])
    normal_returns = returns[regimes == 0].flatten()
    crisis_returns = returns[regimes == 1].flatten()
    
    sns.kdeplot(normal_returns, ax=ax2, color="blue", label="Normal Regime", fill=True, alpha=0.2)
    sns.kdeplot(crisis_returns, ax=ax2, color="red", label="Crisis Regime", fill=True, alpha=0.2)
    ax2.set_xlim(-0.08, 0.08)
    ax2.set_title("Asset Return Density (Fat-Tail Spike in Crisis)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Daily Log-Return")
    ax2.legend()

    # 3. Rolling Graph Sparsity / Density over Time
    ax3 = fig.add_subplot(gs[1, 1])
    # Density = non-zero edges / total possible edges (excluding diagonal)
    N = adj_matrices.shape[1]
    max_edges = N * (N - 1)
    edge_densities = [(np.count_nonzero(A) - N) / max_edges for A in adj_matrices]
    
    window_dates = dates[len(dates) - len(adj_matrices):]
    ax3.plot(window_dates, edge_densities, color="purple", linewidth=1.0)
    ax3.set_title("Dynamic Graph Edge Density (T=60 Window)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sparsity Ratio")
    ax3.set_xlabel("Date")

    # 4. Heatmap: Adjacency Matrix during a Normal Window
    ax4 = fig.add_subplot(gs[2, 0])
    normal_idx = np.where(regimes[60:] == 0)[0][100]  # Pick representative normal sample
    sns.heatmap(adj_matrices[normal_idx], ax=ax4, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    ax4.set_title(f"Empirical Graph Target $A_t^{{emp}}$ (Normal Day)", fontsize=11, fontweight="bold")

    # 5. Heatmap: Adjacency Matrix during a Crisis Window
    ax5 = fig.add_subplot(gs[2, 1])
    crisis_idx = np.where(regimes[60:] == 1)[0][10]   # Pick representative crisis sample
    sns.heatmap(adj_matrices[crisis_idx], ax=ax5, cmap="Reds", cbar=True, xticklabels=False, yticklabels=False)
    ax5.set_title(f"Empirical Graph Target $A_t^{{emp}}$ (Crisis Day)", fontsize=11, fontweight="bold")

    # Save visualization dashboard
    plt.tight_layout()
    output_filepath = OUT_DIR / "nifty50_eda_dashboard.png"
    plt.savefig(output_filepath, dpi=300)
    print(f"EDA Dashboard successfully generated: {output_filepath}")
    plt.show()


if __name__ == "__main__":
    plot_eda()