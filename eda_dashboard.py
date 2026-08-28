"""Generate a five-panel exploratory dashboard from prepared arrays."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "plots"


def plot_eda() -> Path:
    returns = np.load(RAW_DIR / "returns.npy")
    regimes = np.load(RAW_DIR / "regimes.npy")
    windows = np.load(RAW_DIR / "windows.npy")
    labels = np.load(RAW_DIR / "window_regimes.npy")
    adjacency = np.load(RAW_DIR / "adj_matrices.npy")
    close = pd.read_csv(RAW_DIR / "nifty50_close.csv", index_col=0, parse_dates=True)
    dates = close.index[:len(returns)]
    window_dates = dates[windows.shape[1] - 1:windows.shape[1] - 1 + len(adjacency)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    figure = plt.figure(figsize=(16, 12))
    grid = figure.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    axis = figure.add_subplot(grid[0, :])
    cumulative = np.cumsum(returns.mean(axis=1))
    axis.plot(dates, cumulative, color="#1f77b4", label="Equal-weighted cumulative return")
    axis.fill_between(dates, cumulative.min(), cumulative.max(), where=regimes == 1,
                      color="crimson", alpha=0.25, label="Crisis regime")
    axis.set_title("NIFTY-50 Cumulative Returns and Crisis Regimes")
    axis.legend(loc="upper left")

    axis = figure.add_subplot(grid[1, 0])
    sns.kdeplot(returns[regimes == 0].ravel(), ax=axis, fill=True, alpha=0.2, label="Normal")
    if np.any(regimes == 1):
        sns.kdeplot(returns[regimes == 1].ravel(), ax=axis, fill=True, alpha=0.2, label="Crisis")
    axis.set_xlim(-0.08, 0.08)
    axis.set_title("Asset Return Density")
    axis.legend()

    axis = figure.add_subplot(grid[1, 1])
    asset_count = adjacency.shape[1]
    density = [np.count_nonzero(matrix) / (asset_count * (asset_count - 1)) for matrix in adjacency]
    axis.plot(window_dates, density, color="purple")
    axis.set_title("Dynamic Graph Edge Density")

    normal = np.flatnonzero(labels == 0)
    crisis = np.flatnonzero(labels == 1)
    for position, selected, color, title in [
        (grid[2, 0], normal, "Blues", "Normal Window"),
        (grid[2, 1], crisis, "Reds", "Crisis Window"),
    ]:
        axis = figure.add_subplot(position)
        index = int(selected[0]) if len(selected) else 0
        sns.heatmap(adjacency[index], ax=axis, cmap=color, xticklabels=False, yticklabels=False)
        axis.set_title(f"Empirical Adjacency ({title})")

    output = OUTPUT_DIR / "nifty50_eda_dashboard.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"EDA dashboard saved to {output}")
    return output


if __name__ == "__main__":
    plot_eda()