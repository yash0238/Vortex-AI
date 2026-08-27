# Vortex-AI

Dynamic graph-based financial risk and regime-switching experiments using NIFTY-50 returns.

## Setup

Install the dependencies with:

```powershell
py -m pip install -r requirements.txt
```

## Pipeline

Run these commands from the repository root:

```powershell
py data/preprocess.py
py data/graph_builder.py
py inspect_data.py
py eda_visualizations.py
py -m training.trainer --epochs 50 --device auto
```

Preprocessing creates return windows and regime labels in `data/raw/`. Graph construction
then creates empirical Pearson/Granger adjacency targets in `data/raw/adj_matrices.npy`.
Training uses those targets to jointly reconstruct the graph and classify the regime; the
best checkpoint is written to `models/checkpoints/best_model.pt`.

Use `--device cuda` to require NVIDIA CUDA, `--device cpu` to force CPU execution, or
`--device auto` to select CUDA when the installed PyTorch build supports it.

The default window is 60 trading days, and the model expects arrays with shapes:

- `windows.npy`: `(num_windows, 60, num_assets)`
- `adj_matrices.npy`: `(num_windows, num_assets, num_assets)`
- `window_regimes.npy`: `(num_windows,)`