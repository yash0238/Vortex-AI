# VORTEX-AI Quick Start Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training, optional)
- 16GB+ RAM recommended
- 10GB+ free disk space

## Step-by-Step Execution

### Step 1: Environment Setup (5 minutes)

```bash
# Clone and navigate
cd Vortex-AI

# Create virtual environment
python -m venv venv

# Activate
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; import torchsde; import pytorch_lightning; print('✓ All packages installed')"
```

### Step 2: Data Pipeline (10-15 minutes)

```bash
# Download NIFTY-50 historical data (2010-2024)
python data/download.py

# Expected output:
# Downloaded: ~3500 days x 50 stocks
# Files: data/raw/nifty50_close.csv, nifty50_volume.csv

# Preprocess: compute log returns and regime labels
python data/preprocess.py

# Expected output:
# Crisis days labeled: ~500 / 3500 (15%)
# Windowed data shape: (3440, 60, 50)

# Build adjacency matrices
python -m src.data_processor

# Verify pipeline integrity
python check_pipeline.py

# Expected output:
# ✓ All pipeline checks passed
```

**Troubleshooting**:
- If yfinance fails: Check internet connection, retry
- If OOM during preprocessing: Reduce window size or stocks

### Step 3: Train VORTEX-AI Model (2-4 hours on GPU, 12-24 hours on CPU)

```bash
# Quick test run (10 epochs, for verification)
python train_vortex.py \
  --config training/config.yaml \
  --max_epochs 10 \
  --batch_size 16

# Full training (200 epochs)
python train_vortex.py --config training/config.yaml

# Training on CPU (slower)
python train_vortex.py --config training/config.yaml --accelerator cpu

# Monitor training
# - Check wandb dashboard (if wandb_project set in config)
# - Or watch terminal output for loss curves
```

**Expected Training Output**:
```
Epoch 0:  100%|██████████| 85/85 [00:45<00:00,  1.87it/s, loss=2.453]
Epoch 0: train/loss/total=2.453, train/loss/reconstruction=2.201, ...

Epoch 10: 100%|██████████| 85/85 [00:42<00:00,  2.01it/s, loss=1.123]
val/total=1.089, val/reconstruction=0.967, ...

Epoch 50: Best model so far! Saving checkpoint...
```

**Monitor These**:
- Total loss should decrease (target: <1.0 by epoch 100)
- Reconstruction loss < 0.8
- Contrastive loss < 0.3
- No NaN/Inf (if occurs, reduce learning rate or increase gradient_clip_val)

### Step 4: Evaluate Model (5-10 minutes)

```bash
# Comprehensive evaluation
python evaluate_vortex.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --n_scenarios 100

# Generate more scenarios for robust metrics
python evaluate_vortex.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --n_scenarios 500 \
  --output results/eval_500.json
```

**Expected Evaluation Output**:
```
================================================================================
VORTEX-AI (CG-NSDE) Comprehensive Evaluation
================================================================================

Loading model from: models/checkpoints/best_vortex.ckpt
Generating 100 synthetic scenarios...

--------------------------------------------------------------------------------
1. Statistical Fidelity Tests
--------------------------------------------------------------------------------
Kurtosis (real): 4.231
Kurtosis (gen):  3.892
Kurtosis pass:   True

ACF squared (real): 0.0823
ACF squared (gen):  0.0671
ACF pass:           True

Correlation error: 2.234
Correlation pass:  True

✓ All tests pass: True

--------------------------------------------------------------------------------
2. Discriminative Score Test
--------------------------------------------------------------------------------
Discriminative score: 0.543 ± 0.032
Target: <0.60 (closer to 0.5 is better)
Pass: True

--------------------------------------------------------------------------------
3. Contagion Propagation Test
--------------------------------------------------------------------------------
Normal correlation:  0.2134
Crisis correlation:  0.3876
Crisis boost:        81.68%
Target: >50% boost
Pass: True

--------------------------------------------------------------------------------
4. CVaR Regime Ratio
--------------------------------------------------------------------------------
CVaR (normal): 0.002341
CVaR (crisis): 0.008123
CVaR ratio:    3.47x
Target: >3x
Pass: True

================================================================================
EVALUATION SUMMARY
================================================================================
Statistical Fidelity........................ ✓ PASS
Discriminative Score........................ ✓ PASS
Contagion Propagation....................... ✓ PASS
CVaR Ratio.................................. ✓ PASS

================================================================================
Overall: 4/4 tests passed
================================================================================
```

### Step 5: Generate Visualizations (2-3 minutes)

```bash
# Generate all figures
python visualize_results.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --output_dir results/figures \
  --n_scenarios 100

# Custom scenarios count
python visualize_results.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --output_dir results/my_figures \
  --n_scenarios 200
```

**Generated Files**:
- `results/figures/fan_chart.png` - Normal vs crisis scenario distributions
- `results/figures/adjacency_heatmap.png` - Learned graph structure
- `results/figures/pnl_distribution.png` - Strategy P&L with Sharpe ratio

**Example P&L Output**:
```
P&L Statistics:
Mean: 0.002341
Std:  0.018923
Sharpe (annualized): 1.967
Win Rate: 58.34%
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
python train_vortex.py --batch_size 8

# Or use CPU
python train_vortex.py --accelerator cpu
```

### Issue 2: Training Loss = NaN
```bash
# Solution: Increase gradient clipping or reduce learning rate
# Edit training/config.yaml:
gradient_clip_val: 0.5  # Was 1.0
learning_rate: 0.0005   # Was 0.001
```

### Issue 3: Import Errors
```bash
# Reinstall torch-geometric properly
pip uninstall torch-geometric torch-scatter torch-sparse
pip install torch-geometric torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Issue 4: Data Download Fails
```bash
# Check internet connection
# If yfinance is blocked, download manually:
# 1. Visit NSE website or Yahoo Finance
# 2. Download NIFTY-50 historical CSV
# 3. Place in data/raw/nifty50_close.csv
```

### Issue 5: Wandb Login Required
```bash
# If you don't have wandb account, disable it
# Edit training/config.yaml:
wandb_project: null  # Set to null to disable
```

## Performance Benchmarks

### Training Time (RTX 3090, batch_size=32)
- 10 epochs: ~7 minutes
- 50 epochs: ~35 minutes
- 200 epochs: ~2.5 hours

### Training Time (CPU, batch_size=16)
- 10 epochs: ~45 minutes
- 50 epochs: ~4 hours
- 200 epochs: ~16 hours

### Evaluation Time
- 100 scenarios: ~2 minutes (GPU) / ~8 minutes (CPU)
- 500 scenarios: ~10 minutes (GPU) / ~40 minutes (CPU)

### Memory Requirements
- Training (GPU): ~6-8 GB VRAM
- Training (CPU): ~12 GB RAM
- Evaluation: ~4 GB RAM

## Next Steps

### For Research
1. **Run ablations**: Train No-GAT, No-SupCon, Static-Graph variants
2. **Compare baselines**: Train TimeGAN and QuantGAN for comparison
3. **Tune hyperparameters**: Try different lam_con, lam_graph values
4. **Analyze attention**: Visualize which stocks attend to which during crisis

### For Production
1. **Extend to more assets**: Modify n_stocks and retrain
2. **Add more features**: Include volume, sector embeddings
3. **Real-time generation**: Implement streaming inference
4. **Integration**: Connect to backtesting framework

### For Publication
1. **Write paper**: Use structure from masterplan Part H
2. **Run all experiments**: Ensure reproducibility
3. **Generate all figures**: Professional quality (300 DPI)
4. **Prepare code release**: Clean up, add tests, documentation

## Verification Checklist

Before moving forward, verify:

- [ ] Data pipeline completes without errors
- [ ] Training reaches epoch 50+ without NaN
- [ ] Evaluation passes at least 3/4 tests
- [ ] Visualizations are generated
- [ ] Loss curves show decreasing trend
- [ ] Generated scenarios look realistic (not all zeros or flat)

## Getting Help

If you encounter issues:

1. **Check logs**: Look for error messages in terminal
2. **Verify versions**: `pip list | grep torch`
3. **Read documentation**: See README.md and IMPLEMENTATION_SUMMARY.md
4. **Open issue**: GitHub issues with error trace
5. **Contact**: Reach out to maintainers

## Useful Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check dataset integrity
python -c "import numpy as np; print(np.load('data/raw/windows.npy').shape)"

# List checkpoints
ls -lh models/checkpoints/

# View config
cat training/config.yaml

# Tail training logs (if saved)
tail -f logs/training.log
```

## Success Criteria

Your implementation is working if:

✅ Training completes 200 epochs
✅ Validation loss < 1.0
✅ Kurtosis > 3.0 (fat tails)
✅ Discriminative score < 0.60
✅ Crisis correlation boost > 50%
✅ CVaR ratio > 3x
✅ Visualizations show realistic patterns

---

**Estimated total time**: 3-6 hours (including training)
**Difficulty**: Intermediate (Python + ML background helpful)
**Support**: See README.md for detailed documentation
