# VORTEX-AI (CG-NSDE) Implementation Summary

## Overview

VORTEX-AI (Contrastive Graph-Neural Stochastic Differential Equations) framework
for NSE stress testing as specified in the masterplan document.

## Implementation Status (2026-09-25 repair pass, Phases 0-4 per repair plan)

Phases 0-3 complete and smoke-verified (1-epoch train + eval run clean);
Phase 4 partial (4 ablations x 10 epochs distinct, baselines documented as
blocked, 3 figures at 300 DPI). End-to-end quality targets NOT yet met
(0/4 eval on 1-epoch CPU smoke; val loss dominated by stylized-term scale —
see repair-plan notes). Full 200-epoch run + legacy baseline regression still
pending. The "13/13 tasks complete" claim from the prior revision was
inaccurate and is superseded by this section.

### Core Components Implemented

#### 1. Model Architecture
- ✅ **DynamicGATEncoder** (`models/gat_encoder.py`)
  - 2-layer Graph Attention Network (4 heads → 1 head)
  - LayerNorm + ELU activation
  - Learns dynamic adjacency matrices from attention weights
  
- ✅ **GraphConditionedSDE** (`models/neural_sde.py`)
  - Inherits from `torchsde.SDEIto`
  - Drift and diffusion networks conditioned on graph embeddings + time
  - Compatible with Euler-Maruyama solver
  
- ✅ **LatentSDEModel** (`models/neural_sde.py`)
  - GRU encoder: returns → latent z_0
  - Graph-conditioned SDE integration
  - Linear decoder: latent paths → reconstructed returns
  
- ✅ **SupConLoss + ProjectionHead** (`models/contrastive.py`)
  - Khosla et al. 2020 formula with temperature τ=0.07
  - L2 normalization with proper positive/negative pair masking
  - Temporal pooling before projection

#### 2. Loss Functions
- ✅ **Complete Loss Assembly** (`training/losses.py`)
  - `stylized_facts_loss`: Kurtosis + ACF(r²) matching
  - `reconstruction_loss`: MSE + stylized facts
  - `graph_consistency_loss`: Frobenius norm
  - `total_loss`: Weighted combination of all components
  - Loss weights: λ_rec=1.0, λ_graph=0.1, λ_con=0.1

#### 3. NSE Circuit Breakers
- ✅ **Circuit Filter Implementation** (`sandbox/circuit_filter.py`)
  - Band A: ±5% (SBI, Tata Motors, Tata Steel)
  - Band B: ±10% (most NIFTY-50 stocks)
  - Band C: ±20% (less liquid stocks)
  - Hard clipping (post-generation) + soft penalty (training)

#### 4. Training Infrastructure
- ✅ **VORTEXModel** (`training/vortex_model.py`)
  - PyTorch Lightning module integrating all components
  - training_step, validation_step with full loss computation
  - Adam optimizer + CosineAnnealingLR scheduler
  - Scenario generation method for inference
  
- ✅ **Training Script** (`train_vortex.py`)
  - VortexDataset with 6-dimensional node features
  - Train/val/test split with proper random seeding
  - Wandb logging integration
  - ModelCheckpoint + EarlyStopping callbacks
  - Gradient clipping (max_norm=1.0) for SDE stability
  
- ✅ **Configuration** (`training/config.yaml`)
  - All masterplan hyperparameters
  - T=60, latent_dim=64, proj_dim=32, batch_size=32
  - max_epochs=200, lr=0.001, gat_heads=4

#### 5. Evaluation Metrics
- ✅ **Statistical Fidelity** (`eval/statistical.py`)
  - Kurtosis test (target: >3.0 for fat tails)
  - ACF of squared returns (target: >0.05 for volatility clustering)
  - Correlation matrix error (target: <2.5 Frobenius norm)
  - Distribution moment comparisons
  
- ✅ **Discriminative Score** (`eval/discriminative.py`)
  - GradientBoostingClassifier with 5-fold CV
  - Target: <0.60 accuracy (closer to 0.5 is better)
  - Predictive score test (train on synthetic, test on real)
  
- ✅ **Contagion Tests** (`eval/contagion.py`)
  - Crisis correlation boost (target: >50%)
  - CVaR regime ratio (target: >3x)
  - Granger causality analysis (optional)
  - Sector contagion patterns
  
- ✅ **Comprehensive Evaluation Script** (`evaluate_vortex.py`)
  - Generates N synthetic scenarios
  - Runs all metric tests
  - Compares against masterplan targets
  - Outputs JSON results with pass/fail status

#### 6. Visualizations
- ✅ **Visualization Suite** (`visualize_results.py`)
  - **Fan Charts**: Normal vs crisis scenario distributions with percentile bands
  - **Adjacency Heatmaps**: Empirical vs learned vs difference
  - **P&L Distributions**: Momentum strategy with Sharpe ratio
  - High-resolution PNG output (300 DPI)

### Hyperparameters (As Per Masterplan)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| T | 60 days | Window length (3 months) |
| n_stocks | 50 | NIFTY-50 constituents |
| latent_dim | 64 | Latent SDE dimension |
| proj_dim | 32 | Contrastive projection space |
| hidden_dim | 64 | Network hidden layers |
| gat_heads | 4 | GAT attention heads |
| tau | 0.07 | SupCon temperature |
| batch_size | 32 | Training batch size |
| lr | 0.001 | Learning rate |
| max_epochs | 200 | Training epochs |
| gradient_clip | 1.0 | Max gradient norm (critical for SDEs) |
| lam_rec | 1.0 | Reconstruction loss weight |
| lam_graph | 0.1 | Graph consistency weight |
| lam_con | 0.1 | Contrastive loss weight |

### Evaluation Targets

| Metric | Target | Test |
|--------|--------|------|
| Kurtosis | > 3.0 | Fat tails |
| ACF(r²) | > 0.05 | Volatility clustering |
| Corr Error | < 2.5 | Frobenius norm |
| Discriminative | < 0.60 | GBC accuracy |
| Crisis Boost | > 50% | Correlation increase |
| CVaR Ratio | > 3x | Risk measure |

## File Structure

### New Files Created
```
train_vortex.py              # Main training script (NEW)
evaluate_vortex.py           # Comprehensive evaluation (NEW)
visualize_results.py         # Figure generation (NEW)
IMPLEMENTATION_SUMMARY.md    # This file (NEW)

models/
├── gat_encoder.py           # UPDATED: Added DynamicGATEncoder
├── neural_sde.py            # UPDATED: Added GraphConditionedSDE + LatentSDEModel
└── contrastive.py           # UPDATED: Added SupConLoss + ProjectionHead

training/
├── vortex_model.py          # NEW: PyTorch Lightning module
├── losses.py                # UPDATED: Added all loss components
└── config.yaml              # UPDATED: Masterplan hyperparameters

eval/
├── statistical.py           # NEW: Statistical fidelity tests
├── discriminative.py        # NEW: Discriminative score test
└── contagion.py             # NEW: Contagion + CVaR tests

sandbox/
└── circuit_filter.py        # NEW: NSE circuit breaker enforcement

requirements.txt             # UPDATED: Added all dependencies
README.md                    # UPDATED: Complete VORTEX-AI documentation
```

### Dependencies Added
- `torchsde>=0.2.5` - Neural SDE solver
- `pytorch-lightning>=2.0.0` - Training infrastructure
- `wandb>=0.15.0` - Experiment tracking
- `sdmetrics>=0.10.0` - Evaluation metrics
- `plotly>=5.14.0` - Visualizations
- `streamlit>=1.25.0` - Dashboard (optional)
- `backtesting>=0.3.3` - Strategy testing (optional)

## Usage Examples

### 1. Complete Training Pipeline
```bash
# Prepare data
python data/download.py
python data/preprocess.py
python -m src.data_processor
python check_pipeline.py

# Train model
python train_vortex.py --config training/config.yaml

# Evaluate
python evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt

# Visualize
python visualize_results.py --checkpoint models/checkpoints/best_vortex.ckpt
```

### 2. Custom Training
```bash
# Override hyperparameters
python train_vortex.py \
  --batch_size 16 \
  --max_epochs 100 \
  --lam_con 0.2 \
  --lam_graph 0.15 \
  --learning_rate 0.0005

# Train on CPU
python train_vortex.py --accelerator cpu

# Resume training
python train_vortex.py --resume models/checkpoints/last.ckpt
```

### 3. Evaluation Options
```bash
# Generate 500 scenarios
python evaluate_vortex.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --n_scenarios 500

# Include Granger causality (slow)
python evaluate_vortex.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --test_granger

# Custom output
python evaluate_vortex.py \
  --checkpoint models/checkpoints/best_vortex.ckpt \
  --output results/my_evaluation.json
```

## Key Implementation Decisions

### 1. Why PyTorch Lightning?
- Simplifies distributed training
- Built-in checkpointing and logging
- Cleaner code organization
- Easy integration with wandb

### 2. Why torchsde over custom implementation?
- Google Research maintained
- GPU-accelerated
- Multiple solvers (Euler, Milstein, SRK)
- Proper gradient computation through SDE

### 3. Node Feature Engineering
6-dimensional features per stock:
1. Mean return over window
2. Absolute mean return (volatility)
3. Volume proxy (std deviation)
4. Sector encoding (placeholder)
5. Distance to upper circuit band
6. Distance to lower circuit band

### 4. Dataset Organization
- Windows: (N_windows, T=60, n_stocks=50)
- Regimes: (N_windows,) - binary labels
- Adjacency: (N_windows, n_stocks, n_stocks)
- All saved as `.npy` for fast loading

## Testing Checklist

Before publishing, verify:

- [ ] Data pipeline runs without errors
- [ ] Training completes at least 10 epochs
- [ ] Loss curves are decreasing
- [ ] No NaN/Inf in losses or gradients
- [ ] Evaluation metrics are computed correctly
- [ ] Visualizations are generated
- [ ] All imports resolve correctly
- [ ] Requirements.txt is complete
- [ ] README instructions are accurate

## Known Limitations

1. **Sector encoding**: Currently a placeholder (uniform). Should use actual sector mappings.
2. **Stock tickers**: Hardcoded NIFTY-50 list may become outdated.
3. **Volume data**: Using std as proxy; should integrate actual volume if available.
4. **Granger causality**: Computationally expensive; optional in evaluation.
5. **Circuit breaker bands**: Simplified mapping; actual bands may vary by stock.

## Future Extensions

1. **Reflected SDE** (R-NSDE): Native circuit breaker enforcement in SDE geometry
2. **Topologically-Correlated Diffusion**: Inject adjacency directly into diffusion matrix
3. **Path-Signature Contrastive Loss**: Use rough path signatures instead of mean pooling
4. **Riemannian SDE**: Operate directly on SPD manifold for covariance matrices
5. **No-Arbitrage Loss**: Add HJB residual penalty (lam_na currently 0.0)

## Ablation Studies (Recommended)

To validate contributions, train these variants:

1. **No-GAT**: Replace DynamicGATEncoder with identity/MLP
2. **No-SupCon**: Set lam_con=0.0
3. **Static-Graph**: Freeze adjacency to initial correlation
4. **No-Circuit-Filter**: Remove band clipping

Expected results:
- No-GAT: Worse contagion metrics
- No-SupCon: Higher discriminative score (regime blur)
- Static-Graph: Poor Granger causality preservation
- No-Circuit-Filter: Violations of NSE rules

## Baseline Comparisons

For paper submission, compare against:

1. **TimeGAN** (Yoon et al., NeurIPS 2019)
2. **QuantGAN** (Wiese et al., 2020)
3. **SigCWGAN** (Signature Conditional Wasserstein GAN)
4. **FTS-Diffusion** (Denoising diffusion for time series)

Target performance (from masterplan):

| Model | Disc. Score | Kurtosis | Corr Error | Crisis Boost |
|-------|-------------|----------|------------|--------------|
| TimeGAN | ~0.65 | ~2.8 | ~4.2 | ~20% |
| QuantGAN | ~0.60 | ~3.1 | ~3.8 | ~30% |
| **CG-NSDE** | **<0.55** | **>3.5** | **<2.5** | **>50%** |

## Publication Roadmap

### Tier 1 Target: ICAIF 2026 (ACM AI in Finance)
- Submission deadline: ~August 2026
- 8-page conference format
- Explicitly calls for generative AI + synthetic data

### Tier 2 Targets:
- AAAI 2027 (~August 2026 deadline)
- ICLR 2027 Workshop Track
- SSRN Preprint (rolling submission)

### Journal Targets:
- Journal of Financial Data Science (rolling)
- Quantitative Finance (Taylor & Francis)

## Contact & Contribution

For questions, bug reports, or contributions, please:
1. Open an issue on GitHub
2. Submit a pull request with tests
3. Contact the maintainers

## License

MIT License - See LICENSE file for details

---

**Implementation completed**: [Current Date]
**Framework version**: 1.0
**Masterplan reference**: VORTEX-AI-CG-NSDE-Masterplan.docx + gemini major project.pdf
