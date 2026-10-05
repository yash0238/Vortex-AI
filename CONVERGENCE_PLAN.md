# VORTEX-AI Convergence Plan (2026-10-03)

Goal: 0/6 to 4/6+ quality gates + ablation table + baseline comparison.
Working branch: `fix/stabilize`. GPU: RTX 3050 4GB (laptop).

## 1. Reference coverage (all verified present and read)

PDFs in `C:\Users\myind\Documents\GitHub\vortex-ai\*.pdf`:

| File | Identity (first-page verified) |
|---|---|
| `164_graph_attention_networks.pdf` (12pp) | Velickovic et al., GAT, ICLR 2018 |
| `2561_score_based_generative_modelin.pdf` (36pp) | Song et al., Score-SDE, ICLR 2021 |
| `2603.24236v1.pdf` (5pp) | S3G stock state-space graph |
| `2605.27113v1.pdf` + ` (1).pdf` (23pp, duplicates) | Masi et al., CoMeTS-GAN GAN-Diffusion |
| `NeurIPS-2019-...-Paper.pdf` + `NIPS2019_TGAN_Main.pdf` (11pp each) | Yoon et al., TimeGAN (same paper, both copies read) |
| `ai-07-00060-v2.pdf` (23pp) | LFTD latent-diffusion Transformer (MDPI AI 2026) |

GitHub repos (all fetched live, READMEs read):

| Repo | Confirmed content |
|---|---|
| giuseppemasi99/COMETS-GAN (179 commits, 3 stars) | Correlated stock-market generation, src/ + requirements |
| ZhuoHan1998/Diffusion-Models-In-Finance (26 commits) | Curated taxonomy incl. 2605.27113, CoFinDiff, InterDiff, score_sde |
| jsyoon0823/TimeGAN (1.1k stars, 325 forks) | timegan.py, main_timegan.py (gru/24/3/50000/128), metrics/ |
| KseniaKingsep/quantgan (27 stars) | QuantGAN.ipynb, TemporalBlock/TCN/SP500Dataset, Colab |
| PetarV-/GAT (3.6k stars) | TF reference, sparse version batch-size-1 only, recommends PyG/DGL |
| ebrahimpichka/GAT-pt (51 stars) | PyTorch L1 K=8/F'=8 ELU, L2 single-head; train.py lr 0.005, l2 5e-4, dropout 0.6 |
| yang-song/score_sde (1.8k stars) | sde_lib + Predictor/Corrector registry, Euler-Maruyama default, VE sigma_max = max pairwise distance, Langevin snr 0.05-0.2 |

## 2. Why metrics fail (reference-grounded, file:line checked)

- `training/losses.py:94` recon is endpoint MSE only. No TimeGAN supervised loss (eta=10 closed-loop): temporal conditionals unenforced.
- `training/losses.py:105` matches attention weights to Pearson by mean. CoMeTS-GAN proves correlation must be computed on GENERATED returns. Our `training/vortex_model.py` `correlation_loss` (per-regime pooled Fro vs `data/raw/regime_corr.npz`: crisis 0.4141 vs normal 0.2308) is the correct object; attention-MSE stays diagnostic only.
- `models/contrastive.py:43` SupCon tau=0.07 with uniform negatives on random early projections: disc 1.0. TimeGAN/LFTD require a pretrained embedding space first.
- `models/neural_sde.py:60` sigma_max=3.0 learned from scratch through 60 Euler steps. Score-SDE says fix the forward and learn the score; at minimum tighten to 0.5-1.0 and pretrain recon first (7.3a evidence: gen std 0.346 vs real 0.019 when diffusion is unconstrained).
- `models/gat_encoder.py:94,138` per-sample loop + dense_to_sparse, dropout 0.1 vs paper 0.6 + L2 0.0005.

## 3. CUDA workflow (mandatory, 4GB GPU)

Pre-flight every session:

```powershell
nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv
py -3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Rules: `--accelerator cuda --devices 1 --num_workers 0 --batch_size 8`. `gpu` is NOT a valid choice
(choices: auto, cpu, cuda, mps). `num_workers>0` crashes on win32 spawn (pickle truncation +
CombinedLoader teardown error, seen in logs). One training job at a time. Log every run:
`2>&1 | Tee-Object -FilePath runs/<tag>.log`. Workdir is always `Vortex-AI/`
(`Set-Location` first; bare `runs\` resolves to the wrong directory otherwise).

Smoke before long runs (batch 8, 1 epoch). After each run check: `torch.isfinite`,
`zs.std()`, `train/corr_raw`, `train/nll`, `train/rho`, `train/kappa`.

## 4. Staged plan with gates

- Stage 1 recon-only (20 epochs, cuda, batch 8): emission=point, lam_con=0, lam_corr=0,
  lam_graph=0.3, lambda_styl=0, beta_max=0, sde_sigma_max=0.5, GAT dropout 0.6 + L2 0.0005.
  DONE when val/total decreases every epoch and gen std within 2x real.
- Stage 2 corr (R3 verdict): R3 = L_corr ramp 0 to 100/5 epochs + vol-dyn NLL.
  If corr_raw < 0.3: keep lam_corr 100-200, demote lam_graph to 0.1.
  (7.3b V3 boost 9.9% vs static -0.2% proves graph conditioning adds regime structure.)
  DONE when corr error < 10 and boost > 15%.
- Stage 3 regime: SupCon tau=0.1, warmup 30, stratified batches, hard-negative weighting in
  `models/contrastive.py:43`, TimeGAN-style supervised weight 10x. DONE when disc < 0.8, then < 0.6.
- Stage 4 vol/tails: emission=hetero + vol_dyn, lambda_styl 0.5 to 1.0.
  DONE when ACF(r2) > 0.05.
- Stage 5 eval: 4 ablations serialized x 20-50 epochs; fallback baselines iid + t-PPCA + GARCH;
  TimeGAN/QuantGAN stay BLOCKED (no py3.13 wheel, empty submodules), reported honestly.
  DONE: ablation table + eval_final.json (n=500) + 3 figures at 300 DPI.

Minimum viable: 3/6 gates (corr < 10, disc < 0.8, boost > 15%) + ablation + fallback tables.
Lane 1 (L_corr + vol_dyn, flags default OFF) and R3 are done; 7.3b reproduces exactly with flags
off (val NLL 57.493 vs 57.50). GAT/window batching optimization is deferred, not blocking.
