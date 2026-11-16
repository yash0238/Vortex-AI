# Assignment: three neural baselines

Files you own: crisisbench/generators/neural_timevae.py, crisisbench/generators/neural_quantgan.py,
crisisbench/generators/neural_timegan.py, tests/test_neural_baselines.py

Faithful OWN PyTorch reimplementations (class attribute impl_source = "own PyTorch reimplementation"; the original
code bases need TensorFlow and old Python). Registry expects classes TimeVAE, QuantGAN, TimeGAN in those modules,
family = "neural".

1. TimeVAE (Desai et al. 2021, "TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation"),
   BASE architecture: Conv1D encoder, dense latent, dense + transposed-Conv1D decoder (no trend/seasonality
   blocks). Inputs: per-asset z-scored windows (scaler from training days only: days where data.val_mask is False).
   Loss: reconstruction (sum of squared errors with the paper's reconstruction weight) + KL. Early stopping on
   validation ELBO using data.split_windows(). sample(): z ~ N(0, I), decoder, inverse scale.

2. QuantGAN (Wiese et al. 2020, "Quant GANs: Deep Generation of Financial Time Series"): TCN generator and TCN
   discriminator (dilated causal convolutions with skip connections, receptive field about 127); preprocessing =
   standardise, inverse Lambert-W Gaussianisation (estimate delta per asset, e.g. IGMM or moment matching; exact
   inverse at generation), standardise. Multi-asset extension: the generator maps a (T + RF - 1, noise_dim)
   Gaussian noise path to N outputs per day; the discriminator sees (T, N) windows. WGAN-GP or the paper's
   weight-clipping WGAN: state which and why. Fixed iteration budget; log the losses in the fit info.

3. TimeGAN (Yoon, Jarrett, van der Schaar, NeurIPS 2019): embedder, recovery, generator, supervisor,
   discriminator (GRUs); the paper's three phases (1 autoencoding; 2 supervised loss; 3 joint training: generator
   twice per discriminator step, discriminator updated only when its loss > 0.15, gamma = 1, moment terms V1/V2,
   eta = 10). Min-max scaling to [0,1] per asset (from training windows), sigmoid outputs, inverse scaling at
   generation. N = 47 features, so hidden_dim 64, 3 layers, noise dim = N. The paper's 50,000 iterations per phase
   are impossible here: choose the largest per-phase iteration count that fits the budget and document it.

Budget: each DEFAULT configuration must fit in at most 20 minutes on this CPU with about 3400 training windows of
shape (60, 47) and 4 torch threads. Expose "epochs" / "iters" hyperparameters for smoke tests. Each model
implements save(path) (state_dict + preprocessing). The fit info dict should include the loss history.

Tests (tiny budgets): shapes, finiteness, determinism given seed for all three; Lambert-W round trip error below
1e-6; TimeGAN scaler round trip and phase-1 loss decreasing; TimeVAE generated per-asset std within a factor 3 of
real after a short training.
