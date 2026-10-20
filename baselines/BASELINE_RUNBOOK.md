# Baseline Setup Runbook (TimeGAN + QuantGAN)

## Scope

This runbook covers your assigned deliverables:

- Environment setup
- NIFTY-50 dataset prep (2010-2024)
- TimeGAN baseline
- QuantGAN baseline
- Baseline comparison table

## 1) Prepare repo data

From project root:

```powershell
python data/download.py
python data/preprocess.py
python baselines/prepare_baseline_data.py
python inspect_data.py
```

Expected baseline files under `baselines/data`:

- `timegan_windows.npy`
- `timegan_window_regimes.npy`
- `quantgan_market_returns.npy`
- `quantgan_market_windows.npy`
- `quantgan_market_returns.csv`
- `baseline_data_metadata.json`

## 2) Clone external baseline repos

From project root:

```powershell
mkdir baselines\external
cd baselines\external
git clone https://github.com/jsyoon0823/TimeGAN.git
git clone https://github.com/ICascha/QuantGANs-replication.git
```

## 2.1) Environment compatibility notes

- The original TimeGAN repo uses TensorFlow 1.15 and does not support Python 3.12.
- Use a dedicated Python 3.7 or 3.8 environment for TimeGAN.
- QuantGAN replication uses TensorFlow 2.x style Keras and is better on Python 3.10.
- If conda is not available, install these Python versions and use `py -3.7 -m venv` and `py -3.10 -m venv` equivalents.

## 3) TimeGAN baseline

In `baselines/external/TimeGAN`:

```powershell
conda create -n timegan_py37 python=3.7 -y
conda activate timegan_py37
pip install -r requirements.txt
```

Data mapping for TimeGAN:

- Input sequence array: `../../data/timegan_windows.npy`
- Sequence length: 60
- Feature dimension: 50

Run one short sanity training first, then full training.
Save:

- Checkpoint(s)
- Generated synthetic sequences
- Final config values

Evaluate the completed NIFTY-50 run:

```powershell
python baselines/evaluate_timegan.py
```

This writes `baselines/results/timegan_full/evaluation_metrics.json` and
`baselines/results/timegan_full/paper_vs_nifty50_comparison.csv`. The comparison
labels the original paper's dataset-specific RNN metrics separately from the
NIFTY-50 surrogate metrics; they should not be interpreted as identical benchmarks.

## 4) QuantGAN baseline

In `baselines/external/QuantGANs-replication`:

```powershell
conda create -n quantgan_py310 python=3.10 -y
conda activate quantgan_py310
pip install tensorflow==2.12.0 tensorflow-addons==0.20.0 yfinance pandas-datareader scipy scikit-learn matplotlib numpy==1.23.5
```

Data mapping for QuantGAN baseline:

- Start with 1D market return series: `../../data/quantgan_market_returns.csv`
- Window length: 60

The runnable wrapper is `baselines/run_quantgan.py`. It uses the replication's
TCN generator/discriminator, Gaussian noise with three features, and the paper's
learning-rate asymmetry (`1e-4` discriminator, `3e-4` generator). Because this
repository uses 60-step windows rather than the paper's S&P 500 setup, the smoke
and full configs use dilations `[1, 2, 4, 8]` so the receptive field fits the data.

Run a short sanity training first:

```powershell
py -3.10 baselines/run_quantgan.py --config baselines/config_quantgan_smoke.json
```

Then run the full baseline:

```powershell
py -3.10 baselines/run_quantgan.py --config baselines/config_quantgan_full.json
```

Run these commands from a Python 3.10 environment with TensorFlow and
TensorFlow Addons installed. The runner writes generated windows, generator and
discriminator H5 weights, and `training_metadata.json` under the configured output directory.

Save:

- Checkpoint(s)
- Generated synthetic return series
- Final config values

## 5) Record comparison

Fill `baselines/baseline_comparison.csv` after each run.

Minimum fields to track:

- model_name
- dataset_name
- window
- epochs
- batch_size
- learning_rate
- train_time_min
- generated_samples
- mse_mean
- acf_lag1_real
- acf_lag1_synth
- notes

## 6) Acceptance checklist

- Environment runs for both repos
- Baseline datasets are generated
- TimeGAN trained and generated samples saved
- QuantGAN trained and generated samples saved
- Comparison CSV updated
