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

Run one short sanity training first, then full training.
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
