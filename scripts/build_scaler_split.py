"""Stage 3c/3d: decide split, train-only day mask, scaler, train-fit threshold, relabel, rebuild z-space features."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from data.preprocess import label_regimes

px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
px = px.loc["2010-11-08":]
px = px.apply(pd.to_numeric, errors="coerce").ffill().bfill()
ret = np.log(px / px.shift(1)).dropna(how="all").fillna(0.0)
R = ret.values.astype(np.float64)
dates = ret.index
N = len(R) - 60 + 1
wdates = pd.DatetimeIndex([dates[60 - 1 + j] for j in range(N)])
print("days:", len(R), "windows:", N)

# scheme (b) fixed blocks
val_mask = (wdates.year == 2018) | (wdates.year == 2019)
test_mask = wdates.year >= 2022
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]
GAP = 60
purge = set()
for edge in [val_idx[0], val_idx[-1], test_idx[0], test_idx[-1]]:
    for j in range(edge - GAP, edge + GAP):
        purge.add(j)
train_idx = np.array(sorted(set(range(N)) - set(val_idx.tolist()) - set(test_idx.tolist()) - purge))
print(f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} purged={len(purge)}")

# train-only days: covered by train windows, by zero val/test windows
cover_tr = np.zeros(len(R), dtype=bool)
cover_vt = np.zeros(len(R), dtype=bool)
for j in train_idx:
    cover_tr[j:j + 60] = True
for j in list(val_idx) + list(test_idx):
    cover_vt[j:j + 60] = True
train_days = cover_tr & ~cover_vt
print(f"train-only days: {int(train_days.sum())} / {len(R)}")

# scaler on train-only daily returns
mu = R[train_days].mean(axis=0)
sd = R[train_days].std(axis=0)
zero_std = np.where(sd <= 0)[0]
if len(zero_std):
    print("zero-variance train-only columns; using unit scale:", px.columns[zero_std].tolist())
    sd[zero_std] = 1.0
np.savez("data/raw/scaler.npz", mean=mu.astype(np.float32), std=sd.astype(np.float32),
         tickers=np.array(px.columns.tolist()))
print("scaler saved. mean abs:", round(float(np.abs(mu).mean()), 6), "std median:", round(float(np.median(sd)), 6))

# composite scores everywhere, threshold fit on train-only days
rv = pd.DataFrame(R)
realized_vol = rv.std(axis=1).rolling(20).mean()
roll5 = rv.rolling(5).sum()
dd = (roll5 < -0.05).mean(axis=1)
cp = (rv.abs() > 0.095).mean(axis=1)
q99 = realized_vol.quantile(0.99)
score = (0.5 * realized_vol / q99 + 0.3 * dd + 0.2 * cp).fillna(0)
thr = float(np.percentile(score.values[train_days], 85))
daily = (score.values >= thr).astype(np.int64)
print(f"train-fit threshold: {thr:.5f} daily crisis frac: {float(daily.mean()):.4f}")
np.save("data/raw/regimes.npy", daily)
v2 = np.array([daily[k + 59] for k in range(N)], dtype=np.int64)
np.save("data/raw/window_regimes_v2.npy", v2)
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    seg = v2[idx]
    print(f"  {name}: n={len(idx)} crisis={int(seg.sum())} ({float(seg.mean()):.2%})")
np.savez("data/raw/split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
         scheme=np.array(["episode-blocks-val2018-19-test2022plus-purge60"]))

# z-space node features: standardize cols r, |r|, vol with train stats
vol_df = pd.read_csv("data/raw/nifty50_volume.csv", index_col=0, parse_dates=True)
vol_df = vol_df.loc[px.index][px.columns].values.astype(np.float64)[-len(R):]
assert vol_df.shape == R.shape
mu_abs = np.abs(R[train_days]).mean(axis=0)
sd_abs = np.abs(R[train_days]).std(axis=0)
mu_vol = vol_df[train_days].mean(axis=0)
sd_vol = vol_df[train_days].std(axis=0) + 1e-8
from data.node_features import NIFTY50_SECTORS, NUM_SECTORS, _band_limit
tickers = px.columns.tolist()
T, NS = R.shape
F = np.zeros((T, NS, 6), dtype=np.float32)
F[:, :, 0] = (R - mu) / sd
F[:, :, 1] = (np.abs(R) - mu_abs) / sd_abs
F[:, :, 2] = (vol_df - mu_vol) / sd_vol
for j, tk in enumerate(tickers):
    F[:, j, 3] = NIFTY50_SECTORS.get(tk, 0) / NUM_SECTORS
    b = _band_limit(tk)
    F[:, j, 4] = b
    F[:, j, 5] = b - np.abs(R[:, j])
n_win = T - 60 + 1
out = np.stack([F[i:i + 60] for i in range(n_win)]).astype(np.float32)
assert out.shape[0] == N
np.save("data/raw/window_node_features.npy", out)
print("z-space window_node_features saved:", out.shape)
print("col0 global std (should be ~1):", round(float(out[:, :, :, 0].std()), 4))
print("DONE")
