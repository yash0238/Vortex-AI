"""Stage 3a/3b read-only probe: normalization + label alignment."""
import numpy as np
import pandas as pd

w = np.load("data/raw/windows.npy")
r = np.load("data/raw/returns.npy")
reg = np.load("data/raw/regimes.npy")
wr = np.load("data/raw/window_regimes.npy")
v2 = np.load("data/raw/window_regimes_v2.npy")
print("shapes:", w.shape, r.shape, reg.shape, wr.shape, v2.shape)

# (a) normalization: raw log returns are ~0.01 scale; per-window z-scoring would give std 1
print("windows global mean/std:", round(float(w.mean()), 6), round(float(w.std()), 6))
pw_mean = w.mean(axis=(1, 2))
pw_std = w.std(axis=(1, 2))
print("per-window mean: avg", round(float(pw_mean.mean()), 6), "std", round(float(pw_mean.std()), 6))
print("per-window std: avg", round(float(pw_std.mean()), 6), "min", round(float(pw_std.min()), 6))
ps_std = r.std(axis=0)
print("per-stock std: median", round(float(np.median(ps_std)), 6), "min", round(float(ps_std.min()), 6))
print("=> windows.npy is RAW log returns (no normalization)" if abs(float(w.std()) - 0.02) < 0.02 else "=> SCALED?")

# (b) which regimes.npy: recompute composite (preprocess) vs vol-only (data_processor)
px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
px = px.loc["2010-11-08":]
ret = np.log(px / px.shift(1)).dropna()
print("csv-derived returns shape:", ret.shape, "npy returns shape:", r.shape)
print("returns match csv:", bool(np.allclose(ret.values, r, atol=1e-6)))
rv = ret.values
realized_vol = pd.DataFrame(rv).std(axis=1).rolling(20).mean()
thr = np.percentile(realized_vol.dropna(), 85)
vol_only = ((realized_vol > thr).astype(int).fillna(0).values).astype(int)
print("vol-only daily crisis frac:", round(float(vol_only.mean()), 4), "npy regimes frac:", round(float(reg.mean()), 4))
print("vol-only matches regimes.npy:", bool((vol_only == reg).all()))
# composite
roll5 = pd.DataFrame(rv).rolling(5).sum()
dd = (roll5 < -0.05).mean(axis=1)
cp = (pd.DataFrame(rv).abs() > 0.095).mean(axis=1)
q99 = realized_vol.quantile(0.99)
score = (0.5 * realized_vol / q99 + 0.3 * dd + 0.2 * cp).fillna(0)
cth = np.percentile(score, 85)
comp = ((score >= cth).astype(int).values).astype(int)
print("composite daily crisis frac:", round(float(comp.mean()), 4))
print("composite matches regimes.npy:", bool((comp == reg).all()))

# (b2) v2 alignment: v2[j] should equal daily label of last day of window j
# data_processor windows: window k = values[k:k+60], last row index k+59
for j in [0, 100, 1000, 2000, 3425]:
    print(f"window {j}: v2={v2[j] if j < len(v2) else 'MISSING'}, daily[{59 + j}]={reg[59 + j]}, match={bool(v2[j] == reg[59 + j]) if j < len(v2) else False}")
print("last window 3426 daily label:", reg[59 + 3426], "(v2 has no entry)")
