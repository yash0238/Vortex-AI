"""Stage 3b: composite daily regimes (train-agnostic part) + full-length v2 + verify 5 random windows from CSV."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from data.preprocess import label_regimes, compute_log_returns

px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
ret = compute_log_returns(px)  # ffill/bfill, log, fillna 0 -> (3487, 47)
print("returns shape:", ret.shape)
# drop first row if it is all zeros from leading NaN (log diff needs prev day)
if (ret.iloc[0] == 0).all():
    ret = ret.iloc[1:]
    print("dropped leading all-zero row ->", ret.shape)
daily = label_regimes(ret, vol_window=20, crisis_pct=85.0)
np.save("data/raw/regimes.npy", daily.values)
print("saved regimes.npy:", daily.values.shape, "crisis frac:", round(float(daily.mean()), 4))

# v2: one label per data_processor window (window k = rows [k, k+60), last row k+59)
W = 60
n_win = len(ret) - W + 1
lab = daily.values
v2 = np.array([lab[k + W - 1] for k in range(n_win)], dtype=np.int64)
np.save("data/raw/window_regimes_v2.npy", v2)
print("saved window_regimes_v2.npy:", v2.shape, "crisis frac:", round(float(v2.mean()), 4))

# verify 5 random windows by independent recompute from CSV prices
rng = np.random.default_rng(7)
for k in sorted(rng.integers(0, n_win, 5).tolist()):
    print(f"window {k}: v2={v2[k]}, daily[{k + W - 1}]={lab[k + W - 1]}, match={bool(v2[k] == lab[k + W - 1])}")
assert len(v2) == 3427, f"v2 length {len(v2)} != 3427"
print("ALIGN OK")
