"""Stage 3c: crisis counts for scheme (a) chronological 70/15/15 + purge, (b) episode blocks."""
import numpy as np
import pandas as pd

v2 = np.load("data/raw/window_regimes_v2.npy")
px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
px = px.loc["2010-11-08":]
r = np.log(px / px.shift(1)).dropna()
N = len(v2)
wdates = pd.DatetimeIndex([r.index[59 + j] for j in range(N)])
print("windows:", N, "crisis:", int(v2.sum()), f"({float(v2.mean()):.2%})")
print("span:", wdates[0].date(), "to", wdates[-1].date())

GAP = 60
# (a) chronological 70/15/15 with purge gaps
n_train = int(0.70 * N)
tr = (0, n_train)
va = (n_train + GAP, n_train + GAP + int(0.15 * N))
te = (va[1] + GAP, N)
print("\n(a) chronological:")
for name, (a, b) in [("train", tr), ("val", va), ("test", te)]:
    seg = v2[a:b]
    print(f"  {name} [{a}:{b}] n={len(seg)} crisis={int(seg.sum())} ({float(seg.mean()):.2%}) {wdates[a].date()} to {wdates[b-1].date()}")

# (b) episode blocks: val 2018-2019, test 2022-2024, train rest, 60-day purges at edges
val_mask = (wdates.year == 2018) | (wdates.year == 2019)
test_mask = wdates.year >= 2022
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]
purge = set()
for edge in [val_idx[0], val_idx[-1], test_idx[0]]:
    for j in range(edge - GAP, edge + GAP):
        purge.add(j)
train_idx = np.array([j for j in range(N) if j not in set(val_idx) | set(test_idx) | purge])
print("\n(b) episode blocks (val=2018-19, test=2022+, 60-window purge at edges):")
for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    seg = v2[idx]
    print(f"  {name} n={len(idx)} crisis={int(seg.sum())} ({float(seg.mean()):.2%}) {wdates[idx[0]].date()} to {wdates[idx[-1]].date()}")
print("\nrule: use (a) if val and test each have >= 40 crisis windows")
