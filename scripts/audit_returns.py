"""Stage 2 post-rebuild audit: asserts on cleaned universe + crisis episode list."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
print("universe N:", px.shape[1])
print("rows:", px.shape[0], "from", px.index.min().date(), "to", px.index.max().date())

# A1: no all-NaN column
allnan = px.columns[px.isna().all()].tolist()
print("all-NaN columns:", allnan)
assert not allnan, "all-NaN column remains"

r = np.log(px).diff().iloc[1:]
print("NaN in returns:", int(r.isna().sum().sum()))
assert int(r.isna().sum().sum()) == 0, "NaN in returns"

# A2: no zero-std column
zstd = r.columns[r.std() == 0].tolist()
print("zero-std columns:", zstd)
assert not zstd, "zero-std column remains"


def max_zero_run(c):
    z = (c.fillna(0) == 0).astype(int)
    return int((z.groupby((z != z.shift()).cumsum()).cumsum() * z).max())


zr = r.apply(max_zero_run)
print("max zero run overall:", int(zr.max()))
print(zr.sort_values(ascending=False).head(5).to_string())
assert int(zr.max()) <= 5, "zero run over 5 days"

# A3: extremes only on documented genuine dates
WHITELIST = {
    ("2020-03-23", "AXISBANK.NS"),  # COVID crash Monday, NIFTY down about 13pct, genuine
    ("2020-03-26", "INDUSINDBK.NS"),  # relief rally, genuine March 2020 volatility
}
s = r.stack()
big = s[s.abs() > 0.30]
print("count |r| > 0.30:", len(big))
unlisted = []
for (d, t), v in big.items():
    key = (d.strftime("%Y-%m-%d"), t)
    flag = "WHITELISTED" if key in WHITELIST else "UNLISTED"
    print(d.date(), t, round(float(v), 4), flag)
    if key not in WHITELIST:
        unlisted.append(key)
assert not unlisted, f"unlisted extremes: {unlisted}"

# Kurtosis
k = r.apply(lambda c: kurtosis(c.dropna()))
print("per-stock kurtosis median:", round(float(k.median()), 2), "max:", round(float(k.max()), 2))

# Crisis episodes from window_regimes_v2
wr = np.load("data/raw/window_regimes_v2.npy")
print("windows:", len(wr), "crisis fraction:", round(float(wr.mean()), 4))
dates = r.index
# v2 label j corresponds to returns row 59 + j (last day of window j)
wdates = [dates[59 + j].date().isoformat() for j in range(len(wr))]
episodes = []
start = None
prev = None
for j, lab in enumerate(wr):
    if lab == 1 and start is None:
        start = j
    if lab == 0 and start is not None:
        episodes.append((start, prev))
        start = None
    if lab == 1:
        prev = j
if start is not None:
    episodes.append((start, prev))
print("distinct crisis episodes:", len(episodes))
for a, b in episodes:
    print(f"  {wdates[a]} to {wdates[b]} ({b - a + 1} windows)")
print("AUDIT OK")
