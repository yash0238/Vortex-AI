import numpy as np, pandas as pd
from scipy.stats import kurtosis
px = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True)
r = np.log(px).diff().iloc[1:]
print("shape", r.shape, "NaN", int(r.isna().sum().sum()), "zero share", float((r == 0).mean().mean()))
s = r.stack()
print("TOP 30 |r|:")
for (d, t), v in s.abs().sort_values(ascending=False).head(30).items():
    print(d.date(), t, round(float(s[(d, t)]), 4))
print("per-stock kurtosis (median, max):",
      float(r.apply(lambda c: kurtosis(c.dropna())).median()),
      float(r.apply(lambda c: kurtosis(c.dropna())).max()))
def max_zero_run(c):
    z = (c.fillna(0) == 0).astype(int)
    return int((z.groupby((z != z.shift()).cumsum()).cumsum() * z).max())
print("longest zero run per stock (top 10):")
print(r.apply(max_zero_run).sort_values(ascending=False).head(10))
