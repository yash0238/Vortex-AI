"""Stage 2 universe fix: drop late-listed, repair Tata Motors via TMPV.NS, fix NESTLEIND tick, cut pre-COALINDIA history."""
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
close = pd.read_csv(RAW / "nifty50_close.csv", index_col=0, parse_dates=True)
vol = pd.read_csv(RAW / "nifty50_volume.csv", index_col=0, parse_dates=True)
log = []

# 1. Fetch TMPV.NS (Tata Motors demerger successor) full history
import yfinance as yf
df = yf.download(["TMPV.NS"], start="2010-01-01", end="2024-12-31", auto_adjust=True, progress=False)
tmp_close = df["Close"]["TMPV.NS"]
tmp_vol = df["Volume"]["TMPV.NS"]
r = np.log(tmp_close / tmp_close.shift(1)).dropna()
assert len(tmp_close.dropna()) == len(tmp_close), "TMPV has gaps"
assert tmp_close.index.min().date().isoformat() <= "2010-01-06", "TMPV history not continuous from 2010"
assert float(r.abs().max()) < 0.30, "TMPV has |r| above 0.30"
log.append(("TMPV.NS", "all", "accepted as TATAMOTORS successor", f"days={len(tmp_close)}, max|r|={float(r.abs().max()):.4f}"))

# 2. NESTLEIND 2010-01-08 bad tick: replace that day's return with cross-sectional median
px = close["NESTLEIND.NS"]
bad_day = "2010-01-08"
prev = px.loc[:bad_day].dropna().iloc[-2]
med = float(np.log(close.loc[bad_day] / close.shift(1).loc[bad_day]).dropna().median())
close.loc[bad_day, "NESTLEIND.NS"] = float(prev) * float(np.exp(med))
log.append(("NESTLEIND.NS", bad_day, "bad tick replaced with cross-sectional median return", f"median={med:.5f}"))
flat = px.iloc[0:4]
log.append(("NESTLEIND.NS", "2010-01-04 to 2010-01-07", "stale flat prices 433.38 x4, excluded by start cut", "logged only"))

# 3. Universe surgery
for dead in ["LTI.NS", "HDFCLIFE.NS", "SBILIFE.NS"]:
    del close[dead]
    if dead in vol.columns:
        del vol[dead]
    log.append((dead, "all", "dropped, late listing, no full history", ""))
del close["TATAMOTORS.NS"]
if "TATAMOTORS.NS" in vol.columns:
    del vol["TATAMOTORS.NS"]
close["TMPV.NS"] = tmp_close
vol["TMPV.NS"] = tmp_vol
log.append(("TATAMOTORS.NS", "all", "delisted symbol replaced by TMPV.NS successor history", ""))

# 4. Start after COALINDIA listing
first_coal = close["COALINDIA.NS"].dropna().index.min()
start = first_coal + pd.Timedelta(days=1)
close = close.loc[start:]
vol = vol.loc[close.index]
log.append(("UNIVERSE", str(start.date()), "start cut after COALINDIA listing", f"N={close.shape[1]}"))

# 5. Guards before save
assert not close.isna().all().any(), "all-NaN column remains"
assert (close.std() > 0).all(), "zero-std column remains"
print("NaN remaining:", int(close.isna().sum().sum()))
close.to_csv(RAW / "nifty50_close.csv")
vol.to_csv(RAW / "nifty50_volume.csv")
pd.DataFrame(log, columns=["ticker", "date", "action", "detail"]).to_csv(RAW / "bad_ticks.csv", index=False)
print("saved:", close.shape, "bad_ticks rows:", len(log))
