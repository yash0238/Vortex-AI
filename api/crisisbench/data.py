"""Reproducible panel builder: yfinance -> cleaned daily log-return matrix.

    python -m crisisbench.data --market nse
    python -m crisisbench.data --market us
    python -m crisisbench.data --market kr
    python -m crisisbench.data --market nse_ext     (NSE panel extended to 2026-10-31)

Writes data/panels/<market>/{returns.npy, volume.npy, dates.csv, tickers.csv,
audit.json, audit.txt, sha256.txt}.  Raw vendor prices are cached under
data/panels/<market>/raw/ (git-ignored: Yahoo data must not be redistributed).

Cleaning rules are the same for every market and were fixed before any
generator was scored:
  1. Candidate list is fixed in MARKETS (order = priority).
  2. A ticker is kept only if its first valid price is on or before the
     market's start date + 5 trading days, it has < 2% missing days, its
     history runs to the end of the sample, and its price is never unchanged
     for more than 10 consecutive trading days (suspension).
  3. Keep the first `n_assets` surviving candidates.
  4. Forward-fill price gaps of at most 5 days; daily log returns.
  5. Bad-tick rule on |r| > 0.30:
       - spike that reverses by >= 80% within the next two days -> both days replaced by the
         cross-sectional median return (logged as BAD_TICK);
       - |r| > 0.50 without reversal -> treated as an unadjusted corporate action,
         replaced by the cross-sectional median (logged as CORP_ACTION);
       - otherwise kept and logged as WHITELISTED (a real move).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANELS = ROOT / "data" / "panels"

NSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "ITC.NS", "AXISBANK.NS", "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
    "POWERGRID.NS", "NTPC.NS", "M&M.NS", "TECHM.NS", "TMPV.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "BAJAJ-AUTO.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "ONGC.NS", "COALINDIA.NS",
    "BPCL.NS", "GRASIM.NS", "ADANIPORTS.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "HINDALCO.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "SHREECEM.NS", "UPL.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS",
    # late listings, expected to be dropped by rule 2:
    "SBILIFE.NS", "HDFCLIFE.NS", "LTIM.NS",
]
# Largest US companies (end-2024 order, author's fixed candidate list).
US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "BRK-B", "TSLA", "JPM", "LLY", "AVGO", "XOM", "UNH", "V", "MA",
    "COST", "HD", "PG", "WMT", "JNJ", "NFLX", "BAC", "CRM", "ORCL", "CVX", "KO", "MRK", "CSCO", "WFC", "ACN",
    "ADBE", "PEP", "LIN", "MCD", "AMD", "IBM", "DIS", "TMO", "ABT", "PM", "GE", "CAT", "GS", "ISRG", "QCOM",
    "TXN", "VZ", "INTU", "AXP", "MS", "CMCSA", "T", "PFE", "AMGN", "NEE", "RTX", "HON", "UNP", "LOW",
]
# KOSPI large caps (fixed candidate list).
KR = [
    "005930.KS", "000660.KS", "005380.KS", "000270.KS", "005490.KS", "035420.KS", "051910.KS", "006400.KS",
    "035720.KS", "012330.KS", "055550.KS", "105560.KS", "086790.KS", "066570.KS", "003550.KS", "017670.KS",
    "030200.KS", "015760.KS", "034730.KS", "096770.KS", "010950.KS", "009150.KS", "032830.KS", "000810.KS",
    "033780.KS", "090430.KS", "051900.KS", "011170.KS", "010130.KS", "009540.KS", "010140.KS", "034020.KS",
    "011200.KS", "003490.KS", "024110.KS", "036570.KS", "000720.KS", "004020.KS", "011780.KS", "021240.KS",
    "097950.KS", "001040.KS", "023530.KS", "004170.KS", "069960.KS", "000100.KS", "128940.KS", "006800.KS",
    "016360.KS", "005830.KS", "001450.KS", "042660.KS", "012450.KS", "000880.KS", "009830.KS", "078930.KS",
    "006360.KS", "002790.KS", "035250.KS", "030000.KS", "008770.KS", "005940.KS", "071050.KS", "029780.KS",
    "000120.KS", "086280.KS",
]

MARKETS = {
    "nse": dict(candidates=NSE, start="2010-11-04", end="2024-12-31", n_assets=47, label="NSE-47 (India)"),
    "nse_ext": dict(candidates=NSE, start="2010-11-04", end="2026-10-31", n_assets=47,
                    label="NSE-47 extended to Oct 2026"),
    "us": dict(candidates=US, start="2010-11-04", end="2024-12-31", n_assets=47, label="US large-cap 47"),
    "kr": dict(candidates=KR, start="2010-11-04", end="2024-12-31", n_assets=47, label="KOSPI large-cap 47"),
}


@dataclass
class Panel:
    market: str
    returns: np.ndarray          # (D, N) daily log returns, float64
    volume: np.ndarray           # (D, N) share volume (NaN -> 0)
    dates: pd.DatetimeIndex      # (D,)
    tickers: list[str]

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def windows(self, T: int = 60) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """All overlapping (T, N) windows; window i ends on dates[i + T - 1]."""
        v = np.lib.stride_tricks.sliding_window_view(self.returns, T, axis=0)  # (D-T+1, N, T)
        return np.ascontiguousarray(v.transpose(0, 2, 1)), self.dates[T - 1:]


def _download(tickers: list[str], start: str, end: str, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    cpath, vpath = cache / "close.csv", cache / "volume.csv"
    if cpath.exists() and vpath.exists():
        return (pd.read_csv(cpath, index_col=0, parse_dates=True),
                pd.read_csv(vpath, index_col=0, parse_dates=True))
    import yfinance as yf
    closes, vols = {}, {}
    for i in range(0, len(tickers), 20):          # small batches: fewer silent drops
        chunk = tickers[i:i + 20]
        df = yf.download(chunk, start="2010-01-01", end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False, group_by="column", threads=False)
        for tk in chunk:
            try:
                c = df["Close"][tk] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
                v = df["Volume"][tk] if isinstance(df.columns, pd.MultiIndex) else df["Volume"]
            except KeyError:
                continue
            if c.notna().sum() > 0:
                closes[tk], vols[tk] = c, v
    close, vol = pd.DataFrame(closes), pd.DataFrame(vols)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = pd.to_datetime(vol.index).tz_localize(None)
    close.to_csv(cpath)
    vol.to_csv(vpath)
    return close, vol


def _max_flat_run(px: np.ndarray) -> int:
    same = np.diff(px) == 0
    run = best = 0
    for x in same:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def build_panel(market: str, refresh: bool = False) -> Panel:
    spec = MARKETS[market]
    out = PANELS / market
    if refresh:
        for f in (out / "raw").glob("*.csv"):
            f.unlink()
    close, vol = _download(spec["candidates"], spec["start"], spec["end"], out / "raw")
    log: list[dict] = []
    close = close.loc[:spec["end"]]
    # Trading calendar = days on which at least half of the candidates traded.
    cal = close.index[close.notna().mean(axis=1) >= 0.5]
    close, vol = close.loc[cal], vol.reindex(cal)
    start = pd.Timestamp(spec["start"])
    cal_after = cal[cal >= start]
    deadline = cal_after[5]
    kept = []
    for tk in spec["candidates"]:
        if tk not in close.columns:
            log.append(dict(ticker=tk, action="DROPPED", detail="no data from vendor"))
            continue
        s = close[tk]
        first = s.first_valid_index()
        miss = float(s.loc[deadline:].isna().mean())
        if first is None or first > deadline:
            log.append(dict(ticker=tk, action="DROPPED", detail=f"late listing, first price {None if first is None else first.date()}"))
        elif miss >= 0.02:
            log.append(dict(ticker=tk, action="DROPPED", detail=f"{miss:.1%} missing days"))
        elif s.loc[deadline:].last_valid_index() < cal[-6]:
            log.append(dict(ticker=tk, action="DROPPED", detail="history ends early (delisted or renamed)"))
        elif _max_flat_run(s.loc[deadline:].ffill().to_numpy()) > 10:
            log.append(dict(ticker=tk, action="DROPPED",
                            detail=f"price unchanged for {_max_flat_run(s.loc[deadline:].ffill().to_numpy())} consecutive days (suspension)"))
        else:
            kept.append(tk)
    extra = kept[spec["n_assets"]:]
    kept = kept[:spec["n_assets"]]
    for tk in extra:
        log.append(dict(ticker=tk, action="NOT_USED", detail="beyond n_assets"))
    close = close[kept].loc[deadline:].ffill(limit=5)
    # First day on which every kept ticker has a price.
    first_all = close.dropna().index[0]
    close = close.loc[first_all:].ffill().bfill()
    vol = vol[kept].reindex(close.index).fillna(0.0)
    ret = np.log(close / close.shift(1)).iloc[1:]
    vol = vol.iloc[1:]
    R = ret.to_numpy(dtype=np.float64).copy()
    med = np.nanmedian(R, axis=1)
    D, N = R.shape
    for j in range(N):
        big = np.where(np.abs(R[:, j]) > 0.30)[0]
        for t in big:
            r = R[t, j]
            if abs(r) <= 0.30:      # already repaired as the partner of a reversal
                continue
            rev = None
            for h in (1, 2):                      # reversal on one of the next two days
                if t + h < D and np.sign(R[t + h, j]) == -np.sign(r) and abs(R[t + h, j]) >= 0.8 * abs(r):
                    rev = t + h
                    break
            if rev is not None:
                nxt = R[rev, j]
                R[t, j], R[rev, j] = med[t], med[rev]
                log.append(dict(ticker=kept[j], date=str(ret.index[t].date()), action="BAD_TICK",
                                detail=f"r={r:+.4f} reversed by {nxt:+.4f} on {ret.index[rev].date()}; both days set to cross-sectional median"))
            elif abs(r) > 0.50:
                R[t, j] = med[t]
                log.append(dict(ticker=kept[j], date=str(ret.index[t].date()), action="CORP_ACTION",
                                detail=f"r={r:+.4f} with no reversal; set to cross-sectional median"))
            else:
                log.append(dict(ticker=kept[j], date=str(ret.index[t].date()), action="WHITELISTED",
                                detail=f"r={r:+.4f} kept as a real move"))
    assert np.isfinite(R).all(), "non-finite returns after cleaning"
    assert (R.std(axis=0) > 0).all(), "zero-variance asset"
    panel = Panel(market, R, vol.to_numpy(dtype=np.float64), ret.index, kept)

    from scipy import stats
    zero_run = 0
    for j in range(N):
        z = (R[:, j] == 0).astype(int)
        run = best = 0
        for x in z:
            run = run + 1 if x else 0
            best = max(best, run)
        zero_run = max(zero_run, best)
    kurt = stats.kurtosis(R, axis=0, fisher=True)
    sha = hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()
    audit = dict(
        market=market, label=spec["label"], n_assets=N, n_days=D,
        first_date=str(ret.index[0].date()), last_date=str(ret.index[-1].date()),
        tickers=kept, nan_in_returns=int(np.isnan(R).sum()), max_zero_run=int(zero_run),
        n_abs_ret_gt_030=int((np.abs(R) > 0.30).sum()),
        kurtosis_median=float(np.median(kurt)), kurtosis_max=float(kurt.max()),
        sha256_returns=sha, log=log, source="Yahoo Finance via yfinance (auto-adjusted close)",
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "returns.npy", R)
    np.save(out / "volume.npy", panel.volume)
    pd.Series(ret.index.strftime("%Y-%m-%d"), name="date").to_csv(out / "dates.csv", index=False)
    pd.Series(kept, name="ticker").to_csv(out / "tickers.csv", index=False)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "sha256.txt").write_text(f"{sha}  returns.npy\n")
    lines = [f"{k}: {v}" for k, v in audit.items() if k not in ("log", "tickers")]
    lines += ["tickers: " + ", ".join(kept), "", "cleaning log:"]
    lines += [f"  {e.get('action'):12s} {e.get('ticker'):14s} {e.get('date', ''):10s} {e.get('detail', '')}" for e in log]
    (out / "audit.txt").write_text("\n".join(lines), encoding="utf-8")
    return panel


def load_panel(market: str) -> Panel:
    out = PANELS / market
    if not (out / "returns.npy").exists():
        return build_panel(market)
    R = np.load(out / "returns.npy")
    V = np.load(out / "volume.npy")
    dates = pd.DatetimeIndex(pd.read_csv(out / "dates.csv")["date"])
    tickers = pd.read_csv(out / "tickers.csv")["ticker"].tolist()
    return Panel(market, R, V, dates, tickers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=list(MARKETS))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    p = build_panel(a.market, refresh=a.refresh)
    print((PANELS / a.market / "audit.txt").read_text(encoding="utf-8"))
