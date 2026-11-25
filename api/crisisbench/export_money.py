"""Scenario engine for the consumer app (app/money.html): one compact JSON, everything else runs in the browser.

    python -m crisisbench.export_money [--gen cgnsde2] [--n 4000] [--n-crash 1500]

Contents of app/public/money/engine.json
  assets       ticker, display name, sector for the 47 NSE stocks
  gen_month    unconditional CG-NSDE windows cut into 3 monthly log returns: (n*3, 47), int16 x 1e4, base64
  crash_week   crisis-regime CG-NSDE windows (the generator's crisis prior; the window starts a stress
               episode) as 12 weekly log returns: (n_crash, 12, 47), int16 x 1e4, base64
  corr_calm / corr_crash   47x47 correlation of daily returns inside generated windows that the frozen rule
               labels calm / crisis (the 'diversification that disappears' numbers)
  real_month   real monthly log returns 2010-11..2026-10, (months, 47), int16 x 1e4, base64, with month labels
  real_ew_daily  real equal-weight daily log returns + ISO dates (to estimate a mutual fund's market beta)
  real_week_max  for the tip check: per stock, all rolling 1/4/13/52-week returns (quantiles only)
Generated scenarios are synthetic and releasable; the real series are derived from vendor data and are for
local, non-commercial use only (stated in the app).
"""
from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import pandas as pd

from .bench import ARTIFACTS, make_fit_data
from .data import ROOT, load_panel
from .generators import get
from .labels import label_windows

OUT = ROOT / "app" / "public" / "money"

NAMES = {
    "RELIANCE": ("Reliance Industries", "Energy"), "TCS": ("TCS", "IT"), "HDFCBANK": ("HDFC Bank", "Banks"),
    "INFY": ("Infosys", "IT"), "HINDUNILVR": ("Hindustan Unilever", "FMCG"), "ICICIBANK": ("ICICI Bank", "Banks"),
    "KOTAKBANK": ("Kotak Mahindra Bank", "Banks"), "BHARTIARTL": ("Bharti Airtel", "Telecom"), "ITC": ("ITC", "FMCG"),
    "AXISBANK": ("Axis Bank", "Banks"), "SBIN": ("State Bank of India", "Banks"), "LT": ("Larsen & Toubro", "Infra"),
    "BAJFINANCE": ("Bajaj Finance", "Finance"), "HCLTECH": ("HCL Tech", "IT"), "ASIANPAINT": ("Asian Paints", "Consumer"),
    "MARUTI": ("Maruti Suzuki", "Auto"), "SUNPHARMA": ("Sun Pharma", "Pharma"), "TITAN": ("Titan", "Consumer"),
    "ULTRACEMCO": ("UltraTech Cement", "Materials"), "NESTLEIND": ("Nestle India", "FMCG"), "WIPRO": ("Wipro", "IT"),
    "POWERGRID": ("Power Grid", "Utilities"), "NTPC": ("NTPC", "Utilities"), "M&M": ("Mahindra & Mahindra", "Auto"),
    "TECHM": ("Tech Mahindra", "IT"), "TMPV": ("Tata Motors", "Auto"), "TATASTEEL": ("Tata Steel", "Metals"),
    "JSWSTEEL": ("JSW Steel", "Metals"), "BAJAJ-AUTO": ("Bajaj Auto", "Auto"), "CIPLA": ("Cipla", "Pharma"),
    "DRREDDY": ("Dr Reddy's", "Pharma"), "DIVISLAB": ("Divi's Labs", "Pharma"), "HEROMOTOCO": ("Hero MotoCorp", "Auto"),
    "ONGC": ("ONGC", "Energy"), "COALINDIA": ("Coal India", "Energy"), "BPCL": ("BPCL", "Energy"),
    "GRASIM": ("Grasim", "Materials"), "ADANIPORTS": ("Adani Ports", "Infra"), "EICHERMOT": ("Eicher Motors", "Auto"),
    "APOLLOHOSP": ("Apollo Hospitals", "Healthcare"), "HINDALCO": ("Hindalco", "Metals"),
    "TATACONSUM": ("Tata Consumer", "FMCG"), "BRITANNIA": ("Britannia", "FMCG"), "SHREECEM": ("Shree Cement", "Materials"),
    "UPL": ("UPL", "Materials"), "BAJAJFINSV": ("Bajaj Finserv", "Finance"), "INDUSINDBK": ("IndusInd Bank", "Banks"),
}


def b64_int16(a: np.ndarray) -> str:
    q = np.clip(np.round(np.asarray(a) * 1e4), -32000, 32000).astype("<i2")
    return base64.b64encode(q.tobytes()).decode()


def corr_inside(W: np.ndarray) -> np.ndarray:
    Z = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
    C = np.einsum("wti,wtj->ij", Z, Z) / (W.shape[0] * W.shape[1])
    return np.round(C, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="cgnsde2")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-crash", type=int, default=1200)
    ap.add_argument("--pool", type=int, default=24000, help="generated windows searched for stress scenarios")
    ap.add_argument("--growth", type=float, default=0.105, help="assumed long-run yearly growth (median) of every stock")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    klass = type(get(a.gen))
    g = klass.load(ARTIFACTS / "models" / "nse" / a.gen / "full" / "seed0.pt")
    rng = np.random.default_rng(2026)
    Wn = g.sample(a.n, rng)                                  # (n, 60, 47) daily log returns
    # Drift: the fitted model reproduces 2010-24 average returns of today's surviving large caps (~20%+ a year),
    # far too rosy for planning. Every stock's own mean is removed and one stated long-run growth is added, so
    # no stock is assumed to beat another and the plan does not bank on a repeat of a survivor bull market.
    mu_i = Wn.mean((0, 1), keepdims=True)
    drift = np.log1p(a.growth) / 252.0
    Wn = Wn - mu_i + drift
    # Stress scenarios: the worst 5% of generated 3-month windows by equal-weight drawdown (a stress test asks
    # "how bad can a plausible bad quarter get", not "what does an average day in a crisis regime look like").
    pool = np.concatenate([g.sample(a.pool // 2, rng), g.sample_regime(a.pool // 2, 1, rng)]) - mu_i + drift
    cum = np.cumsum(pool.mean(2), 1)
    dd = (cum - np.maximum.accumulate(np.concatenate([np.zeros((len(cum), 1)), cum], 1)[:, 1:])).min(1)
    worst = np.argsort(dd)[: a.n_crash]
    Wc = pool[worst]
    print(f"stress pool {len(pool)}: worst-{a.n_crash} equal-weight drawdowns median {np.expm1(np.median(dd[worst])):.3f}, "
          f"range [{np.expm1(dd[worst].min()):.3f}, {np.expm1(dd[worst].max()):.3f}]")
    fd, Wr, _ = make_fit_data("nse", "full")
    gen_month = Wn.reshape(a.n, 3, 20, -1).sum(2).reshape(a.n * 3, -1)
    crash_week = Wc.reshape(a.n_crash, 12, 5, -1).sum(2)
    lab = label_windows(Wn, fd.rule)[0].astype(bool)
    labc = label_windows(Wc, fd.rule)[0].astype(bool)
    lab_pool = label_windows(pool, fd.rule)[0].astype(bool)
    calm = Wn[~lab]
    crash = np.concatenate([Wn[lab], pool[lab_pool]])

    p = load_panel("nse_ext")
    df = pd.DataFrame(p.returns, index=p.dates, columns=p.tickers)
    mon = df.groupby(df.index.to_period("M")).sum()
    mon = mon.iloc[1:-1] if len(mon) > 2 else mon            # drop partial first / last month
    ew = df.mean(1)
    weekly = df.groupby(np.arange(len(df)) // 5).sum().values
    tip = {}
    for h in (1, 4, 13, 52):
        c = np.cumsum(np.vstack([np.zeros((1, weekly.shape[1])), weekly]), 0)
        roll = (c[h:] - c[:-h]).ravel()
        tip[str(h)] = [round(float(np.expm1(np.quantile(roll, q))), 4) for q in (0.5, 0.9, 0.99, 0.999, 1.0)]
    assets = []
    for t in p.tickers:
        k = t.replace(".NS", "")
        nm, sec = NAMES.get(k, (k, "Other"))
        assets.append(dict(ticker=k, name=nm, sector=sec))
    out = dict(
        version=2, model=a.gen, growth=a.growth, stress_share=round(a.n_crash / len(pool), 4), n_month=int(gen_month.shape[0]), n_crash=int(a.n_crash), n_assets=len(assets),
        assets=assets, crash_label_share=float(labc.mean()),
        gen_month=b64_int16(gen_month), crash_week=b64_int16(crash_week),
        corr_calm=corr_inside(calm).tolist(), corr_crash=corr_inside(crash).tolist(),
        real_corr_calm=corr_inside(Wr[~label_windows(Wr, fd.rule)[0].astype(bool)]).tolist(),
        real_corr_crash=corr_inside(Wr[label_windows(Wr, fd.rule)[0].astype(bool)]).tolist(),
        real_months=[str(m) for m in mon.index], real_month=b64_int16(mon.values),
        real_ew_dates=[d.strftime("%Y-%m-%d") for d in ew.index], real_ew=b64_int16(ew.values),
        tip_quantiles=tip, tip_q=[0.5, 0.9, 0.99, 0.999, 1.0],
        note="Scenarios: synthetic output of CG-NSDE. Real series: derived from Yahoo Finance prices, "
             "local educational use only.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("engine.json", round((OUT / "engine.json").stat().st_size / 1e6, 2), "MB;",
          f"calm windows {len(calm)}, crash windows {len(crash)}, crash label share of crisis prior {labc.mean():.2f}")
    mc = np.mean(np.abs(np.array(out["corr_calm"]))[~np.eye(47, dtype=bool)])
    mx = np.mean(np.abs(np.array(out["corr_crash"]))[~np.eye(47, dtype=bool)])
    print(f"mean |corr| calm {mc:.3f}  crash {mx:.3f}")


if __name__ == "__main__":
    main()
