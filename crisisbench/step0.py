"""Step 0: does a cascade mechanism fitted on ordinary days predict record-breaking crashes?
Implements docs/step0_prereg.md exactly (frozen before the data was opened).

    python -m crisisbench.step0            -> results/step0/{blocks_*.csv, summary.json, fingerprints.json}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step0"
BLOCK = 20
NSIM = 20_000
FLOOR = 1e-6
EDGES49 = [0, 1, 3, 6, 11, 21, 41, 81, 161, 321]          # bin k = [EDGES[k], EDGES[k+1]-1]; last bin open


# ------------------------------------------------------------------ data
def load_french() -> pd.DataFrame:
    f = next((ROOT / "data" / "external" / "french").glob("49_Industry_Portfolios_Daily.csv"))
    lines = f.read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if "Average Value Weighted Returns" in l) + 1
    header = lines[start].split(",")
    rows = []
    for l in lines[start + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        rows.append(l.split(","))
    df = pd.DataFrame(rows, columns=["date"] + [h.strip() for h in header[1:]])
    df.index = pd.to_datetime(df.pop("date").str.strip(), format="%Y%m%d")
    df = df.apply(lambda c: pd.to_numeric(c.str.strip(), errors="coerce"))
    return df.mask(df <= -99.0) / 100.0


def load_nse() -> pd.DataFrame:
    p = ROOT / "data" / "panels" / "nse_ext"
    r = np.load(p / "returns.npy")
    d = pd.read_csv(p / "dates.csv").iloc[:, 0]
    return pd.DataFrame(r, index=pd.to_datetime(d))


def counts(R: pd.DataFrame, fit_mask: np.ndarray) -> np.ndarray:
    """Daily event counts with 1% thresholds from the fit sample (industries with >= 250 valid fit days)."""
    F = R[fit_mask]
    ok = F.notna().sum() >= 250
    q = F.loc[:, ok].quantile(0.01)
    E = R.loc[:, ok].lt(q, axis=1) & R.loc[:, ok].notna()
    N = E.sum(axis=1).to_numpy(dtype=np.int64)
    counts.K = int(ok.sum())             # number of series that can have an event (daily cap)
    return N


def block_sums(N: np.ndarray) -> np.ndarray:
    nb = len(N) // BLOCK
    return N[:nb * BLOCK].reshape(nb, BLOCK).sum(1)


def edges_for(n_series: int) -> np.ndarray:
    return np.array([0] + [max(1, int(round(e * n_series / 49))) for e in EDGES49[1:]])


def bin_of(s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, s, side="right") - 1


# ------------------------------------------------------------------ M1 / M0b
def nb_logpmf(k, mean, r):
    p = r / (r + mean)
    return gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log1p(-p)


def excitation(N: np.ndarray, beta: float) -> np.ndarray:
    """x_t = beta x_{t-1} + (1-beta) N_{t-1}, x_0 = mean."""
    prev = np.concatenate([[N.mean()], N[:-1]]).astype(float)
    x = lfilter([1 - beta], [1, -beta], prev, zi=[beta * N.mean()])[0]
    return x


def fit_m1(N: np.ndarray, alpha_zero: bool = False) -> dict:
    sig = lambda z: 1 / (1 + np.exp(-z))

    def unpack(th):
        mu = math.exp(th[0]); r = math.exp(th[1])
        if alpha_zero:
            return mu, 0.0, 0.5, r
        return mu, 0.999 * sig(th[2]), 0.999 * sig(th[3]), r

    def nll(th):
        mu, a, b, r = unpack(th)
        lam = mu + a * excitation(N, b) if a > 0 else np.full(len(N), mu)
        return -nb_logpmf(N, lam, r).sum()

    m = max(N.mean(), 1e-3)
    starts = [[math.log(m), 0.0]] if alpha_zero else \
        [[math.log(m * (1 - a)), 0.0, math.log(a / (1 - a)), math.log(b / (1 - b))] for a in (0.3, 0.6, 0.85)
         for b in (0.5, 0.9)]
    best = None
    for s0 in starts:
        res = optimize.minimize(nll, s0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    mu, a, b, r = unpack(best.x)
    return dict(mu=mu, alpha=a, beta=b, r=r, nll=float(best.fun), n=a)


def sim_blocks(par: dict, x0: float, rng: np.random.Generator, K: int) -> np.ndarray:
    """Simulated 20-day severities; a daily count cannot exceed K, the number of series (change 1 in prereg)."""
    x = np.full(NSIM, x0, dtype=float)
    tot = np.zeros(NSIM, dtype=np.int64)
    for _ in range(BLOCK):
        lam = par["mu"] + par["alpha"] * x
        k = np.minimum(rng.negative_binomial(par["r"], par["r"] / (par["r"] + lam)), K)
        tot += k
        x = par["beta"] * x + (1 - par["beta"]) * k
    return tot


def probs_from_sims(sims: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.bincount(bin_of(sims, edges), minlength=len(edges)) / len(sims)


# ------------------------------------------------------------------ M0a
def evt_probs(S_fit: np.ndarray, edges: np.ndarray) -> np.ndarray:
    u = np.quantile(S_fit, 0.90)
    exc = S_fit[S_fit > u] - u
    c, _, scale = stats.genpareto.fit(exc, floc=0) if len(exc) >= 10 else (0.0, 0, max(exc.mean(), 1.0))
    pu = (S_fit <= u).mean()

    def F(s):                       # P(S <= s), s integer (may be -1)
        if s < 0:
            return 0.0
        if s <= u:
            return float((S_fit <= s).mean())
        return pu + (1 - pu) * float(stats.genpareto.cdf(s - u, c, scale=scale))

    out = []
    for k in range(len(edges)):
        lo = edges[k]
        hi = edges[k + 1] - 1 if k + 1 < len(edges) else None
        out.append((F(hi) if hi is not None else 1.0) - F(lo - 1))
    return np.array(out)


# ------------------------------------------------------------------ evaluation
def evaluate(R: pd.DataFrame, fit_dates: list, horizon_end, tag: str, rng) -> pd.DataFrame:
    dates = R.index
    edges = edges_for(R.shape[1])
    rows = []
    for i, fd in enumerate(fit_dates):
        fit_mask = np.asarray(dates < fd)
        nxt = fit_dates[i + 1] if i + 1 < len(fit_dates) else horizon_end
        N = counts(R, fit_mask)
        K = counts.K
        nfit = int(fit_mask.sum())
        m1, m0b = fit_m1(N[:nfit]), fit_m1(N[:nfit], alpha_zero=True)
        nb_fit = nfit // BLOCK
        S_all = block_sums(N)
        S_fit = S_all[:nb_fit]
        p0a = np.maximum(evt_probs(S_fit, edges), FLOOR)
        p0b = np.maximum(probs_from_sims(sim_blocks(m0b, 0.0, rng, K), edges), FLOOR)
        x_all = excitation(N, m1["beta"])
        for b in range(nb_fit, len(S_all)):
            d0 = dates[b * BLOCK]
            if d0 < fd:
                continue
            if d0 >= nxt:
                break
            p1 = np.maximum(probs_from_sims(sim_blocks(m1, x_all[b * BLOCK], rng, K), edges), FLOOR)
            k = int(bin_of(np.array([S_all[b]]), edges)[0])
            rows.append(dict(data=tag, fit_date=str(fd.date()), block_start=str(d0.date()), severity=int(S_all[b]),
                             bin=k, record=bool(S_all[b] > S_all[:b].max()),
                             ls_m1=math.log(p1[k]), ls_m0a=math.log(p0a[k]), ls_m0b=math.log(p0b[k]),
                             n_hat=m1["n"], beta=m1["beta"], state=float(x_all[b * BLOCK])))
        print(f"{tag} fit {fd.date()}  n={m1['n']:.3f} beta={m1['beta']:.3f} r={m1['r']:.2f}  blocks={len(rows)}",
              flush=True)
    return pd.DataFrame(rows)


def rolling_n(R: pd.DataFrame, years: int = 5) -> pd.Series:
    out = {}
    for y in range(R.index[0].year + years, R.index[-1].year + 2):
        end = pd.Timestamp(f"{y}-01-01")
        win = (R.index >= end - pd.DateOffset(years=years)) & (R.index < end)
        if win.sum() < 250 * years * 0.8:
            continue
        N = counts(R[win], np.ones(win.sum(), bool))
        out[end] = fit_m1(N)["n"]
    return pd.Series(out)


def decay_exponent(N: np.ndarray, start: int, horizon: int = 120) -> tuple[float, float]:
    """Poisson regression log E[N_{start+tau}] = c - p log(tau), tau = 1..horizon. Returns (p, se)."""
    y = N[start + 1:start + 1 + horizon].astype(float)
    if len(y) < horizon:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(horizon), -np.log(np.arange(1, horizon + 1))])
    beta = np.array([math.log(max(y.mean(), 1e-3)), 0.0])
    for _ in range(50):
        mu = np.exp(X @ beta)
        H = X.T @ (X * mu[:, None])
        g = X.T @ (y - mu)
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(X.T @ (X * np.exp(X @ beta)[:, None]))
    disp = max(1.0, float(((y - np.exp(X @ beta)) ** 2 / np.exp(X @ beta)).sum() / (horizon - 2)))
    return float(beta[1]), float(math.sqrt(cov[1, 1] * disp))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118)
    fr = load_french()
    fits = [pd.Timestamp(f"{y}-01-01") for y in range(1946, fr.index[-1].year + 1)]
    b1 = evaluate(fr, fits, fr.index[-1] + pd.Timedelta(days=1), "french49", rng)
    b1.to_csv(OUT / "blocks_french49.csv", index=False)
    nse = load_nse()
    b2 = evaluate(nse, [pd.Timestamp("2020-01-01")], nse.index[-1] + pd.Timedelta(days=1), "nse47", rng)
    b2.to_csv(OUT / "blocks_nse47.csv", index=False)

    rec1, rec2 = b1[b1.record], b2[b2.record]
    rec = pd.concat([rec1, rec2[rec2.block_start.str.startswith("2020")]])
    summary = dict(
        french_blocks=len(b1), french_mean_ls=dict(m1=b1.ls_m1.mean(), m0a=b1.ls_m0a.mean(), m0b=b1.ls_m0b.mean()),
        nse_blocks=len(b2), nse_mean_ls=dict(m1=b2.ls_m1.mean(), m0a=b2.ls_m0a.mean(), m0b=b2.ls_m0b.mean()),
        records=rec[["data", "block_start", "severity", "ls_m1", "ls_m0a", "ls_m0b", "n_hat"]].to_dict("records"),
        P1=bool(b1.ls_m1.mean() > b1.ls_m0a.mean()),
        P2_wins=int((rec.ls_m1 > rec.ls_m0a).sum()), P2_total=int(len(rec)),
    )
    summary["P2"] = bool(summary["P2_wins"] * 2 > summary["P2_total"])
    summary["PASS"] = summary["P1"] and summary["P2"]

    # fingerprints (D1)
    rn = rolling_n(fr)
    f1 = []
    for _, r in rec1.iterrows():
        prior = rn[rn.index <= pd.Timestamp(r.block_start)]
        if len(prior):
            f1.append(dict(block_start=r.block_start, n_5y=float(prior.iloc[-1]), above_median=bool(prior.iloc[-1] > rn.median())))
    F1 = bool(f1) and sum(x["above_median"] for x in f1) * 2 > len(f1)
    Nfull = counts(fr, np.ones(len(fr), bool))
    S = block_sums(Nfull)
    top = np.argsort(S)[::-1]
    chosen = []
    for b in top:                                  # 10 largest, at least 120 days apart
        if all(abs(b - c) * BLOCK > 120 for c in chosen):
            chosen.append(b)
        if len(chosen) == 10:
            break
    f2 = []
    for b in chosen:
        d0 = fr.index[b * BLOCK]
        prior = rn[rn.index <= d0]
        p, se = decay_exponent(Nfull, b * BLOCK + BLOCK - 1)
        f2.append(dict(block_start=str(d0.date()), severity=int(S[b]), p=p, se=se,
                       precursor=bool(len(prior) and prior.iloc[-1] > rn.median())))
    g = pd.DataFrame(f2).dropna()
    A, B = g[g.precursor], g[~g.precursor]
    F2 = None
    if len(A) >= 2 and len(B) >= 2:
        seA, seB = A.p.std(ddof=1) / math.sqrt(len(A)), B.p.std(ddof=1) / math.sqrt(len(B))
        F2 = bool(abs(A.p.mean() - B.p.mean()) > max(seA, seB))
        f2_stats = dict(p_precursor=A.p.mean(), se_precursor=seA, p_none=B.p.mean(), se_none=seB)
    else:
        f2_stats = dict(note="fewer than 2 events in a group; F2 not testable")
    fp = dict(F1=F1, F1_detail=f1, F2=F2, F2_detail=f2, F2_stats=f2_stats,
              rolling_n=dict(median=float(rn.median()), series={str(k.date()): float(v) for k, v in rn.items()}))
    summary["fingerprints"] = dict(F1=F1, F2=F2)
    summary["DECISION"] = "build" if summary["PASS"] and (F1 or bool(F2)) else "drop or reformulate"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (OUT / "fingerprints.json").write_text(json.dumps(fp, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, default=float))
    print("records:")
    for r in summary["records"]:
        print(f"  {r['data']} {r['block_start']} sev={r['severity']}  M1 {r['ls_m1']:.2f}  EVT {r['ls_m0a']:.2f}  "
              f"noCascade {r['ls_m0b']:.2f}  n={r['n_hat']:.2f}")


if __name__ == "__main__":
    main()
