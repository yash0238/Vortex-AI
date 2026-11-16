"""Controlled-ground-truth panels and contaminated copies of the NSE panel.

    python -m crisisbench.dgp --build       write every panel below
    python -m crisisbench.dgp --validate    results/01_dgp/metric_validation.csv + README.md

All panels are written in the on-disk format of data.load_panel (returns.npy,
volume.npy, dates.csv, tickers.csv, audit.json, sha256.txt) with the NSE panel's
dates and shape, so the unchanged runner scores generators on them:
    python -m crisisbench.bench --market sim_boost40 --gen iid_gaussian --fit full

1. Simulated panels with a KNOWN crisis correlation boost
   sim_boostXX (XX = 00, 20, 40, 80): two-state Markov regime-switching one-factor model
       r_it = c_s * (beta_i * k_s * F_t + sig_i * e_it),   s = state of day t,
     F and e_i independent Student-t(NU) shocks scaled to unit variance, so the population
     covariance in state s is c_s^2 (k_s^2 beta beta' + diag(sig^2)) exactly.
       normal (s=0): c = k = 1; loadings and volatilities are fixed heterogeneous grids scaled
                     so that the mean off-diagonal correlation is 0.25 and the mean daily
                     volatility is 1.3% (all correlations are positive: mean |rho| = mean rho).
       crisis (s=1): k (market-factor volatility multiplier) solves
                     mean rho_crisis = (1 + XX/100) * mean rho_normal, and every return is
                     also multiplied by c = CRISIS_SCALE, which raises volatility and
                     cross-sectional dispersion (what the labelling rule looks at) without
                     touching correlation.  sim_boost00 has k = 1: same correlation matrix in
                     both states, true boost exactly 0.
     Chain: P(normal -> crisis) = 1/170, P(crisis -> normal) = 1/30, i.e. stationary crisis
     probability 0.15 and expected crisis duration 30 days.  The four panels share the state
     path and the shocks (common random numbers): they differ only in k.
     The true state of every day is stored in state.npy.
   sim_ccc: NO regimes.  Constant-conditional-correlation GARCH(1,1):
       r_it = sqrt(h_it) z_it,  h_it = w_i + A r_{i,t-1}^2 + B h_{i,t-1}   (one recursion per asset),
     z_t has the normal-state correlation matrix above on every day.  Volatility clusters,
     the conditional correlation never changes; state.npy is all zeros and the true boost is 0.

2. Contaminated copies of the cleaned NSE panel (models of data that was NOT cleaned)
   nse_c_tick   one cell in 2015 REPLACED by a log return of -0.76
   nse_c_1pct   1% of all asset-days REPLACED by spikes of magnitude U(0.30, 0.60), random sign
   nse_c_split  one cell in 2016 REPLACED by -ln 2 (an unadjusted 2:1 split)
   Cells are drawn with a fixed seed and listed in audit.json ("injected").

Notes
  * "True boost" is a statement about POPULATION correlation matrices.  The benchmark's
    boost_pct averages |sample correlation| over 60-day windows, which has a positive floor
    even for independent data, so it is not expected to equal the population number; validate()
    measures the gap (see results/01_dgp/README.md).
  * The cells of the contaminated panels are replaced, not added to: the injected value is the
    recorded return itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import protocol as P
from .data import PANELS, ROOT, load_panel

SEED = 20261112
NU = 6.0                         # Student-t degrees of freedom of every shock
RHO_NORMAL = 0.25                # population mean off-diagonal correlation, normal state
VOL_NORMAL = 0.013               # mean daily volatility, normal state
CRISIS_SCALE = 2.0               # common return multiplier in the crisis state
P_NC, P_CN = 1.0 / 170.0, 1.0 / 30.0     # stationary crisis prob = P_NC / (P_NC + P_CN) = 0.15
CCC_A, CCC_B = 0.07, 0.91        # GARCH(1,1) of sim_ccc (finite 4th moment with t(6) shocks)
CCC_BURN = 1000
ORACLE_LOOKBACK, ORACLE_MIN = 20, 10     # window is crisis if >= 10 of its last 20 days are crisis

SIM_BOOSTS = {"sim_boost00": 0.0, "sim_boost20": 20.0, "sim_boost40": 40.0, "sim_boost80": 80.0}
SIM_PANELS = (*SIM_BOOSTS, "sim_ccc")
CONTAMINATED = ("nse_c_tick", "nse_c_1pct", "nse_c_split")
OUT = ROOT / "results" / "01_dgp"


# ------------------------------------------------------------------ the model
def mean_offdiag(C: np.ndarray, absolute: bool = True) -> float:
    off = ~np.eye(len(C), dtype=bool)
    return float((np.abs(C[off]) if absolute else C[off]).mean())


def _corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def base_params(n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    """(beta, sig) of the normal state: r_i = beta_i F + sig_i e_i with unit-variance F, e."""
    rng = np.random.default_rng(SEED)
    lam = rng.permutation(np.linspace(0.6, 1.4, n_assets))       # relative factor loadings (correlation scale)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    lam *= np.sqrt(RHO_NORMAL * n_assets * (n_assets - 1) / (s1 * s1 - s2))   # mean_{i != j} lam_i lam_j = RHO_NORMAL
    if lam.max() >= 1:
        raise ValueError("loading grid too wide for the requested correlation")
    vol = rng.permutation(np.linspace(0.8, 1.2, n_assets)) * VOL_NORMAL
    return lam * vol, vol * np.sqrt(1 - lam ** 2)


def state_cov(beta: np.ndarray, sig: np.ndarray, k: float = 1.0, c: float = 1.0) -> np.ndarray:
    return c * c * (k * k * np.outer(beta, beta) + np.diag(sig ** 2))


def model(name: str, n_assets: int | None = None) -> dict:
    """Parameters and population moments of a simulated panel."""
    if n_assets is None:
        n_assets = load_panel("nse").n_assets
    beta, sig = base_params(n_assets)
    cov_n = state_cov(beta, sig)
    if name == "sim_ccc":
        return dict(name=name, kind="ccc", beta=beta, sig=sig, corr=_corr(cov_n), vol=np.sqrt(np.diag(cov_n)),
                    a=CCC_A, b=CCC_B)
    target = (1 + SIM_BOOSTS[name] / 100.0) * mean_offdiag(_corr(cov_n))
    gap = lambda k: mean_offdiag(_corr(state_cov(beta, sig, k))) - target
    k = 1.0 if SIM_BOOSTS[name] == 0 else float(brentq(gap, 1.0, 50.0, xtol=1e-14, rtol=1e-14))
    cov_c = state_cov(beta, sig, k, CRISIS_SCALE)
    return dict(name=name, kind="boost", beta=beta, sig=sig, k=k, c=CRISIS_SCALE, cov_normal=cov_n, cov_crisis=cov_c,
                corr_normal=_corr(cov_n), corr_crisis=_corr(cov_c))


def true_boost(name: str) -> float:
    """Population boost in percent: 100 * (mean|rho_crisis| / mean|rho_normal| - 1)."""
    m = model(name)
    if m["kind"] == "ccc":
        return 0.0
    return 100.0 * (mean_offdiag(m["corr_crisis"]) / mean_offdiag(m["corr_normal"]) - 1.0)


# ----------------------------------------------------------------- simulation
def _std_t(rng: np.random.Generator, size) -> np.ndarray:
    return rng.standard_t(NU, size) * np.sqrt((NU - 2.0) / NU)


def simulate_chain(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Two-state Markov chain started from its stationary law; 1 = crisis."""
    u = rng.random(n_days)
    s = np.empty(n_days, dtype=np.int8)
    s[0] = u[0] < P_NC / (P_NC + P_CN)
    for t in range(1, n_days):
        s[t] = (u[t] >= P_CN) if s[t - 1] else (u[t] < P_NC)
    return s


def simulate_boost(m: dict, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Returns (len(state), N) of a sim_boostXX model given the state path."""
    crisis = np.asarray(state).astype(bool)
    F = _std_t(rng, len(crisis))
    E = _std_t(rng, (len(crisis), len(m["beta"])))
    k = np.where(crisis, m["k"], 1.0)
    c = np.where(crisis, m["c"], 1.0)
    return c[:, None] * ((k * F)[:, None] * m["beta"][None, :] + E * m["sig"][None, :])


def simulate_ccc(m: dict, n_days: int, rng: np.random.Generator) -> np.ndarray:
    lam = m["beta"] / m["vol"]                                   # corr(z_i, F): z has correlation m["corr"]
    n = n_days + CCC_BURN
    Z = _std_t(rng, n)[:, None] * lam[None, :] + _std_t(rng, (n, len(lam))) * np.sqrt(1 - lam ** 2)[None, :]
    w = m["vol"] ** 2 * (1 - m["a"] - m["b"])
    h = m["vol"] ** 2
    R = np.empty_like(Z)
    for t in range(n):
        R[t] = np.sqrt(h) * Z[t]
        h = w + m["a"] * R[t] ** 2 + m["b"] * h
    return R[CCC_BURN:]


def build_sim(name: str, n_days: int, n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    """(returns, true state).  Deterministic; the boost panels share chain and shocks."""
    m = model(name, n_assets)
    if m["kind"] == "ccc":
        return simulate_ccc(m, n_days, np.random.default_rng([SEED, 2])), np.zeros(n_days, dtype=np.int8)
    state = simulate_chain(n_days, np.random.default_rng([SEED, 0]))
    return simulate_boost(m, state, np.random.default_rng([SEED, 1])), state


def build_contaminated(name: str) -> tuple[np.ndarray, list[dict]]:
    """(returns, injected cells) of a contaminated copy of the cleaned NSE panel."""
    nse = load_panel("nse")
    R = nse.returns.copy()
    D, N = R.shape
    rng = np.random.default_rng([SEED, 10 + CONTAMINATED.index(name)])
    if name == "nse_c_1pct":
        flat = np.sort(rng.choice(D * N, size=int(round(0.01 * D * N)), replace=False))
        days, assets = np.divmod(flat, N)
        vals = rng.uniform(0.30, 0.60, flat.size) * rng.choice([-1.0, 1.0], flat.size)
    else:
        year = {"nse_c_tick": 2015, "nse_c_split": 2016}[name]
        days = np.array([rng.choice(np.where(nse.dates.year == year)[0])])
        assets = np.array([rng.integers(N)])
        vals = np.array([-0.76 if name == "nse_c_tick" else -np.log(2.0)])
    rec = [dict(day=int(d), date=str(nse.dates[d].date()), asset=int(j), ticker=nse.tickers[j],
                clean=float(R[d, j]), injected=float(v)) for d, j, v in zip(days, assets, vals)]
    R[days, assets] = vals
    return R, rec


# ------------------------------------------------------------------- writing
def write_panel(name: str, R: np.ndarray, volume: np.ndarray, dates: pd.DatetimeIndex, tickers: list[str],
                audit: dict, state: np.ndarray | None = None) -> None:
    if not np.isfinite(R).all():
        raise ValueError(f"{name}: non-finite returns")
    out = PANELS / name
    out.mkdir(parents=True, exist_ok=True)
    R = np.ascontiguousarray(R, dtype=np.float64)
    sha = hashlib.sha256(R.tobytes()).hexdigest()
    np.save(out / "returns.npy", R)
    np.save(out / "volume.npy", np.asarray(volume, dtype=np.float64))
    if state is not None:
        np.save(out / "state.npy", np.asarray(state, dtype=np.int8))
    pd.Series(dates.strftime("%Y-%m-%d"), name="date").to_csv(out / "dates.csv", index=False)
    pd.Series(tickers, name="ticker").to_csv(out / "tickers.csv", index=False)
    audit = dict(market=name, n_assets=int(R.shape[1]), n_days=int(R.shape[0]), first_date=str(dates[0].date()),
                 last_date=str(dates[-1].date()), sha256_returns=sha, **audit)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=float))
    (out / "sha256.txt").write_text(f"{sha}  returns.npy\n")


def build_all(verbose: bool = True) -> None:
    nse = load_panel("nse")
    D, N = nse.returns.shape
    for name in SIM_PANELS:
        R, state = build_sim(name, D, N)
        m = model(name, N)
        audit = dict(label=f"simulated ({name})", source="crisisbench.dgp", seed=SEED, nu=NU,
                     true_boost_pct=true_boost(name), beta=m["beta"].tolist(), sig=m["sig"].tolist(),
                     mean_daily_vol=float(R.std(axis=0).mean()), crisis_day_share=float(state.mean()))
        if m["kind"] == "boost":
            audit.update(factor_vol_multiplier=m["k"], crisis_scale=m["c"], p_normal_to_crisis=P_NC,
                         p_crisis_to_normal=P_CN, mac_population_normal=mean_offdiag(m["corr_normal"]),
                         mac_population_crisis=mean_offdiag(m["corr_crisis"]),
                         n_crisis_episodes=int((np.diff(np.r_[0, state]) == 1).sum()))
        else:
            audit.update(garch_alpha=CCC_A, garch_beta=CCC_B, mac_population=mean_offdiag(m["corr"]))
        write_panel(name, R, np.zeros_like(R), nse.dates, [f"SIM{j:02d}" for j in range(N)], audit, state)
        if verbose:
            print(f"{name:12s} true boost {audit['true_boost_pct']:6.2f}%  crisis days {state.mean():.3f}  "
                  f"mean vol {audit['mean_daily_vol']:.4f}")
    for name in CONTAMINATED:
        R, rec = build_contaminated(name)
        write_panel(name, R, nse.volume, nse.dates, nse.tickers,
                    dict(label=f"contaminated NSE ({name})", source="cleaned NSE panel + crisisbench.dgp injection",
                         seed=SEED, n_injected=len(rec), injected=rec))
        if verbose:
            print(f"{name:12s} {len(rec)} injected cell(s)" + (f": {rec[0]}" if len(rec) == 1 else ""))


# ---------------------------------------------------------------- validation
def load_state(name: str) -> np.ndarray:
    return np.load(PANELS / name / "state.npy")


def oracle_labels(state: np.ndarray, T: int = P.T) -> np.ndarray:
    """Window i covers days i .. i+T-1; crisis if >= ORACLE_MIN of its last ORACLE_LOOKBACK days are crisis."""
    c = np.concatenate([[0], np.cumsum(np.asarray(state, dtype=np.int64))])
    ends = np.arange(T, len(state) + 1)                          # exclusive end of each window
    return ((c[ends] - c[ends - ORACLE_LOOKBACK]) >= ORACLE_MIN).astype(np.int8)


def window_mac(W: np.ndarray) -> np.ndarray:
    """The benchmark's per-window statistic, taken from metrics.window_stats itself."""
    from . import metrics as M
    from .labels import fit_rule
    return M.window_stats(W, fit_rule(W)).mac


def _boost(mac: np.ndarray, lab: np.ndarray) -> float:
    cr, no = lab == 1, lab == 0
    if cr.sum() == 0 or no.sum() == 0:
        return float("nan")
    return float(100.0 * (mac[cr].mean() - mac[no].mean()) / mac[no].mean())


def oracle_window_boost(name: str) -> float:
    """boost_pct of the panel's 60-day windows with ORACLE labels (NaN for sim_ccc: no crisis state)."""
    W, _ = load_panel(name).windows(P.T)
    return _boost(window_mac(W), oracle_labels(load_state(name)))


def pure_state_window_mac(name: str, n_windows: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Mean window statistic of T-day windows that lie entirely in the normal / crisis state (Monte Carlo)."""
    m = model(name)
    rng = np.random.default_rng([SEED, 99, seed])
    out = []
    for s in (0, 1):
        R = simulate_boost(m, np.full(n_windows * P.T, s), rng).reshape(n_windows, P.T, -1)
        out.append(float(np.mean([mean_offdiag(np.corrcoef(w, rowvar=False)) for w in R])))
    return out[0], out[1]


def iid_floor(n_assets: int, n_windows: int = 2000, seed: int = 0) -> float:
    """Mean |sample correlation| in T-day windows of INDEPENDENT Gaussian assets (population value 0)."""
    rng = np.random.default_rng([SEED, 98, seed])
    return float(np.mean([mean_offdiag(np.corrcoef(rng.standard_normal((P.T, n_assets)), rowvar=False))
                          for _ in range(n_windows)]))


def validate(B: int = 200, write: bool = True) -> pd.DataFrame:
    """One row per simulated panel: what the benchmark's boost_pct measures against the truth."""
    from . import metrics as M
    from .bench import make_fit_data
    rows = []
    for name in SIM_PANELS:
        fd, W, _ = make_fit_data(name, "full")                   # exactly the runner's "full" fit
        ws = M.window_stats(W, fd.rule)
        ref = M.real_reference(ws, B=B)
        b = ref["metrics"]["boost_pct"]
        orc, rule = oracle_labels(load_state(name)), ws.labels
        tp = int(((rule == 1) & (orc == 1)).sum())
        n_rule = int(rule.sum())
        row = dict(panel=name, true_boost_pct=true_boost(name), pure_window_boost_pct=np.nan,
                   oracle_boost_pct=_boost(ws.mac, orc), rule_boost_pct=b["real"], rule_lo=b["lo"], rule_hi=b["hi"],
                   mac_normal_population=np.nan, mac_normal_window=ref["full"]["mac_normal"],
                   mac_crisis_window=ref["full"]["mac_crisis"], crisis_day_share=float(load_state(name).mean()),
                   oracle_crisis_windows=int(orc.sum()), rule_crisis_windows=n_rule,
                   label_precision=tp / n_rule if n_rule else np.nan,
                   label_recall=tp / int(orc.sum()) if orc.sum() else np.nan)
        m = model(name)
        if m["kind"] == "boost":
            mn, mc = pure_state_window_mac(name)
            row.update(pure_window_boost_pct=100 * (mc / mn - 1), mac_normal_population=mean_offdiag(m["corr_normal"]))
        else:
            row.update(mac_normal_population=mean_offdiag(m["corr"]))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    if write:
        floor = iid_floor(len(model("sim_ccc")["beta"]))
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "metric_validation.csv", index=False, float_format="%.4f")
        (OUT / "README.md").write_text(_readme(df, floor, B), encoding="utf-8")
    return df


def _readme(df: pd.DataFrame, floor: float, B: int) -> str:
    d = df.set_index("panel")
    tab = ["| panel | true boost % | pure-state window boost % | oracle-label window boost % | benchmark boost % "
           "[95% CI] | precision | recall |", "|---|---|---|---|---|---|---|"]
    f = lambda v, p=1: "n/a" if pd.isna(v) else f"{v:.{p}f}"
    for n, r in d.iterrows():
        tab.append(f"| {n} | {f(r.true_boost_pct)} | {f(r.pure_window_boost_pct)} | {f(r.oracle_boost_pct)} | "
                   f"{f(r.rule_boost_pct)} [{f(r.rule_lo)}, {f(r.rule_hi)}] | {f(r.label_precision, 2)} | {f(r.label_recall, 2)} |")
    b80, b00 = d.loc["sim_boost80"], d.loc["sim_boost00"]
    return "\n".join([
        "# Does boost_pct measure what it claims? (controlled ground truth)",
        "",
        "Produced by `python -m crisisbench.dgp --validate` from the simulated panels written by",
        "`python -m crisisbench.dgp --build`. Each panel has the NSE panel's dates and shape. The data come from a",
        "model whose crisis correlation boost is known (see the docstring of `crisisbench/dgp.py`).",
        "",
        *tab,
        "",
        "Columns",
        "- **true boost**: from the model's population correlation matrices, 100 * (mean|rho| crisis / mean|rho| normal - 1).",
        "- **pure-state window boost**: the benchmark's window statistic (mean |off-diagonal| of a 60-day sample",
        "  correlation matrix) averaged over 2000 simulated windows lying entirely in one state. No labelling error,",
        "  no mixed windows: only the effect of estimating correlations from 60 days.",
        "- **oracle-label window boost**: the same statistic on the panel's own overlapping windows, a window being",
        f"  crisis when at least {ORACLE_MIN} of its last {ORACLE_LOOKBACK} days are in the true crisis state.",
        "- **benchmark boost**: `boost_pct` exactly as the runner computes it for the `full` fit (frozen composite",
        f"  rule fitted on all windows), with the circular block bootstrap interval (B = {B}).",
        "- **precision / recall**: rule labels against oracle labels (CSV: label_precision, label_recall).",
        "",
        "CSV columns: true_boost_pct (population), pure_window_boost_pct, oracle_boost_pct, rule_boost_pct with",
        "rule_lo / rule_hi (the benchmark's boost_pct and its bootstrap interval), mac_* (mean |correlation|),",
        "crisis_day_share, oracle_crisis_windows, rule_crisis_windows, label_precision, label_recall.",
        "",
        "What the table shows",
        f"- Finite-sample floor. For independent Gaussian assets (population correlation 0) the mean |sample",
        f"  correlation| in a 60-day window of {len(model('sim_ccc')['beta'])} assets is {floor:.4f}, not 0. The window statistic of the normal",
        f"  state is {b80.mac_normal_window:.3f} on the panels against a population value of {b80.mac_normal_population:.3f}. Because this",
        "  upward bias is larger when the true correlation is lower, relative boosts are compressed: a true boost of",
        f"  {b80.true_boost_pct:.0f}% is {b80.pure_window_boost_pct:.1f}% even with perfect labels and pure windows.",
        f"- With oracle labels on the actual panel the {b80.true_boost_pct:.0f}% panel gives {b80.oracle_boost_pct:.1f}%, and the frozen rule gives",
        f"  {b80.rule_boost_pct:.1f}% (interval {b80.rule_lo:.1f} to {b80.rule_hi:.1f}).",
        f"- Null case. `sim_boost00` (volatility doubles, correlation unchanged) is scored {b00.rule_boost_pct:.1f}%",
        f"  (interval {b00.rule_lo:.1f} to {b00.rule_hi:.1f}); `sim_ccc` (volatility clustering, constant correlation, no regimes) is scored",
        f"  {d.loc['sim_ccc'].rule_boost_pct:.1f}% (interval {d.loc['sim_ccc'].rule_lo:.1f} to {d.loc['sim_ccc'].rule_hi:.1f}).",
        "- The benchmark value is therefore an ordinal, biased-towards-zero measure of the population boost: compare",
        "  it between a generator and real data under the same rule, do not read it as the population number.",
        "",
        "Limits: one realisation per panel (the bootstrap interval is the only uncertainty shown); a one-factor model",
        "with independent Student-t shocks; the crisis state doubles all volatilities, which makes labelling easier",
        "than on real data.",
        "",
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if not (a.build or a.validate):
        ap.error("nothing to do: pass --build and/or --validate")
    if a.build:
        build_all()
    if a.validate:
        print(validate().to_string(index=False))
