"""The single metric module. Every number in every table comes through here.

Design: a set of (T, N) windows is reduced to per-window sufficient statistics
(`WindowStats`).  Any metric on any subset or bootstrap resample of windows is
an aggregation over window indices, so real data, bootstrap replicates of real
data and generated data all go through exactly the same code path.

Metric kinds
  stat : a statistic with a real-data value and a bootstrap CI
         (normalised error = |gen - real| / CI half-width)
  dist : a distance between the generated and the real sample
         (normalised error = gen distance / 95th percentile of the distance
          between a bootstrap resample of real data and the real sample)

Definitions (W has shape (n, T, N); "pooled" = over all windows and days):
  A  std_ratio     mean over assets of pooled std(gen) / pooled std(real)
     kurt_med      median over assets of pooled excess kurtosis
     acf1_ret      mean over assets of the lag-1 autocorrelation of returns
     acf1_sq       mean over assets of the lag-1 autocorrelation of squared returns
     acf5_sq_mean  mean over assets and lags 1..5 of the autocorrelation of squared returns
     skew_ew       skewness of pooled equal-weight portfolio daily returns
  B  corr_fro      Frobenius norm of (pooled corr gen - pooled corr real), full matrix
     corr_mae      mean absolute off-diagonal difference of the same matrices
     eig_dist      L2 distance between sorted eigenvalue spectra of the two matrices / sqrt(N)
     pc1_share     largest eigenvalue of the pooled correlation matrix / N
  C  crisis_freq   share of windows labelled crisis by the frozen rule
     boost_pct     100 * (mean_crisis(m) - mean_normal(m)) / mean_normal(m),
                   m = mean |off-diagonal| of the window's own 60-day correlation matrix
     tail_coexc    average over ordered asset pairs of P(r_j < q_j | r_i < q_i), q = pooled 5% quantile
     cvar_ratio    ES_5%(crisis) / ES_5%(normal) of equal-weight daily returns taken from the
                   last 20 days of each window, pooled by window label
     pc1_share_gap mean_crisis - mean_normal of the window's top-eigenvalue share
     connect_gap   mean_crisis - mean_normal of the share of pairs with |rho| > 0.5 in the window
  D  disc_feat     blocked-CV accuracy of a gradient-boosted classifier, real vs generated,
                   on 20 order-free summary features of each window
     disc_raw      same classifier on the first 64 principal components of the flattened window
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import protocol as P
from .labels import label_windows

KIND = {
    "std_ratio": "stat", "kurt_med": "stat", "acf1_ret": "stat", "acf1_sq": "stat", "acf5_sq_mean": "stat",
    "skew_ew": "stat", "corr_fro": "dist", "corr_mae": "dist", "eig_dist": "dist", "pc1_share": "stat",
    "crisis_freq": "stat", "boost_pct": "stat", "tail_coexc": "stat", "cvar_ratio": "stat",
    "pc1_share_gap": "stat", "connect_gap": "stat", "disc_feat": "disc", "disc_raw": "disc",
}
CRISIS_METRICS = ("boost_pct", "cvar_ratio", "pc1_share_gap", "connect_gap")
NLAG = 5


@dataclass
class WindowStats:
    n: int
    T: int
    N: int
    s1: np.ndarray      # (n, N) sum x
    s2: np.ndarray
    s3: np.ndarray
    s4: np.ndarray
    sxx: np.ndarray     # (n, N, N) sum x x'
    rP: np.ndarray      # (n, N)  sum x_t x_{t+1}
    rA: np.ndarray      # (n, N)  sum x_t      (t < T-1)
    rB: np.ndarray      # (n, N)  sum x_{t+1}
    qP: np.ndarray      # (n, NLAG, N) same for y = x^2 at lags 1..NLAG
    qA: np.ndarray
    qB: np.ndarray
    q1: np.ndarray      # (n, N) sum y
    q2: np.ndarray      # (n, N) sum y^2
    e1: np.ndarray      # (n,) sums of EW portfolio returns
    e2: np.ndarray
    e3: np.ndarray
    ew_tail: np.ndarray  # (n, 20) EW returns, last 20 days
    mac: np.ndarray     # (n,) mean |offdiag corr| in the window
    pc1: np.ndarray     # (n,) top eigenvalue share in the window
    dens: np.ndarray    # (n,) share of pairs with |rho| > CONNECT_RHO
    tj: np.ndarray      # (n,) joint lower-tail exceedance pair count
    tm: np.ndarray      # (n,) marginal lower-tail exceedance count
    labels: np.ndarray  # (n,) crisis labels by the frozen rule
    labels_alt: np.ndarray
    tail_q: np.ndarray  # (N,) thresholds used for tj/tm


def window_stats(W: np.ndarray, rule: dict, tail_q: np.ndarray | None = None) -> WindowStats:
    W = np.asarray(W, dtype=np.float64)
    n, T, N = W.shape
    Y = W * W
    s1, s2 = W.sum(1), Y.sum(1)
    s3, s4 = (Y * W).sum(1), (Y * Y).sum(1)
    sxx = np.einsum("ntj,ntk->njk", W, W, optimize=True)
    rP = (W[:, :-1] * W[:, 1:]).sum(1)
    rA, rB = W[:, :-1].sum(1), W[:, 1:].sum(1)
    qP = np.stack([(Y[:, :-k] * Y[:, k:]).sum(1) for k in range(1, NLAG + 1)], 1)
    qA = np.stack([Y[:, :-k].sum(1) for k in range(1, NLAG + 1)], 1)
    qB = np.stack([Y[:, k:].sum(1) for k in range(1, NLAG + 1)], 1)
    ew = W.mean(2)
    # per-window correlation structure
    mu = s1 / T
    cov = sxx / T - mu[:, :, None] * mu[:, None, :]
    sd = np.sqrt(np.clip(np.einsum("njj->nj", cov), 1e-20, None))
    C = cov / sd[:, :, None] / sd[:, None, :]
    C = np.nan_to_num(C, nan=0.0)
    off = ~np.eye(N, dtype=bool)
    absC = np.abs(C[:, off])
    mac = absC.mean(1)
    dens = (absC > P.CONNECT_RHO).mean(1)
    pc1 = np.linalg.eigvalsh(C)[:, -1] / N
    if tail_q is None:
        tail_q = np.quantile(W.reshape(-1, N), P.TAIL_Q, axis=0)
    c = (W < tail_q[None, None, :]).sum(2).astype(np.float64)     # (n, T) assets in their tail
    tj, tm = (c * (c - 1)).sum(1), c.sum(1)
    lab, _ = label_windows(W, rule)
    lab_alt, _ = label_windows(W, rule, alt=True)
    return WindowStats(n, T, N, s1, s2, s3, s4, sxx, rP, rA, rB, qP, qA, qB, s2.copy(), s4.copy(),
                       ew.sum(1), (ew ** 2).sum(1), (ew ** 3).sum(1), ew[:, -20:].copy(),
                       mac, pc1, dens, tj, tm, lab, lab_alt, tail_q)


def _acf(Psum, Asum, Bsum, tot1, tot2, cnt_pairs, cnt_all):
    """Pooled autocorrelation from sums. All args are (N,) arrays except counts."""
    m = tot1 / cnt_all
    var = tot2 / cnt_all - m * m
    num = (Psum - m * (Asum + Bsum)) / cnt_pairs + m * m
    return num / np.clip(var, 1e-30, None)


def _es(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return np.nan
    k = max(1, int(np.ceil(q * x.size)))
    return float(-np.sort(x)[:k].mean())


def aggregate(ws: WindowStats, idx: np.ndarray | None = None, alt_labels: bool = False) -> dict:
    """Sample-level statistics for the windows in idx (repeats allowed)."""
    if idx is None:
        idx = np.arange(ws.n)
    n, T, N = len(idx), ws.T, ws.N
    cnt = n * T
    S1, S2, S3, S4 = (a[idx].sum(0) for a in (ws.s1, ws.s2, ws.s3, ws.s4))
    m = S1 / cnt
    var = S2 / cnt - m * m
    std = np.sqrt(var)
    m4 = S4 / cnt - 4 * m * S3 / cnt + 6 * m * m * S2 / cnt - 3 * m ** 4
    kurt = m4 / var ** 2 - 3.0
    cov = ws.sxx[idx].sum(0) / cnt - np.outer(m, m)
    corr = cov / np.outer(std, std)
    eig = np.linalg.eigvalsh(corr)[::-1]
    acf1 = _acf(ws.rP[idx].sum(0), ws.rA[idx].sum(0), ws.rB[idx].sum(0), S1, S2, n * (T - 1), cnt)
    Q1, Q2 = ws.q1[idx].sum(0), ws.q2[idx].sum(0)
    acq = np.stack([_acf(ws.qP[idx, k].sum(0), ws.qA[idx, k].sum(0), ws.qB[idx, k].sum(0), Q1, Q2,
                         n * (T - k - 1), cnt) for k in range(NLAG)])          # (NLAG, N)
    e1, e2, e3 = ws.e1[idx].sum() / cnt, ws.e2[idx].sum() / cnt, ws.e3[idx].sum() / cnt
    ev = e2 - e1 * e1
    skew = (e3 - 3 * e1 * e2 + 2 * e1 ** 3) / ev ** 1.5
    lab = (ws.labels_alt if alt_labels else ws.labels)[idx]
    cr, no = lab == 1, lab == 0
    n_cr = int(cr.sum())
    out = dict(
        std=std, corr=corr, eig=eig,
        kurt_med=float(np.median(kurt)), acf1_ret=float(acf1.mean()), acf1_sq=float(acq[0].mean()),
        acf5_sq_mean=float(acq.mean()), skew_ew=float(skew), pc1_share=float(eig[0] / N),
        crisis_freq=float(cr.mean()), n_crisis=n_cr,
        tail_coexc=float(ws.tj[idx].sum() / max(ws.tm[idx].sum() * (N - 1), 1e-12)),
    )
    if n_cr >= P.MIN_CRISIS_WINDOWS and no.sum() >= P.MIN_CRISIS_WINDOWS:
        mac, pc1, dens, tail = ws.mac[idx], ws.pc1[idx], ws.dens[idx], ws.ew_tail[idx]
        es_n = _es(tail[no].ravel(), P.TAIL_Q)
        out.update(
            boost_pct=float(100 * (mac[cr].mean() - mac[no].mean()) / mac[no].mean()),
            cvar_ratio=float(_es(tail[cr].ravel(), P.TAIL_Q) / es_n) if es_n > 0 else np.nan,
            pc1_share_gap=float(pc1[cr].mean() - pc1[no].mean()),
            connect_gap=float(dens[cr].mean() - dens[no].mean()),
            mac_crisis=float(mac[cr].mean()), mac_normal=float(mac[no].mean()),
        )
    else:
        out.update({k: np.nan for k in CRISIS_METRICS}, mac_crisis=np.nan, mac_normal=np.nan)
    return out


def compare(g: dict, r: dict) -> dict:
    """Scalar metric values of a sample `g` against the reference sample `r` (both from aggregate)."""
    N = len(g["std"])
    d = g["corr"] - r["corr"]
    off = ~np.eye(N, dtype=bool)
    vals = dict(
        std_ratio=float((g["std"] / r["std"]).mean()),
        corr_fro=float(np.linalg.norm(d)),
        corr_mae=float(np.abs(d[off]).mean()),
        eig_dist=float(np.linalg.norm(g["eig"] - r["eig"]) / np.sqrt(N)),
    )
    for k in ("kurt_med", "acf1_ret", "acf1_sq", "acf5_sq_mean", "skew_ew", "pc1_share", "crisis_freq",
              "tail_coexc", *CRISIS_METRICS):
        vals[k] = g[k]
    return vals


def block_bootstrap_idx(n: int, rng: np.random.Generator, block: int = P.BOOT_BLOCK) -> np.ndarray:
    """Circular block bootstrap over window indices."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def real_reference(ws: WindowStats, B: int = P.BOOT_B, seed: int = 12345, alt_labels: bool = False) -> dict:
    """Real values, bootstrap CIs (stat metrics) and noise ceilings (dist metrics)."""
    rng = np.random.default_rng(seed)
    full = aggregate(ws, alt_labels=alt_labels)
    point = compare(full, full)
    reps = []
    for _ in range(B):
        a = aggregate(ws, block_bootstrap_idx(ws.n, rng), alt_labels=alt_labels)
        reps.append(compare(a, full))
    lo_q, hi_q = (1 - P.CI_LEVEL) / 2, 1 - (1 - P.CI_LEVEL) / 2
    ref = {}
    for k, kind in KIND.items():
        if kind == "disc":
            continue
        x = np.array([r[k] for r in reps], dtype=float)
        x = x[np.isfinite(x)]
        if kind == "stat":
            lo, hi = (np.quantile(x, [lo_q, hi_q]) if x.size else (np.nan, np.nan))
            ref[k] = dict(kind=kind, real=float(point[k]), lo=float(lo), hi=float(hi),
                          half=float((hi - lo) / 2), n_boot=int(x.size))
        else:
            ref[k] = dict(kind=kind, real=0.0, ceiling=float(np.quantile(x, P.CI_LEVEL)),
                          median=float(np.median(x)), n_boot=int(x.size))
    return dict(full=full, metrics=ref)


def normalised_errors(vals: dict, ref: dict) -> dict:
    """Metric -> error in units of real sampling noise (capped). inside = within the real band."""
    out = {}
    for k, v in vals.items():
        if k not in ref:
            continue
        r = ref[k]
        if not np.isfinite(v):
            out[k] = dict(value=None, err=P.NORM_ERR_CAP, inside=False, produced=False)
            continue
        if r["kind"] == "stat":
            e = abs(v - r["real"]) / max(r["half"], 1e-12)
            inside = bool(r["lo"] <= v <= r["hi"])
        elif r["kind"] == "dist":
            e = v / max(r["ceiling"], 1e-12)
            inside = bool(v <= r["ceiling"])
        else:                                   # disc: accuracy above chance vs real-vs-real excess
            e = max(v - 0.5, 0.0) / max(r["half"], 1e-12)
            inside = bool(v <= r["hi"])
        out[k] = dict(value=float(v), err=float(min(e, P.NORM_ERR_CAP)), inside=inside, produced=True)
    return out


def family_scores(errs: dict) -> dict:
    return {f: float(np.mean([errs[m]["err"] for m in ms if m in errs]))
            for f, ms in P.METRIC_FAMILIES.items() if any(m in errs for m in ms)}


# ------------------------------------------------------------- discriminator
def window_features(W: np.ndarray) -> np.ndarray:
    """Order-free summary features of each window (for disc_feat)."""
    W = np.asarray(W, dtype=np.float64)
    n, T, N = W.shape
    ew = W.mean(2)
    sd = W.std(1)
    mu = W.mean(1)
    Z = (W - mu[:, None, :]) / np.clip(sd[:, None, :], 1e-12, None)
    C = np.einsum("ntj,ntk->njk", Z, Z, optimize=True) / T
    off = ~np.eye(N, dtype=bool)
    absC = np.abs(C[:, off])
    eigs = np.linalg.eigvalsh(C)
    ewc = ew - ew.mean(1, keepdims=True)
    ewv = ewc.var(1) + 1e-30
    e2 = ewc ** 2 - (ewc ** 2).mean(1, keepdims=True)
    cum = np.cumsum(ew, 1)
    dd = (np.maximum.accumulate(cum, 1) - cum).max(1)
    disp = W.std(2)
    f = [ew.mean(1), np.sqrt(ewv), sd.mean(1), sd.std(1), disp.mean(1), disp.std(1), disp.max(1),
         absC.mean(1), absC.std(1), eigs[:, -1] / N, eigs[:, -2] / N,
         (ewc ** 4).mean(1) / ewv ** 2, (ewc ** 3).mean(1) / ewv ** 1.5,
         (ewc[:, 1:] * ewc[:, :-1]).mean(1) / ewv, (e2[:, 1:] * e2[:, :-1]).mean(1) / ((e2 ** 2).mean(1) + 1e-30),
         dd, ew.min(1), ew.max(1), (Z ** 4).mean((1, 2)), np.abs(W).max((1, 2))]
    return np.stack(f, 1)


def discriminator_accuracy(real: np.ndarray, gen: np.ndarray, mode: str, seed: int = 0) -> float:
    """Blocked, purged CV accuracy of real (chronological order!) vs generated.

    real is cut into DISC['folds'] contiguous time blocks; training folds drop real
    windows within DISC['purge'] positions of the test block (overlapping windows).
    Generated windows are exchangeable, so they are split at random.  Classes are
    balanced in every fold.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    cfg = P.DISC
    rng = np.random.default_rng(seed)
    nmax = cfg["n_per_class"]
    if len(real) > nmax:                       # thin chronologically, keep order
        keep = np.linspace(0, len(real) - 1, nmax).round().astype(int)
        stride = (len(real) - 1) / (nmax - 1)
        real = real[keep]
    else:
        stride = 1.0
    n = min(len(real), len(gen))
    gen = gen[rng.choice(len(gen), n, replace=False)]
    real = real[:n] if len(real) == n else real[np.linspace(0, len(real) - 1, n).round().astype(int)]
    feat = window_features if mode == "feat" else (lambda a: a.reshape(len(a), -1).astype(np.float32))
    Xr, Xg = feat(real), feat(gen)
    purge = int(np.ceil(cfg["purge"] / stride))
    edges = np.linspace(0, n, cfg["folds"] + 1).astype(int)
    gperm = rng.permutation(n)
    correct = total = 0
    for k in range(cfg["folds"]):
        a, b = edges[k], edges[k + 1]
        tr = np.ones(n, bool)
        tr[max(0, a - purge):min(n, b + purge)] = False
        gte = gperm[a:b]
        gtr = np.setdiff1d(gperm, gte)[: int(tr.sum())]
        if tr.sum() < 20:
            continue
        Xtr = np.concatenate([Xr[tr], Xg[gtr]])
        ytr = np.r_[np.ones(int(tr.sum())), np.zeros(len(gtr))]
        Xte = np.concatenate([Xr[a:b], Xg[gte]])
        yte = np.r_[np.ones(b - a), np.zeros(len(gte))]
        if mode == "raw":                       # compress the flattened window first (fit on train only)
            from sklearn.decomposition import PCA
            pca = PCA(min(cfg["raw_pca"], Xtr.shape[0] - 1, Xtr.shape[1]), random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = HistGradientBoostingClassifier(max_iter=cfg["max_iter"], max_depth=cfg["max_depth"],
                                             learning_rate=cfg["lr"], random_state=seed)
        clf.fit(Xtr, ytr)
        correct += (clf.predict(Xte) == yte).sum()
        total += len(yte)
    return float(correct / max(total, 1))


def real_vs_real_disc(real: np.ndarray, mode: str, n_rep: int = 5, block: int = P.BOOT_BLOCK) -> dict:
    """Noise ceiling for the discriminator: real blocks A vs interleaved real blocks B."""
    n = len(real)
    accs = []
    for rep in range(n_rep):
        off = (rep * block) // n_rep
        blk = ((np.arange(n) + off) // block) % 2
        A, B = real[blk == 0], real[blk == 1]
        m = min(len(A), len(B))
        accs.append(discriminator_accuracy(A[:m], B[np.random.default_rng(rep).permutation(len(B))[:m]], mode, seed=rep))
    accs = np.array(accs)
    hi = float(accs.max())
    return dict(kind="disc", real=float(accs.mean()), lo=0.5, hi=hi, half=float(max(hi - 0.5, 0.02)),
                reps=accs.tolist())
