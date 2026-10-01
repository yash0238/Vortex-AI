"""Stage 4 oracle v3: interleaved blocks span same regimes; 60-window purge at edges."""
import json
import numpy as np
import sys
sys.path.insert(0, ".")
from eval.statistical import evaluate_stylized_facts
from eval.discriminative import discriminative_score

rng = np.random.default_rng(11)
W = np.load("data/raw/windows.npy").astype(float)
lab = np.load("data/raw/window_regimes_v2.npy")
N = len(W)
BS, PH = 100, 30
bounds = list(range(BS, N, BS))
drop = set()
for b in bounds:
    for j in range(max(0, b - PH), min(N, b + PH)):
        drop.add(j)
A_idx, B_idx = [], []
blocks = [list(range(k * BS, min((k + 1) * BS, N))) for k in range((N + BS - 1) // BS)]
kept = [[j for j in blk if j not in drop] for blk in blocks]
# pair adjacent blocks (same regime) and deal one to each set: matched fractions
for p in range(0, len(blocks) - 1, 2):
    first, second = kept[p], kept[p + 1]
    if rng.random() < 0.5:
        A_idx.extend(first)
        B_idx.extend(second)
    else:
        A_idx.extend(second)
        B_idx.extend(first)
if len(blocks) % 2 == 1:
    B_idx.extend(kept[-1])
A_idx, B_idx = np.array(A_idx), np.array(B_idx)
A, B = W[A_idx], W[B_idx]
print(f"N={N} kept A={len(A)} B={len(B)} dropped={len(drop)}")
print(f"crisis frac A={float(lab[A_idx].mean()):.4f} B={float(lab[B_idx].mean()):.4f} overall={float(lab.mean()):.4f}")
out = {}
out["styl"] = evaluate_stylized_facts(A, B)
out["disc"] = discriminative_score(A, B, n_samples=min(500, len(A), len(B)))
for k, v in [("styl", out["styl"]), ("disc", out["disc"])]:
    print(k, {kk: (round(float(vv), 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print("corr floor (real-vs-real): 2.87 | MAE:", round(float(out["styl"]["corr_mae"]), 4),
      "| kurt interval:", [round(float(x), 2) for x in out["styl"]["kurtosis_interval"]])
n = min(500, len(A), len(B))
mu, sd = A[:n].mean(axis=(0, 1)), A[:n].std(axis=(0, 1))
G = rng.normal(mu, sd, size=(n, W.shape[1], W.shape[2]))
out["G_styl"] = evaluate_stylized_facts(A[:n], G)
out["G_disc"] = discriminative_score(A[:n], G, n_samples=n)
for k, v in [("G_styl", out["G_styl"]), ("G_disc", out["G_disc"])]:
    print(k, {kk: round(float(vv), 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
out["meta"] = {"A": len(A), "B": len(B), "dropped": len(drop),
               "crisis_A": float(lab[A_idx].mean()), "crisis_B": float(lab[B_idx].mean())}
from eval.contagion import cvar_regime_ratio
lab_full = np.load("data/raw/window_regimes_v2.npy")
Wfull = np.load("data/raw/windows.npy").astype(float)
rrng = np.random.default_rng(3)
ratios = []
NB = len(Wfull) // 100
for _ in range(200):
    picks = rrng.integers(0, NB, size=NB)
    s = np.concatenate([Wfull[k * 100:(k + 1) * 100] for k in picks], axis=0)
    l = np.concatenate([lab_full[k * 100:(k + 1) * 100] for k in picks], axis=0)
    ratios.append(cvar_regime_ratio(s, l)["cvar_ratio"])
ratios = np.array(ratios)
print("CVaR ratio ref 4.04x, bootstrap 5-95pct:",
      round(float(np.percentile(ratios, 5)), 2), round(float(np.percentile(ratios, 95)), 2))
with open("results/oracle_v3.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("saved results/oracle_v3.json")
