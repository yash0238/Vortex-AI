# Implementation Verification Report

Verification of `implementation_plan.md` (Road to Journal Publication) against the
live codebase. This is a **verification** artifact: it records what was checked, the
evidence, and what remains incomplete. Last verified: 2026-11-18.

Python used throughout: `C:\Users\myind\Documents\GitHub\vortex-ai\.venv\Scripts\python.exe`
(the plan's `Vortex-AI/.venv` path does not exist).

## Summary

| Phase | Status | Evidence |
|---|---|---|
| 1 - Sampler alignment | VERIFIED COMPLETE | `tests/test_sampler_alignment.py` passes; config `stylized_sampler: "seq"`; gradient assertion present |
| 2 - Likelihood dominance | VERIFIED COMPLETE | three source fixes present; `tests/test_likelihood_dominance.py` passes |
| 3 - Corrections & honest reporting | VERIFIED COMPLETE | main.tex corrections applied; bibliography deduped; no "first-to-combine" claims |
| 4 - Remaining experiments | INCOMPLETE (blocked on compute/user) | queue counts below; no `_fs` GPU baselines; no ablation table |
| 5 - Multi-seed aggregation | PARTIAL | analysis.py computes seed mean/sd; `T6_multiseed.md` absent; gate files carry sd but no all-seeds pass rule |
| 6 - Paper finalization | PARTIAL | paper_build now succeeds after stale-JSON fix; RESULT-DEPENDENT / NEED-MACRO markers remain (expected until Phase 4) |
| 7 - Dashboard & docs | PARTIAL | HONEST_SUMMARY.md / README.md / api/server.py present; dashboard not browser-verified |

Tests: `test_sampler_alignment.py + test_likelihood_dominance.py` -> **20 passed**;
`test_regressions.py + test_smoke.py + test_gate_reproducibility.py +
test_g2_episodes.py + test_vares_backtest.py` -> **32 passed**.

## Phase 1 - Sampler alignment

- `training/config.yaml:68` -> `stylized_sampler: "seq"` (confirmed).
- `tests/test_sampler_alignment.py` asserts: training/validation/generation all call
  `emission_sample_seq` when configured; `emission_sample_seq` output receives non-zero
  gradients; `seq` vs `single` produce different loss and gradients.
- Result: all sampler-alignment tests pass. Defect 2 (sampler mismatch) is fixed.

## Phase 2 - Likelihood dominance

Three code-level checks, all confirmed in source:
1. `training/vortex_model.py:231` - `_nll_per_asset()` divides by `max(1, n_stocks)`;
   `training_step` line 330 uses `self._nll_per_asset(nll) + self._eff_beta() * kl`.
2. `training/config.yaml:63` - `cgnsde.beta_max: 0.05`.
3. `models/neural_sde.py:303` - `kappa = sigmoid(kappa_raw).clamp(min=0.6)`.

Note: during the first verification run, 3 tests failed because of *test* bugs
(stray spaces in a space-stripped needle; writing to the read-only LightningModule
`current_epoch` property). Those tests were subsequently corrected by concurrent work
to use a whitespace-normalized needle and `patch.object(..., PropertyMock)`; the suite
now passes 20/20. The implementation itself was never at fault.

## Phase 3 - Corrections

`paper/main.tex` contains, verbatim: weighted objective "rather than a valid ELBO";
"No finite-dimensional marginal (FDM) matching term is implemented"; "per-factor degrees
of freedom were not implemented"; 13 variants; survivorship-bias limitation paragraph;
"correlation is not crisis" treated as hypothesis H1; v2 explicitly "post hoc" throughout;
legacy 6-gate evaluator separated from CrisisBench gates.
`paper/references.bib`: arXiv 2102.03657 and 2109.10072 now appear once each with
`% note-src:` provenance comments; the conflicting 2008.04139 entry is gone.
No "first-to-combine" / "we are the first" strings remain.

## Phase 4 - Experiments (INCOMPLETE)

`python -m crisisbench.queue --list`:

```
v2_main        9 done /   10        v2_risk      0 done /   4
v2_markets     3 done /   19        v2_ablation  0 done /  39
nse_main     105 done /  110        nse_risk    20 done /  44
markets      134 done /  176        nse_ext      3 done /  33
contamination 63 done /  63        sim          18 done /  18
```

Discrepancy with the plan's "Current State Summary": it claims NSE main 110/110 and
risk backtest produced; the live queue shows 105/110 and 20/44. The gate files and T4
risk table do exist, so results were produced from a earlier, larger run and the queue
bookkeeping has since diverged. Treat the queue as the authoritative progress measure.

- `results/runs/` metrics.json counts: nse 174, us 79, kr 58, nse_ext 3.
- No `T2_ablation_*` table (v2_ablation 0/39).
- No `timegan_fs`, `quantgan_fs`, or `diffusion_fs` results: the cloud GPU notebook
  (`cloud/vortex_fullscale.ipynb`) has not been run. This is a user action and gates
  the full-scale neural baseline rows.

Gate verdicts (files in `results/gates/`): G1 v1 FAIL / v2 PASS (both fits);
G2 FAIL for v1 and v2; G3 PASS (`tau_C = 0.80`, CI `[0.60, 1.00]`); G4 PASS is recorded
only in `hypotheses.json` - there is no `GATE_G4.md` file (noted in HONEST_SUMMARY.md).

## Phase 5 - Multi-seed (PARTIAL)

`crisisbench/analysis.py` already computes per-generator seed mean and sd (lines 95-105)
and uses fixed-seed bootstrap intervals. `results/tables/T6_multiseed.md` does not exist.
Gate JSON bodies include per-generator `*_sd`, but gates are not gated on all seeds passing.

## Phase 6 - Paper finalization (PARTIAL, one defect fixed)

Found and fixed a real integration defect during verification: `python -m
crisisbench.paper_build` crashed with `AttributeError: 'str' object has no attribute
'get'` in `tab_hypotheses`. Root cause: `results/tables/hypotheses.json` on disk was a
**stale artifact in the old schema** (plain string descriptions), while current
`crisisbench.analysis_extra.hypotheses()` emits `{verdict, evidence}` dicts. Running
`python -m crisisbench.analysis` regenerated the file in the correct schema; after that,
`paper_build` succeeded:

```
zip: ...\paper\vortex_paper_overleaf.zip 0.22 MB
figures: 7 tables: 9 macros: 87
skipped (inputs missing): fig_ablation, tab_ablation
```

RESULT-DEPENDENT and NEED-MACRO markers remain in main.tex. That is expected: they are
keyed to Phase 4 results (GPU baselines, ablations, markets) that are not yet available.

## Phase 7 - Dashboard & docs (PARTIAL)

Present: `docs/HONEST_SUMMARY.md`, `README.md`, `api/server.py`, `app/` dashboard,
`app/public/data/{core,markets,samples}.json`. Not verified: every dashboard page in a
real browser, and the Model page's v2 text. Browser check is still pending.

## Verification log - second pass (same session)

- `python -m crisisbench.run_risk` initially failed with
  `ModuleNotFoundError: No module named 'arch'`. `arch>=7.0` is declared in
  `requirements.txt:9` but was missing from the venv. Installed `arch 9.0.0`;
  `run_risk` then completed with `G3: PASS`, `tau_C = 0.8 [0.6, 1.0]`.
- After `run_risk`, re-ran `paper_build`: now emits **200 macros** (up from 87),
  7 figures, 10 tables, and a valid 0.22 MB `paper/vortex_paper_overleaf.zip`.
  Only `fig_ablation` / `tab_ablation` are skipped (v2_ablation 0/39).
- `python -m crisisbench.export_web` succeeds (core 0.59 MB, markets 0.57 MB,
  samples 0.19 MB).
- `tests/test_risk.py`: 16 passed. `tests/test_classical_garch.py`: 4 errors at
  setup because the file uses script-style global parameters `R`/`Z` as test
  arguments (docstring says "Run: python tests/test_classical_garch.py") and there
  is no `conftest.py` defining them. This is a pre-existing test-harness issue,
  not an implementation defect, and is outside this plan's scope.

## Verification log - Phase 7 browser pass

Built dashboard served with `npx vite preview` (bound to 4174 because 4173 was
occupied). Browser-checked routes `#/`, `#/benchmark`, `#/model`, `#/honest`, `#/risk`,
`#/glossary`: all render non-empty, none stuck on "Loading results…", and no console
errors on any page. `app/src/pages/Evidence.tsx` already describes v2 explicitly as
"post hoc" with a v1-vs-v2 architecture comparison and an ablation section that shows
an empty state because v2_ablation is 0/39. Phase 7's "Model page describes v2" item
is therefore satisfied; only the full multi-page visual review remains a judgment call.

## Recommended next actions (in order)

1. Run the experiment queue to completion (`crisisbench.queue --stage all --workers 5`);
   this gates ablations, markets, nse_ext and the risk pools.
2. User: run `cloud/vortex_fullscale.ipynb` on a Kaggle/Colab GPU, return
   `fullscale_results.zip`, unzip at project root.
3. Re-run analysis -> run_risk -> analysis -> export_web -> paper_build.
4. Resolve all RESULT-DEPENDENT / NEED-MACRO markers from final numbers.
5. Browser-verify every dashboard page; add T6_multiseed.md and all-seed gate rules.
