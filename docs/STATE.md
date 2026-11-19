# VORTEX-AI working state (handoff file)

A new Claude session must read ONLY this file to resume. Do not read the PDFs, the legacy docs (README,
PRESENTATION, CLAUDE_HANDOFF, CONVERGENCE_PLAN ...) or any earlier conversation: everything needed is here.
Keep the Status section current. Last update: 2026-11-14.

## How to work on this project (cost rules, set by the user)
- The user is on a usage-limited plan. Keep the session SMALL: read files only when you must edit them, batch
  shell commands, never paste large outputs, no parallel sub-agents (at most one, with a brief in docs/briefs/).
- Long compute (training, the experiment queue) costs zero tokens when it runs as a plain OS process. Start it
  with run_in_background or ask the user to run the command in their own terminal, then end the turn. Do not poll.
- When the session gets long (roughly 100k tokens), update this file and tell the user to /clear and resume.
- The bash tool here breaks on heredocs containing quotes: write scripts with the Write tool, then run them.
- Never run blanket process kills.

## What the user wants
Yash (final-year BE Computer Engg, Fr. CRCE Mumbai; team Yashovardhan Thopte, Alok Shukla, Aaryan Singh):
1. a publishable research project on synthetic NSE stock returns for stress testing;
2. an IEEE-format paper as a .zip for Overleaf;
3. a very good-looking, very informative dashboard that professors and laypeople understand;
4. full training on full data (no toy runs). Ask only if truly blocked.
Honesty rule: never invent or tune numbers; report what the runs give, including failures and failed gates.

## Environment
Windows 11, i5-1235U (10 cores), 12 GB RAM, NO GPU. `.venv/Scripts/python.exe` (Python 3.13, torch 2.14 CPU, arch,
hmmlearn, tabulate). Node 22; `app/` has node_modules. No LaTeX installed. Nothing is committed to git yet (user
has not asked); new work is untracked, old dashboard pages were deleted on purpose.

## Audit facts (why everything was rebuilt)
- Old headline numbers (boost 65%, CVaR 2.16, disc 0.766) are not reproducible: cleaned data and checkpoints were
  lost, tables mixed evaluation setups, and generated windows inherited the crisis label of the real window used
  to condition them. Old model: GAT-conditioned latent neural SDE + rank-8 multivariate Student-t emission; SupCon
  and graph losses were dead. All of this is fixed in the new package `crisisbench/`.

## Paper plan (benchmark-first)
Claim to TEST: correlation-fidelity metrics (family B) and crisis-dependence metrics (family C) rank generators
differently; validated by a VaR/ES backtest; replicated on US and Korea; CG-NSDE is the reference method.
Hypotheses H1-H5 and gates G1-G4 are frozen in `crisisbench/protocol.py`; verdicts are produced by code.
Working title: "Correlation Is Not Crisis: Evaluating Synthetic Multi-Asset Returns Under Regime Stress".

## Code map
- `crisisbench/protocol.py` frozen protocol. `data.py` panels in `data/panels/<market>/` (nse, us, kr, nse_ext,
  sim_boost00/20/40/80, sim_ccc, nse_c_tick/1pct/split all built). `labels.py` crisis rule. `metrics.py` single
  metric module. `bench.py` runner. `queue.py` resumable job queue. `analysis.py` + `analysis_extra.py` tables
  T1/T2/T3/T7, taus, gates G1/G2, hypotheses.json. `run_risk.py` VaR/ES backtest -> T4, T5, gate G3 (written,
  not yet run: needs risk pools). `risk.py` statistics (tests pass). `dgp.py` (tests pass; `--validate` not run).
  `export_web.py` -> `app/public/data/{core,markets,samples}.json`.
- Generators (`crisisbench/generators/`): classical_static (iid_gaussian, t_ppca, stationary_bootstrap),
  classical_regime (regime_gaussian, hmm_gaussian), classical_garch (garch_fhs, dcc_garch): tests pass, all run
  through the runner. `cgnsde.py` (CG-NSDE + VARIANTS for ablations): runs through the runner, about 30-45 s per
  epoch, default 40 epochs max. neural_timevae / neural_quantgan / neural_timegan: files exist but the agent was
  cut off; UNVERIFIED (run tests/test_neural_baselines.py, then a smoke run each).
- Dashboard `app/` (React 19 + Vite, plain CSS tokens, hash router): `src/index.css`, `src/ui/charts.tsx`,
  `src/ui/kit.tsx`, `src/lib/data.ts`, `src/glossary.ts`, `src/App.tsx`, pages `Story.tsx` (Overview, How,
  Glossary, Honest), `Evidence.tsx` (Benchmark, DataPage, CrisisTest, Risk, Model, Markets, Checks), `Lab.tsx`.
  `npm run build` passes. Not yet looked at in a browser. Legacy tool pages MarketExplorer/MutualFunds kept
  (need the old `api/main.py` server). Lab's live button expects POST /api/generate (not written yet).
- Briefs for sub-agents: `docs/briefs/` (common.md, neural.md, review.md).

## Design decisions
- Sampling is unconditional for every generator; crisis labels of generated windows come from the frozen rule on
  the generated returns. CG-NSDE samples [z0, graph embedding] from per-regime Gaussian mixtures fitted on
  training windows.
- Normalised error = |gen - real| / half-width of the real block-bootstrap CI (stat metrics) or distance /
  real-vs-real 95% ceiling (distance metrics), capped at 10; family score = mean; rank by family score.
- Real NSE reference: boost 29% [8, 57] in-sample; 53% [7, 97] on 2020-24 held-out; per-episode COVID 101%
  [65, 140], 2022 shock 49% [31, 72]. Static baselines give 1-2%; regime Gaussian / HMM / GARCH give 4-11%.
- Risk test: for each real day, the 5% of a generator's synthetic (20-day vol, next-day return) pairs nearest to
  the real state give VaR/ES; yearly refits (fits pre2020, upto2020..upto2023); rank by FZ0 at 97.5%.

## Known open issues
- tests/test_cgnsde.py: `test_stratified_batches` assertion fails (line ~216: expects 10 batches of 16 with
  exactly 6 crisis); check whether the test or `stratified_batches` is wrong. All earlier tests in the file pass.
- CG-NSDE after only 2 epochs gives boost about 3% and crisis_freq 26%: judge it only after a full 40-epoch run.
- The independent review of cgnsde.py / risk.py / dgp.py (docs/briefs/review.md) was cut off; redo it inline or
  with one agent before trusting final numbers. `python -m crisisbench.dgp --validate` still has to be run.
- Literature: `paper/references.bib` must contain only web-verified entries; old notes cite several arXiv IDs
  that look unverifiable. Also verify NSE price-band facts (F&O stocks have dynamic, not fixed, bands?).

## Next steps, in order
1. Fix the cgnsde test issue; verify neural baselines (tests + 1 smoke each, check time per fit <= 20 min).
2. Start the full queue (zero tokens; resumable; about 1-2 days of CPU). The user can run this themselves:
       cd "C:\Users\DELL\Downloads\Major Project\Vortex-AI"
       .\.venv\Scripts\python.exe -m crisisbench.queue --stage all --workers 2
   Progress: `.\.venv\Scripts\python.exe -m crisisbench.queue --list`. Failures are in results/queue_log.csv.
3. When stages finish: `python -m crisisbench.dgp --validate`, `python -m crisisbench.analysis`,
   `python -m crisisbench.run_risk`, `python -m crisisbench.analysis` again, `python -m crisisbench.export_web`.
4. Dashboard: `cd app; npm run dev`, look at every page, fix layout problems; write `api/server.py`
   (FastAPI, POST /api/generate using CGNSDE.load on artifacts/models/nse/cgnsde/<fit>/seed0, returns the same
   shape as export_web.sample_block's per-generator dict); add CG-NSDE explain export (latent PCA, attention).
5. Paper: figures (matplotlib, PDF) + LaTeX tables generated from results/tables, IEEEtran conference template,
   verified references only, zip to `paper/vortex_paper_overleaf.zip`. Authors: the three students; guide
   name unknown (ask the user).
6. Rewrite README, update this file, final honest summary: which hypotheses/gates passed or failed and what can
   be claimed.

## Update 2026-11-15 23:55 (read this first)
- The detached queues of 14 Nov died after 2 jobs because the laptop was shut down. Experiments now run from
  `run_experiments.bat` (own console window, `queue --stage all --workers 3`, keeps Windows awake, resumable by
  double-click). The USER must keep the laptop on for about 1.5-2 days. Check with `queue --list`.
- kr garch_fhs failure fixed (effective persistence clipped at 0.999 instead of raising).
- `paper/main.tex` drafted by an agent (808 lines, IEEEtran, not compiled; no LaTeX here). It contains
  `% RESULT-DEPENDENT:` comments (abstract and results sections) and `% NEED-MACRO:` comments (ablation variant
  names, held-out/per-episode real boosts, DGP readings, nse_ext table, contamination flip counts / a T7 LaTeX
  table). After the final analysis: add those macros/tables to `crisisbench/paper_build.py`, resolve every
  marker from the real numbers, rebuild the zip (`python -m crisisbench.paper_build`).
- Awkward facts the paper must state (from partial results, re-check on final ones): GARCH models without
  regimes score better on family C than the HMM, and the boost statistic reads about 14% on a no-regime GARCH
  null, so family C partly measures volatility clustering; the stationary bootstrap has the best in-sample
  family-C score; US tau(B,C) is above 0.3 in-sample; on COVID-2020 every classical generator is far below the
  real interval; protocol deviations (ablations 3 seeds in-sample only, DGP bootstrap 200, risk pools seed 0);
  protocol SAMPLE_START says 2010-11-08 but the panel starts 2010-11-15.
- Still to ask the user: project guide's name and designation for the author list.

## Update 2026-11-16 09:40
- Queue at 09:30: nse_main 110/110, nse_risk 38/44, nse_ablation 0/36, markets 109/176, nse_ext 0/33,
  contamination and sim complete. 31 failed rows in results/queue_log.csv are old (fixed and re-run); re-check
  with `queue --list` rather than the log.
- User asked for all cores. Now two queue windows: the original (3 workers x 3 threads, forward) and
  `run_experiments_extra.bat` (2 workers x 2 threads, `--reverse`). `bench.run` takes a `.claim` file in the run
  folder so two queues never run the same job (stale claims are taken over). `run_experiments.bat` now uses
  5 workers x 2 threads, so after a reboot only that one file is needed. RAM (12 GB) is the limit, not CPU: do
  not add more workers. Intel Xe iGPU is NOT used (CPU-only torch; changing device mid-protocol would make
  seeds incomparable).

## Update 2026-11-16 11:30: v1 CG-NSDE FAILED, v2 in development (READ THIS)
- Main NSE table complete (11 models x 5 seeds x 2 fits). `python -m crisisbench.analysis` verdicts:
  H1 NOT supported (tau(B,C) = 0.49 [0.38, 0.56] in-sample, 0.31 [0.02, 0.38] held-out over 11 models; it was
  negative over the 7 classical models: the three weak neural baselines are bad at everything and pull tau up),
  G1 FAIL, G2 FAIL (CG-NSDE boost 3-4% vs real 29% / 101% COVID), H2 supported (regime baselines beat CG-NSDE),
  H4 supported, H5 supported. CG-NSDE v1 ranks 9/11 on family C in-sample. Stationary bootstrap is best.
- Diagnosis of v1 (from metrics.json info): validation NLL best at epoch 2 of 10 -> the scored model is nearly
  untrained (nu 7.88 = init, kappa 0.118 = init). Cause: z0 (beta 0.01) and the graph embedding g both feed the
  decoder, so it memorises the overlapping windows; and the vol state scales the whole covariance by one
  scalar, which cannot change correlations. Samples are close to iid multivariate t (acf1_sq 0.01, kurt 1.5).
- v2 written: `crisisbench/generators/cgnsde2.py` (registry name `cgnsde2`, VARIANTS for ablations). Graph
  embedding only in the posterior, KL summed with beta 1, latent 16-d, EGARCH-type log-variance state per
  factor + idiosyncratic (correlation rises when factor variance rises), leverage term. 48,810 parameters,
  about 33 s/epoch. v1 file and v1 results are untouched and stay in the paper (v2 is post hoc; say so).
- Development rule for v2 (honesty): tune ONLY with `python -m crisisbench.dev_cgnsde2 --tag X
  [--hparams k=v,k=v]` which fits on the pre2020 fit and compares with the SAME 2010-2019 real windows
  (results/dev_cgnsde2.jsonl). Never look at held-out 2020-24 to choose anything. Dev-period real values:
  boost 14.5, crisis_freq 0.15, kurt 3.2, acf1_sq 0.10, cvar_ratio 1.49. Baselines on dev: stationary bootstrap
  boost 10.1 / acf 0.095 / kurt 3.1; garch_fhs 8.9 / 0.112 / 5.1; regime_gaussian 5.1 / 0.026 / 0.2.
  v2 after ONE epoch: boost 7.4, acf 0.013, kurt 1.6.
- Once v2 is frozen: add `cgnsde2` to queue.py (NEURAL list / ALL, ablations = cgnsde2@variant; drop the
  remaining v1 ablation jobs to save CPU), run it through every stage, re-run analysis; analysis/export/paper
  code refers to generator name `cgnsde` in places (H2, G2, cgnsde_block, api/server.py): extend to v2.

## Update 2026-11-16 evening: v2 FROZEN and queued (supersedes the "v2 in development" notes above)
- User decision: "do whatever you want", go with v2 as the project model, finish ASAP.
- v2 final design (4 dev configs tried, see results/dev_cgnsde2.jsonl; first two failed): static PCA-initialised
  loadings B and idiosyncratic variances d; EGARCH-type log-variance state per factor (K=8, with leverage) and
  per asset; latent SDE path only shifts the K+1 log-variances; graph embedding only in the posterior; KL
  summed, beta 1; dynamics scalars at lr 2e-2; sampling by filtered historical simulation of the model's own
  standardised residuals (`innovations="fhs"`, default; "mt" = parametric, ablation `param_innov`).
  Dev period (real boost 14.5, kurt 3.2, acf1_sq 0.10, crisis_freq 0.15): v2 gives 21.3 / 2.9 / 0.074 / 0.14.
- Queue: new stages v2_main (10), v2_risk (4), v2_markets (19), v2_ablation (39) run from
  `run_experiments_v2.bat` (2 workers x 2 threads). Main window (3 x 3) continues with baselines.
  The extra reverse window was stopped. `results/skip_gens.txt` drops jobs (prefix of gen name or of
  market/gen/fit): all v1 ablations; QuantGAN and TimeGAN on us/kr pre2020 and on nse_ext (1 h each, low value).
  These are PROTOCOL DEVIATIONS to state in the paper. 5 concurrent jobs is the limit (6 caused heavy paging).
- A background waiter in the old session fires when v2_main has 10 metrics.json. In a new session just check
  `queue --list`.
- Code already adapted to v2: analysis.py / analysis_extra.py (v1 keeps the preregistered verdicts; v2 gets
  separate "(v2, post hoc)" rows and GATE_*_v2_posthoc files; T2 ablation table is now the v2 family),
  paper_build.py, run_risk.py, export_web.py (cgnsde block = v2, plus v1_info), api/server.py (loads CGNSDE2).
  NOT yet adapted: family_tau includes v2 among the generators (H1 was preregistered on 11: report both with
  and without v2); dashboard pages (Model page text describes v1); README.
- Paper: Section V rewritten (v1, why it failed, v2 with equations, development protocol, ablation list); uses
  macros \vOneBestEpoch, \vOneEpochs (add to paper_build numbers). nelson1991egarch added to references.bib
  (Crossref-verified). Still to do after results: abstract, intro contributions, Section IV count (12
  generators), protocol deviations paragraph, all RESULT-DEPENDENT / NEED-MACRO markers, title decision (H1 is
  not supported on NSE with all models, so "Correlation Is Not Crisis" may need to change), conclusion.

## Update 2026-11-16 night
- USER: the paper is for a JOURNAL, not a conference (target journal not named yet; asked). Switch main.tex to
  `\documentclass[journal]{IEEEtran}` unless the user names another template; longer text is fine (full
  ablations, replication, v2 development history in the main text).
- User may close VS Code; the two experiment console windows keep running. No watcher after that: on resume
  run `queue --list`, then analysis -> run_risk -> analysis -> export_web -> paper_build.

## Update 2026-11-17: v2 main results in (analysis not yet re-run)
- v2_main 10/10, v2_risk 4/4, v2_markets 2/19, v2_ablation 0/39; nse_risk 44/44 (risk backtest can now run);
  markets 134/176 (12 of the rest are skipped by skip_gens.txt), nse_ext 3/33 (6 skipped).
- v2 family scores from metrics.json (lower is better). In-sample (full, 5 seeds): A 0.73, B 0.31, C 0.145,
  D 0.15. Held-out (pre2020 fit on 2020-24): A 1.21, B 1.27, C 0.74, D 0.39. For comparison the previous best
  on C: stationary bootstrap 0.19 in-sample, DCC 0.87 held-out. So v2 is first on the crisis family in both
  fits; on A it is behind bootstrap/GARCH in-sample. Confirm with `python -m crisisbench.analysis`.
- User restarts the laptop and resumes the queue the next night with `run_experiments.bat` (5 workers, all
  stages including v2). About 48 job-hours left, roughly 10 h wall time.

## Update 2026-11-17 (later): FULL-SCALE BASELINES ON A CLOUD GPU (user wants tier-1, "no laptop excuses")
- TimeGAN / QuantGAN now take the device from env CRISISBENCH_DEVICE (default cpu, behaviour unchanged).
- New baseline `diffusion` (crisisbench/generators/neural_diffusion.py: DDPM, transformer denoiser, DDIM
  sampling; own implementation, CPU smoke passed).
- `crisisbench/fullscale.py` runs timegan (50k iters/phase), quantgan (10k), diffusion (30k) through the
  unchanged runner under labels `timegan_fs`, `quantgan_fs`, `diffusion_fs`: NSE full + pre2020, 5 seeds, then
  the 4 yearly refits (seed 0, risk pools). 42 jobs, resumable, time-guarded, zips results after every run.
- `cloud/vortex_bundle.zip` (2 MB: code + NSE panel + cached reference; rebuild with `python
  cloud/make_bundle.py`) and `cloud/vortex_fullscale.ipynb` (Kaggle or Colab). THE USER must run the notebook
  on a free GPU and bring back `fullscale_results.zip`; unzip it in the project root (it contains results/ and
  artifacts/ paths), then add the `_fs` labels and `diffusion` to analysis.PRETTY/MAIN, run_risk and paper.
- Delete results/runs/nse/*_fs_smoke (pipeline test folders) before the final analysis.
- GPU time per run is NOT measured yet (first cloud session will show it; budgets may need trimming).

## Update 2026-11-17: laptop restarted on the user's request (READ FIRST ON RESUME)
- All experiment processes were stopped cleanly before the restart; no queue is running. Progress then:
  v2_main 10/10, v2_risk 4/4, v2_markets 3/19, v2_ablation 0/39, markets 134/176 (12 skipped),
  nse_ext 3/33 (6 skipped), everything else complete.
- To resume experiments: the USER double-clicks `run_experiments.bat` (5 workers, all stages incl. v2), about
  10 h. Cloud GPU run (cloud/vortex_fullscale.ipynb + cloud/vortex_bundle.zip on Kaggle) is the user's step.
- Nothing is committed to git (user has not asked). Smoke-test folders were already removed.
- Next for Claude once runs are in: unzip fullscale_results.zip into the project root if present; analysis ->
  run_risk -> analysis -> export_web -> paper_build; finish the paper in journal format; dashboard; README.

## Status (2026-11-14 21:10)
- RUNNING DETACHED (survive session end; logs results/queue_run2.log and results/queue_cgnsde.log; if they died,
  restart with `python -m crisisbench.queue --stage all --workers 2`, finished jobs are skipped):
  queue A = neural baselines + remaining classical jobs (about 15 h); queue B = CG-NSDE main, risk refits,
  ablations (in-sample fit only, 3 seeds), other markets, nse_ext (about 1-2 days).
- Done since last update: neural baselines verified by tests (TimeGAN default 400 iters/phase, QuantGAN 500
  iters, TimeVAE 100 epochs); CG-NSDE core reviewed by the lead (density, sampler, vol recursion, scaler OK;
  batch-sampler duplicate bug fixed); small-sample PCA bug in metrics.discriminator fixed (nse_ext / upto2023
  jobs that failed are re-queued); `dgp --validate` run: boost metric is monotone but compressed (true 80% reads
  38%) and reads 13.6% [5.8, 20.6] on the no-regime GARCH null (sim_ccc): MUST be reported in the paper;
  `api/server.py` written (POST /api/generate; import tested); `crisisbench/paper_build.py` written and run on
  partial results (figures, tables, numbers.tex, zip when paper/main.tex exists); literature done:
  paper/references.bib (64 verified entries) + paper/literature_notes.md (novelty wording; NIFTY-50 F&O stocks
  have NO fixed price bands). Dashboard overview page checked by headless-Edge screenshot: renders correctly.
  Paper draft agent launched (brief docs/briefs/paper.md -> paper/main.tex, with RESULT-DEPENDENT markers).
- 3 kr garch_fhs pre2020 jobs failed for an unknown reason: read their `tail` in results/queue_log.csv.
- After the queues finish: step 3 of "Next steps" (analysis, run_risk, analysis, export_web, paper_build), then
  finalise the paper's result-dependent text, check remaining dashboard pages by screenshot
  (Edge headless via PowerShell Start-Process works; `npx vite preview --port 4173 --host 127.0.0.1` in app/),
  add the CG-NSDE explain export (latent PCA), rewrite README.
- [x] audit, protocol frozen, panels built, metric pipeline verified
- [x] nse_main: 42 of 110 jobs done (static baselines complete; classical regime/GARCH partly; neural none)
- [x] dashboard code written and building; analysis T1-T3, T7, hypotheses code written
- [ ] verify neural baselines; cgnsde test fix; independent review
- [ ] full queue; risk backtest; DGP validation; final analysis + export
- [ ] browser check of dashboard; API server; paper + zip; README

## Update 2026-11-17 (evening): pushed to GitHub main
- Commit 1745f6e on main (author YashO238): all CrisisBench code, results/runs (389 runs), tables, dashboard,
  paper, cloud runner. artifacts/ and raw vendor downloads are gitignored. Old root docs moved to docs/archive/,
  new README.md. v0 pipeline folders kept (api/main.py uses models/, training/, data/).
- Project report draft for the college template: Claude Doc "VORTEX-AI Project Report (Draft)",
  https://claude.ai/artifact/Js6ybWoJ6LSuCc7euDHgxu (plain-language; Pending items marked).
- Dashboard: `start_dashboard.bat` (API on 8000 + vite preview on 4173). Rebuild with `cd app; npm run build`
  after `python -m crisisbench.export_web`.

## Update 2026-11-17 (night): queue running, risk backtest, journal format
- run_experiments.bat started (5 workers x 2 threads, CPU 97-99%, ~1.8 GB free; Edge + VS Code hold ~4.6 GB).
  Left at start: v2_markets 16, v2_ablation 39, markets 30, nse_ext 24 (about 109 jobs, roughly 11 h).
- risk.py: a method whose ES forecast is <= 0 (TimeVAE on every day) gets VaR tests only; Z2/FZ0 = NaN and it is
  left out of the FZ0 ranking (printed + es_nonpositive_days column in T4). run_risk writes T5/G3 for registered
  generators and T5_metric_validity_with_v2 / GATE_G3_v2_posthoc / G3_with_v2.json including post hoc rows.
- analysis.POSTHOC = (cgnsde2, timegan_fs, quantgan_fs, diffusion_fs): in every table, excluded from preregistered
  H1/G3 verdicts, reported alongside. PRETTY has the _fs labels for the Kaggle results.
- paper_build: macro names sanitised (digits -> words, e.g. \boostCgnsdeTwo, \verdictHTwovTwoposthoc); new
  macros heldBoost*, heldScoreC*, rankC*, rankA*, realHeldBoost*, realCovid/Shock/MidBoost*, vTwoCovidBoost,
  dgp*Rule/Lo/Hi, vOne/vTwo BestEpoch/Epochs/Params.
- paper/main.tex switched to \documentclass[journal]{IEEEtran} with journal author block.
- WAITING FOR: user's Kaggle fullscale_results.zip (unzip into project root; it holds results/runs/nse/*_fs and
  artifacts/...). Then: analysis -> run_risk -> analysis -> export_web -> paper_build; resolve RESULT-DEPENDENT /
  NEED-MACRO markers in main.tex; update report doc chapter 7; rebuild dashboard; commit as YashO238.
- 2026-11-17 16:00: skip_gens.txt now also matches market/gen/fit/seedK; seeds 3-4 of laptop-scale QuantGAN/TimeGAN on us/kr full skipped (protocol deviation: 3 seeds for those 4 cells). Estimated finish of the queue about 03:00-04:00 on 18 Nov. User asked whether v1 must appear in the journal paper: recommendation given = keep v1 as one 'initial design' row/paragraph, never call v2 preregistered.
- 2026-11-17 evening: Edge closed on user's request; 6th worker added via run_experiments_extra.bat (1 worker x 2 threads, --reverse). After a reboot, run_experiments.bat alone is enough (5 workers); add the extra window only if >2 GB RAM is free.
- 2026-11-18 01:00: overnight extension. New queue stage v2_ablation_held (13 v2 ablations x pre2020 x seeds 0-2 =
  39 jobs, scored on 2020-24). Restored the 8 skipped GAN seeds (us/kr full seeds 3-4; only kr ones were missing).
  Still skipped (deviation, state it): QuantGAN/TimeGAN laptop-scale on us/kr pre2020 and on nse_ext.
  Second run_experiments.bat window started 01:00 (5 workers); expected finish about 08:30-09:30.
  paper_build tab_ablation uses T2_ablation_nse_full only: add the pre2020 ablation table (T2_ablation_nse_pre2020)
  to the paper after these runs.
- 2026-11-18 01:00 IMPORTANT: with all Korea TimeGAN seeds 0-2 in, KR tau(B,C) rose from 0.29 to 0.53, so H4 is
  not supported, G4 FAIL, and the preregistered KILL criterion (tau >= 0.3 on every market) is TRIGGERED.
  Frozen consequence: drop the "correlation is not crisis across generators" benchmark-headline framing.
  Note for the write-up: static models still rank low on family C (iid 8th, t-PPCA 7th); the high tau comes from
  the weak neural baselines being bad on every family. What survives and should lead the paper: (1) downstream
  validity: family C predicts VaR/ES quality (tau 0.56 [0.29, 0.78]) while family B does not (-0.07 [-0.38,
  0.29]); preregistered H3 fails only because the intervals touch at 0.29; (2) CG-NSDE v2 ranks 1/12 on family C
  in-sample and held-out and 1/11 on the risk backtest (post hoc model). Title must change. Re-check after the
  two remaining KR GAN seeds and the Kaggle rows. Discuss the reframing with the user before writing.

## Step 0 (crash-prediction idea), 2026-11-18 02:30
- Prereg docs/step0_prereg.md (1 logged change: daily count capped at K). Code crisisbench/step0.py,
  results in results/step0/. Data: Kenneth French 49 industries daily 1926-07..2026-09 (data/external/french)
  + nse_ext panel.
- Rule verdict: PASS (P1: mean log score cascade -1.46 vs EVT -1.67 vs no-cascade -2.13 over 1025 blocks;
  P2: 3/3 records: 2008-09 M1 8.8% vs EVT 0.33%; 2020-02 2.6% vs 0.47%; NSE 2020 5e-5 vs 1e-6 floor).
  F1 FAILED (5-y branching ratio before 2008 and 2020 was BELOW the median: 0.57, 0.72 vs 0.81).
  F2 passed weakly (decay exponent 0.64 after non-precursor crashes vs 0.10 after precursor ones; 3 vs 7 events).
- Critical caveat (Claude's reading): the model state at record block starts was 85x (2008) and 40x (2020)
  the median: the crashes had already begun (Lehman, 24-25 Feb 2020). The win is "amplification given onset",
  which conditional EVT / GARCH-type forecasts (McNeil-Frey 2000) may also deliver. Not yet evidence for the
  novel claim. Proposed next (post hoc, to log): state-conditional EVT baseline + lead-time test (forecast one
  block before the record). Possible reformulation: exogenous onset (unpredictable, EVT) + endogenous
  amplification (cascade measured on ordinary days). Training branching ratio is near critical (0.93-0.95).

## 2026-11-18 ~04:00 Training finished (all queue stages done 03:44, 0 failures)
- Watcher ran analysis, export_web, paper_build (184 macros, Overleaf zip rebuilt), dashboard build: all OK.
- Final verdicts unchanged: H1 no, H2 yes, G1 FAIL, H3 no, G3 FAIL, H4 no (US 0.60, KR 0.53), G4 FAIL, KILL triggered,
  H5 yes, G2 FAIL. Post hoc v2: H3 supported (0.64 vs 0.09, FZ0 rank 1/11), H2(v2) "not supported" = v2 BEATS the
  simple regime models (0.15 vs 0.53/0.70; good for us), G1(v2) PASS, G2(v2) FAIL.
- Held-out ablation (T2_ablation_nse_pre2020, 3 seeds): what matters = per-factor vol (scalar_vol C 0.90 vs 0.74),
  FHS innovations (param_innov C 0.82, D 1.31), latent SDE (no_sde C 0.79), vol dynamics (no_voldyn A 1.43).
  GAT / SupCon / graph loss / leverage / K make no held-out difference (no_gat C 0.73). Report honestly.
- Waiting on: user's answer on Step 0 follow-up checks; Kaggle fullscale_results.zip (tonight).

## 2026-11-18 afternoon: Kaggle merge, diffusion bug, v3 (graph-Hawkes cascade)
- Kaggle zip (C:\Users\DELL\Downloads\fullscale_results.zip, copy in data/external/kaggle/, gitignored) merged:
  timegan_fs + quantgan_fs (5 seeds full/pre2020 + yearly refits). TimeGAN at 50k iters still fails (disc 0.99,
  Gaussian tails); QuantGAN full scale is decent (C rank 3 in-sample, FZ0 rank 2 of generators).
- diffusion_fs DISCARDED: sampler diverged (std 785x real). Cause: cosine schedule alpha_bar_T = 2.4e-9, epsilon
  parametrisation divides residual error by 5e-5. Fix in neural_diffusion.py: x0 parametrisation (as Diffusion-TS)
  + x0 clip to max |train z|. New Kaggle notebook cloud/vortex_diffusion_rerun.ipynb (+ rebuilt vortex_bundle.zip),
  user must run it (~3 h GPU) and send fullscale_diffusion.zip -> unzip in project root -> analysis pipeline.
- paper_build: new macros (tau intervals, post hoc taus, contamination counts, risk ranks, held-out ablation),
  t2_ablation now has in-sample + held-out columns, t4_risk at 97.5% (frozen ranking level).
- User direction (18 Nov): fix the model/results first, LaTeX last; wants CG-NSDE as a genuinely novel star.
- v3 = crisisbench/generators/cgnsde3.py: v2 + graph-Hawkes cascade in the log-variance states (tail events
  e_ti = 1[z_ti < q_i] add kappa_s e_i + kappa_g (A e)_i to idio log-var, kappa_m mean(e) to factor log-vars;
  A = learnable sparse kNN contagion graph). Motivation: v2 ACF(r^2) 0.075 vs 0.142, held-out COVID boost 22 vs 101,
  GAT irrelevant in ablation; Step 0 near-critical cascade. POST HOC; dev on pre2020 fit only
  (results/dev_logs/, results/dev_cgnsde2.jsonl gen=cgnsde3). Queue stages v3_* (81 jobs) added, ORDER starts with v3.
- crisisbench/figures_extra.py: fan chart (f7), contagion heatmap (f8, needs v3 checkpoint), strategy sandbox +
  leverage/lead-lag table (f9, T9_sandbox_facts). v2 leverage -0.13 vs real -0.19 (GARCH-FHS -0.02).
- diffusion x0 fix CPU test (1500 iters): std ratio 1.63 (was 785 / 11): sampler stable; bundle rebuilt, safe to run on Kaggle.
- v3 DEV RESULTS (pre2020 fit only; real: boost 14.5, acf1_sq 0.101, kurt 3.22, cvar 1.49). v2 ref: val 58.385,
  acf 0.074, kurt 2.92, boost 21.3. v3 k5 s0/s1: val 58.363/58.348, acf 0.078/0.081, kurt 3.1, boost 24.
  v3 k10: val 58.346 (best), acf 0.111, kurt 4.43, boost 25.7, cvar 1.94. tail_q 0.025: val 58.389 (worse).
  In every run kappa_self -> ~0.008 while kappa_graph -> 0.5-1.5: likelihood prefers NEIGHBOUR contagion.
  FROZEN (before any held-out look): contagion_k = 10, chosen by validation ELBO. no_regime variant added.
- VORTEX Money consumer app (18 Nov): app/money.html + app/src/money/* (engine.ts maths in browser, screens: Onboarding,
  Home w/ Money Health score + badges + XP + quiz, Goal planner MC, Crash test + SIP-through-crash + history replay,
  Diversify calm-vs-crash meter + network, Reality/tip check, Portfolio editor incl. real MF search via mfapi.in).
  Data: crisisbench/export_money.py -> app/public/money/engine.json (v2 scenarios; growth assumption 10.5%/yr, own
  stock means removed; stress = worst 5% of 24k generated quarters by EW drawdown). Rerun with --gen cgnsde3 after v3.
  Launcher start_money.bat (port 4174). Sanity: basket 1y median +10.6% [-8%, +31%]; bold crash bad -25%.
- VORTEX Money flagship pass (18 Nov evening): backend money_api/app.py (FastAPI: accounts scrypt+HMAC tokens, state
  sync with version conflict 409, cached mfapi proxy, rate limits, security headers, serves app/dist; tests
  tests/test_money_api.py 6/6 pass; Dockerfile.money). New screens: Fund X-ray (real NAV: CAGR, worst year, max DD,
  NAV + drawdown charts, model crash test), Retirement SWP ("will my money last", safe withdrawal), behaviour-gap
  replay (2020 panic sell) in Crash, chance-vs-SIP bars + outcome histogram in Goal, missions/streak/health history/
  share card/level titles/toasts/level-up in Home+App, account sync sheet, mobile More menu, brand (mark, icons,
  og.png, manifest, docs/brand.md), motion polish. start_money.bat now runs the backend on :8100.
- v3 HELD-OUT RESULTS (18 Nov 18:15; T2_ablation_v3_nse_pre2020, 3 seeds; v3 main 5 seeds): C / A
  v2 0.742/1.211; v3 0.676/0.964; no_cascade 0.732/1.210; no_graph_cascade 0.746/1.212 (graph contagion carries the
  whole gain); no_mkt_cascade 0.730/1.160; uniform_graph 0.683/1.096 (which neighbours matters for A, not C);
  no_gat 0.672/0.906 (GAT encoder irrelevant); no_supcon 0.695/1.032; scalar_vol 0.898/1.383;
  no_regime 0.638/0.878 (LABEL-FREE IS BEST held-out). In-sample v3 A 0.41 (v2 0.73), C 0.165 (v2 0.145).
  Do NOT promote no_regime on held-out evidence (test-set selection). Confirmatory test to pre-register: no_regime vs
  full v3 on us/kr pre2020 held-out family C, 3 seeds; prediction: no_regime <= full on both.
- App now on v3 scenarios: bold-mix severe-crash bad case -30% (v2 -25%; real COVID for the same mix -31%); stress pool worst-5% EW drawdown median -41%.

## 2026-11-18 ~22:30 ALL RUNS FINAL (87 v3 jobs + 6 confirmatory + diffusion_fs merged; 0 failures)
- CONFIRMATORY TEST FAILED: no_regime vs full v3, held-out C: us 0.602 vs 0.647 (pass, within noise), kr 0.518 vs
  0.445 (FAIL, ~3 sd worse). "Crises emerge without labels" is NOT established; report as non-replicating.
- Held-out C (pre2020): nse v3 0.68 (best) > v2 0.74 > dcc 0.87; kr v3 0.44 (best) > boot 0.47 > v2 0.50;
  us v2 0.43 (best) > boot 0.65 = v3 0.65. In-sample C: nse v2 0.15, v3 0.16, boot 0.19; us boot 0.28, v3 0.33;
  kr boot 0.13, reg-G 0.45, v3 0.54.
- Risk FZ0 97.5 EW: GARCH-FHS benchmark -3.495, v3 -3.381 (best generator), v2 -3.343, QuantGAN-GPU -3.259.
- Metric validity with all 15 incl post hoc: tau C 0.68 [0.45,0.83], D 0.75, A 0.52, B 0.31 [-0.01,0.50]:
  G3-with-posthoc now FAIL (overlap). Registered 10: C 0.56 [0.29,0.78], B -0.07 [-0.38,0.29] (FAIL, touch).
- diffusion_fs (x0-param fix): std 2.3x, corr_fro 21, kurt 9.3, boost 24 (in band), disc 0.95.
- Next: rewrite paper/main.tex on frozen results (title/abstract reframed), project report, commit + push.
- 18 Nov late: paper/main.tex rewritten on final results (all numbers via macros; 399 macros; refs added); compile
  not verified locally (running a downloaded TeX binary was blocked) -> user compiles the Overleaf zip.
  README rewritten; project report doc updated (status, acceptance table incl. H4/G4 now FAIL, v3 in 6.5, new 6.8
  VORTEX Money, chapter 7 final results, abstract, future work). app/public/money/engine.json gitignored
  (vendor-derived real returns). Commit + push NOT done: blocked by permission classifier, awaiting user approval.

## 2026-11-18 ~22:45 STRONG BASELINES (user: "make a strong baseline", per research plan sec. 4 + masterplan I.9/I.12)
- crisisbench/generators/classical_strong.py (post hoc rows): gjr_evt_tcopula (GJR-GARCH-t + GPD-tail margins +
  t copula, McNeil-Frey/Demarta-McNeil), adcc_gjr (Cappiello-Engle-Sheppard asymmetric DCC, FHS innovations),
  rs_tcopula (2-state HMM on EW return; per-state t copula AND per-state margins -> common vol + corr switching).
  Smoke (nse full): rs_tcopula boost 32.5 (real 29.3), cvar 2.24 (2.20) -> genuinely strong. Queue stage "strong"
  (90 jobs, run_strong.bat); watcher b9unsoobo reruns analysis/run_risk/export/paper_build when done.
- Official Diffusion-TS (ICLR 2024): cloud/dts_driver.py (authors' classes + energy.yaml config, seq 60, 47 feats,
  25k steps, 1000 sampling steps), cloud/vortex_diffusion_ts.ipynb (smoke test first), bundle rebuilt;
  crisisbench/generators/external.py scores the official samples; queue stage "dts" (14 jobs) after the user's
  Kaggle run returns diffusion_ts_results.zip -> unzip into project root -> queue --stage dts -> pipeline.
  Local pip install / running third-party code was blocked by the permission classifier (so no local smoke test).
- STRONG BASELINE RESULTS (90/90 ok): held-out C nse v3 0.68 (best) > v2 0.74 > dcc 0.87 > adcc/rs/garch/boot 0.92;
  kr rs_tcopula 0.43 vs v3 0.44 (tie) ; us v2 0.43 > rs 0.50 > boot/v3 0.65. In-sample nse rs 0.49, adcc 0.60.
  Risk FZ0: GARCH bench, v3, v2, rs_tcopula (3rd gen), QuantGAN-GPU, DCC, ADCC, GARCH-FHS, GJR-EVT-tcop.
  Metric validity (18 gens): C 0.67, D 0.74, A 0.57, B 0.23. Paper/README corrected (no "best in Korea").
- 19 Nov: official Diffusion-TS merged (4 zips; smoke output had overwritten full/seed0 -> discarded; driver fixed to
  write smoke to _smoke/). Valid: full seeds 1-3, pre2020 seeds 0-3 (2000 windows each, stocks config, GPU,
  10k steps). Result: std 0.67-0.71x real, crisis windows 0.6-0.75% (real 15%) -> crisis metrics not produced,
  family C ~7.1 (bottom), family B 0.86 (decent), disc 0.93. Paper generators + results text updated.

## 2026-11-19 — version labels removed
- Paper, figures, tables, dashboard, report: no "v1/v2/v3". Model described as a design path:
  registered design -> Change 1 (bottleneck + factor-volatility states + FHS) -> Change 2 (graph-Hawkes contagion) = final CG-NSDE.
- New tables: paper tab:path (design path), t10_avgrank (avg rank over 6 cells: CG-NSDE 2.3, worst 4).
- Report doc rev 59: same relabel + design-path table + average-rank table.
- Rebuild OK (11 figures, 12 tables, 489 macros; vite build OK).
- Remaining: Overleaf compile check, commit/push approval, guide name in report, "to be checked" test rows, optional app deploy.

## 2026-11-19 (later) — dashboards re-checked, panel script, report tests
- Browser check of all 14 research-dashboard pages and all 7 VORTEX Money screens (desktop + phone). Fixes:
  - Our model page rewritten as the design path (first design -> Change 1 -> Change 2) + final-model ablations (ablation_v3 exported).
  - export_web relabels hypothesis/gate text (no v1/v2/v3 in core.json).
  - Mutual funds page: mfapi.in blocks browser CORS -> Vite dev/preview proxy /mfapi.
  - api/server.py live generation now uses cgnsde3 (was cgnsde2), warms model at startup; Lab requests 200 windows (5.3 s).
- Screenshots: docs/screenshots/ (29 files).
- Panel script doc: https://claude.ai/artifact/FvEqJitEb8XX4mTsmS3yZU
- Report test rows filled (all Pass) + VORTEX Money test table; backend tests 6/6.
- Deployment: NOT done; needs user's hosting account + decision on public data use.
