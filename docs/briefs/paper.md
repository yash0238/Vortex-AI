# Assignment: write the research paper (LaTeX, IEEE conference format)

Project root: c:/Users/DELL/Downloads/Major Project/Vortex-AI. File you own: paper/main.tex (and nothing else).
Audience: reviewers at an AI-in-finance venue (ACM ICAIF / IEEE conference). Be precise, sober and honest.

Read, in this order, and nothing else unless you need a detail:
1. docs/STATE.md (project state, design decisions, audit facts)
2. crisisbench/protocol.py (the frozen protocol: quote definitions exactly)
3. the module docstrings (top ~40 lines only) of crisisbench/metrics.py, labels.py, data.py, bench.py,
   run_risk.py, dgp.py, and of each file in crisisbench/generators/ (models, deviations, budgets)
4. crisisbench/generators/cgnsde.py DEFAULTS and the loss/sampling functions (for the method section)
5. paper/literature_notes.md (novelty wording, gap table, NSE facts) and the KEYS in paper/references.bib
   (cite only keys that exist there)
6. crisisbench/paper_build.py (which figures, tables and macros exist and their file names)
7. results/tables/*.md and results/01_dgp/README.md that exist now (partial results: classical models only)

Hard rules
- NEVER type a result number into the text. Every number comes from a macro in paper/numbers.tex
  (\input{numbers}) or from an \input{tables/...} file or a figure. If you need a number that has no macro, add a
  line "% NEED-MACRO: <description>" and phrase the sentence without the number.
- Results are still being computed (neural baselines, CG-NSDE, ablations, risk backtest, Korea). So: write every
  result-independent section completely (abstract skeleton, introduction, related work, benchmark design,
  metrics, label rule, generators, CG-NSDE method with equations, experimental protocol, threats to validity,
  reproducibility, ethics). For results sections write the full structure with \input of the tables/figures and
  neutral descriptive text, and mark every interpretive sentence that depends on a not-yet-known outcome with a
  comment line "% RESULT-DEPENDENT:" followed by what must be decided. Do not guess outcomes and do not write
  that our model wins. Missing table/figure files must not break compilation: wrap each in
  \IfFileExists{...}{\input{...}}{\textit{(pending)}}.
- Framing (from the preregistered plan): the contribution order is (1) an empirical finding to be TESTED:
  correlation-fidelity metrics and crisis-dependence metrics rank generators differently; (2) CrisisBench, the
  regime-labelled, noise-ceilinged, preregistered benchmark with a controlled-ground-truth check and a
  downstream VaR/ES validation; (3) CG-NSDE as a reference method, honestly ablated. Hypotheses H1-H5 and gates
  G1-G4 were frozen before the runs; report them in a table (\input{tables/t8_hypotheses}) including failures.
- Report negative and awkward facts plainly: the wide real-data intervals; the controlled-DGP check showing that
  the boost statistic is compressed and reads about 14% on a no-regime GARCH null (results/01_dgp); survivorship
  bias; Yahoo data; CPU-scale reimplementations of neural baselines; the earlier evaluation flaw (labels handed
  to the generator) that motivated the design; NIFTY-50 stocks have no fixed price bands (so no circuit-filter
  clipping is applied).
- Use the novelty wording from literature_notes.md, not a stronger one.
- Style: IEEEtran conference class, two columns, about 8 pages of content. Packages: booktabs, graphicx, amsmath,
  amssymb, url/hyperref, cite. Tables wider than a column go in table*. No em dashes, no hype words.
- Authors: Yashovardhan Thopte, Alok Shukla, Aaryan Singh, Department of Computer Engineering, Fr. Conceicao
  Rodrigues College of Engineering, Mumbai, India; add a commented-out author block for the project guide
  (name unknown). No email addresses.
- Title: "Correlation Is Not Crisis: Evaluating Synthetic Multi-Asset Returns Under Regime Stress".

Self-check before finishing: every \cite key exists in references.bib; every \input path matches a name produced
by paper_build.py; braces and environments balance (count \begin / \end); no raw result number in the text.
No LaTeX compiler is installed, so check by reading.

FINAL REPORT (your return value, at most 250 words): section list with approximate lengths; the list of
RESULT-DEPENDENT and NEED-MACRO markers you left (file line numbers); anything in the code or results that
contradicts the framing.
