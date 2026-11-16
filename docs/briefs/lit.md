# Assignment: literature and fact verification for the research paper

Project root: c:/Users/DELL/Downloads/Major Project/Vortex-AI. Files you own: paper/references.bib,
paper/literature_notes.md. Do not edit anything else. Read docs/STATE.md first (sections "Paper plan" and
"Design decisions") for context; read nothing else in the repo.

Use web search and fetching (load WebSearch and WebFetch via ToolSearch). A reference goes into the .bib ONLY if
you opened a page (arXiv abstract, DOI landing page, publisher or proceedings page) confirming authors, title,
venue and year. Never guess a volume, page range or DOI: omit a field you could not confirm. Be frugal: one
search plus one confirming fetch per item is enough; do not read full papers.

Paper thesis: "Correlation-based fidelity metrics and crisis-dependence metrics rank synthetic multi-asset return
generators differently; a regime-labelled, noise-ceilinged benchmark on Indian equities (NSE), replicated on US
and Korean panels, validated by a VaR/ES backtest; CG-NSDE (graph-conditioned latent neural SDE with a low-rank
multivariate Student-t emission) is the reference method."

Tasks
1. Verified BibTeX entries (keys like yoon2019timegan) for: TimeGAN (Yoon, Jarrett, van der Schaar 2019); Quant
   GANs (Wiese et al. 2020); TimeVAE (Desai et al. 2021); Graph Attention Networks (Velickovic et al. 2018);
   Supervised Contrastive Learning (Khosla et al. 2020); scalable gradients for SDEs / torchsde (Li et al. 2020);
   neural SDEs as infinite-dimensional GANs (Kidger et al. 2021); score-based generative modelling through SDEs
   (Song et al. 2021); GARCH (Bollerslev 1986); DCC (Engle 2002); filtered historical simulation (Barone-Adesi,
   Giannopoulos, Vosper 1999); regime switching (Hamilton 1989); correlations in bear markets (Longin and Solnik
   2001; Ang and Chen 2002); contagion vs interdependence (Forbes and Rigobon 2002); stylised facts (Cont 2001);
   stationary bootstrap (Politis and Romano 1994); Kupiec 1995; Christoffersen 1998; Acerbi and Szekely 2014;
   Fissler and Ziegel 2016; Patton, Ziegel, Chen 2019; Basel traffic-light document (BCBS 1996); Diebold and
   Yilmaz connectedness; absorption ratio (Kritzman et al. 2011); ex-post density estimation / RAE (Ghosh et al.
   2020); Lambert W (Goerg 2015); TSGBench (Ang et al. 2023); Kendall 1938; probabilistic PCA (Tipping and Bishop
   1999); HMM tutorial (Rabiner 1989); SigCWGAN (Ni et al.); one or two recent surveys of synthetic financial data
   / evaluation of financial generative models (2022-2026).
2. Closest prior work on EVALUATING synthetic financial time series, especially anything that evaluates
   CROSS-ASSET dependence under stress/crisis regimes or validates fidelity metrics against a downstream risk
   backtest (2020-2026). Also check these items from the project's old notes and say what each really is, or that
   you could not find it: arXiv 2512.21791, arXiv 2601.12990, arXiv 2401.10370, "FinStressTS", "SFAG", "Financial
   Wind Tunnel", "CoMeTS-GAN" (arXiv 2605.27113), "LFTD" (MDPI AI 2026), "S3G: Stock State Space Graph" (arXiv
   2603.24236), "Sig-Graph GAN", "NANSDE-Net", "Neural Stochastic Flows". One line each.
3. Gap table: rows = the 8 to 12 most relevant prior works; columns = multi-asset? | regime/crisis-conditional
   evaluation? | noise ceiling (real-vs-real)? | held-out crisis test? | downstream VaR/ES validation of metrics?
   | emerging-market data? Fill a cell only if verified; otherwise "n/v".
4. Is the sentence "prior work has not evaluated crisis-time cross-asset dependence of synthetic return generators
   against a noise ceiling" defensible, in need of weakening, or false? Propose the exact wording to use.
5. NSE (India) facts from official NSE / SEBI pages: index-based market-wide circuit breakers (10/15/20%, halt
   durations); price bands for individual securities and specifically whether securities with derivatives (F&O;
   essentially all NIFTY 50 names) have fixed bands or only dynamic bands; T+1 settlement migration dates.

Output: paper/references.bib (verified entries only, each with url, doi or eprint; balanced braces; unique keys)
and paper/literature_notes.md (sections for tasks 2-5, each claim with its source URL; a "could not verify" list).

FINAL REPORT (your return value, at most 350 words): counts of verified vs unverified items, the proposed novelty
wording, the NSE price-band finding, and anything that contradicts the project's existing claims.
