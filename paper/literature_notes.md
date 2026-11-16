# Literature and fact-verification notes

Checked 2026-11-14. Companion to `paper/references.bib`.

Scope of verification, stated once so it is not over-read:
- Bibliographic data (authors, title, venue, year) come from a page that was opened: arXiv abstract page, Crossref
  record, proceedings page or official PDF. The source is recorded above each entry in the .bib.
- Statements about what a paper DOES (data, metrics, evaluation design) come from its ABSTRACT only. Full texts
  were not read. "n/v" means "not verifiable from the abstract", not "the paper does not do it".
- The first pass (previous agent) built most of the .bib. In this pass nine of its entries were re-opened and all
  matched: arXiv 2401.10370, 2601.12990, 2203.01664, 2512.21791, 2605.27113, 2109.10072, 2508.02758, 2410.09850
  and Crossref 10.1145/3604237.3626895. The remaining first-pass entries were kept on the strength of their
  recorded url/doi/eprint and were not re-opened.

## Task 2a. Closest prior work on evaluating synthetic financial time series

| Key | What it is (from the abstract) | Source |
|---|---|---|
| cont2022tailgan | Tail-GAN. Trains a generator for "high-dimensional multi-asset scenarios" using the joint elicitability of VaR and ES; claims correct tail risk for a broad class of trading strategies; synthetic and market data (datasets not named in abstract). VaR/ES is the training objective. Comment says a version is for Management Science; journal details not confirmed. | https://arxiv.org/abs/2203.01664 |
| ericson2024var | Comparative review of deep generative models for VaR. Data: historical USD YIELD CURVE plus simulated GARCH and CIR data (not equities). KPIs: distribution distance, autocorrelation, backtesting. Finding: historical simulation, GARCH and CWGAN perform best. | https://arxiv.org/abs/2401.10370 |
| flaig2022scenario | GAN-based economic scenario generator as a full Solvency II internal market risk model, one-year horizon; results "similar to regulatory approved internal models in Europe". Journal ref on arXiv: Risks 10(11):199 (2022). | https://arxiv.org/abs/2109.10072 |
| ang2025ctbench | CTBench. Crypto benchmark: 452 tokens, 8 models, 13 metrics in 5 dimensions (forecasting accuracy, rank fidelity, trading performance, risk assessment, efficiency), evaluated "over four distinct market regimes"; reports trade-offs between statistical fidelity and profitability. Regimes are not named in the abstract. | https://arxiv.org/abs/2508.02758 |
| caulfield2024systematic | Systematic comparison of deep generative vs parametric models for MULTIVARIATE financial series, on synthetic data of increasing complexity and empirical data; distances on moments of full and rolling series; implied-volatility trading task. | https://arxiv.org/abs/2412.06417 |
| huh2026marketgans | MarketGANs. Joint generation of high-dimensional daily US equity returns; evaluated on stylised facts including "cross-sectional correlation structures and tail co-movement across assets"; baseline is factor-model bootstrap; downstream is covariance estimation. | https://arxiv.org/abs/2601.17773 |
| alzahrani2025marcd | MARCD. Regime-conditioned diffusion for CVaR-constrained allocation on liquid multi-asset ETFs 2005-2025, walk-forward, 2020-2025 out of sample; headline is max drawdown 9.3% vs 14.1% for Black-Litterman. Evaluation is at portfolio-decision level, not generator fidelity. | https://arxiv.org/abs/2510.10807 |
| cetingoz2025synthetic | Argues standard generator evaluation misses what matters for portfolios (especially long-short); large US equity universe; proposes a "regurgitative training" test and a mean-reversion strategy check. Closest in SPIRIT to our "metrics disagree" thesis. | https://arxiv.org/abs/2501.03993 |
| masi2026gandiffusion | Introduces CoMeTS-GAN (conditional GAN for mid-price and volume of correlated stocks) and a GAN-guided diffusion model; claims better stylised facts and inter-asset correlations. Datasets and metrics not named in abstract. | https://arxiv.org/abs/2605.27113 |
| masi2023correlated | ICAIF 2023 paper "On Correlated Stock Market Time Series Generation", pp. 524-532. Bibliographic record only (Crossref has no abstract); content not verified. | https://api.crossref.org/works/10.1145/3604237.3626895 |
| zhang2026sfag | SFAG, see 2b. Shows that visually realistic GAN output "break[s] down under trading backtests". | https://arxiv.org/abs/2601.12990 |
| hounwanou2025synthetic | See 2b. | https://arxiv.org/abs/2512.21791 |
| kwon2024stylized | Tests whether GANs learn stylised facts on univariate and multivariate series; result depends strongly on generator architecture. | https://arxiv.org/abs/2410.09850 |
| alouadi2026sbbts | SBBTS (Schrodinger-Bass). Heston simulations and S&P 500; recovers stochastic-volatility and correlation parameters; augmentation improves classification accuracy and Sharpe ratio. | https://arxiv.org/abs/2604.07159 |

Also surfaced by search but NOT opened, so not in the .bib: "The ETS challenges" (arXiv 1811.07792), LOB-Bench,
"Long-Range Dependence in Financial Markets" (arXiv 2509.19663), "Diachronic Sample Integration" (arXiv
2607.10810 was opened: tail-risk estimation by checkpoint mixing, no relevant evaluation design in abstract; not
added), a Rusnak EPFL master's thesis on C-MTE-GAN.

## Task 2b. Items from the project's old notes: what each really is

| Old-note item | Reality | Source |
|---|---|---|
| arXiv 2512.21791 | EXISTS. Hounwanou, Gaba, Ntakirutimana, "Synthetic Financial Data Generation for Enhanced Financial Modelling" (Dec 2025). Compares ARIMA-GARCH, VAE, TimeGAN on S&P 500 daily data (single index); MMD, autocorrelation, volatility clustering; downstream mean-variance optimisation and volatility forecasting. No crisis or VaR evaluation in abstract. | https://arxiv.org/abs/2512.21791 |
| arXiv 2601.12990 | EXISTS. Zhang et al., "Beyond Visual Realism: Toward Reliable Financial Time Series Generation" (Jan 2026, "Accepted by ICASSP 2026"). This IS the SFAG paper. | https://arxiv.org/abs/2601.12990 |
| arXiv 2401.10370 | EXISTS. Ericson et al., VaR comparative review; USD yield curve, not equities. | https://arxiv.org/abs/2401.10370 |
| "FinStressTS" | EXISTS (arXiv 2606.03184, comment "KDD 2026 (Oral)") but it is a parametric synthetic benchmark for FORECASTING models, not a benchmark of generators. Entry from first pass; not re-opened in this pass. | https://arxiv.org/abs/2606.03184 |
| "SFAG" | Stylized Facts Alignment GAN = arXiv 2601.12990. Single series: Shanghai Composite Index 2004-2024; momentum-strategy backtest; no VaR. | https://arxiv.org/abs/2601.12990 |
| "Financial Wind Tunnel" | EXISTS (arXiv 2503.17909), "A Retrieval-Augmented Market Simulator", Cao et al. 2025. First pass; not re-opened. | https://arxiv.org/abs/2503.17909 |
| "CoMeTS-GAN" (arXiv 2605.27113) | The ID is real but the paper's title is "High-Quality Synthetic Financial Time-Series using a GAN-Diffusion Framework" (Masi, Coletta, Bartolini, May 2026); CoMeTS-GAN is introduced inside it. Cite by the real title. | https://arxiv.org/abs/2605.27113 |
| "LFTD" (MDPI AI 2026) | EXISTS. Choi et al., AI 7(2), article 60, DOI 10.3390/ai7020060. First pass (Crossref); not re-opened. | https://doi.org/10.3390/ai7020060 |
| "S3G" (arXiv 2603.24236) | EXISTS but it is a stock trend PREDICTION model, not a generator. First pass; not re-opened. | https://arxiv.org/abs/2603.24236 |
| "Sig-Graph GAN" | Mapped by the first pass to arXiv 2605.22215, Gregnanin et al., "A Generative Adversarial Graph Neural Network for Synthetic Time Series Data". Not re-opened; confirm the name "Sig-Graph GAN" appears in the paper before using that label. | https://arxiv.org/abs/2605.22215 |
| "NANSDE-Net" | EXISTS (arXiv 2602.08182, Ozai and Nakagawa, "PAKDD2026 Accepted"). First pass; not re-opened. | https://arxiv.org/abs/2602.08182 |
| "Neural Stochastic Flows" | EXISTS (arXiv 2510.25769, Kiyohara, Johns, Li, "NeurIPS 2025 (poster)"). First pass; not re-opened. | https://arxiv.org/abs/2510.25769 |

## Task 3. Gap table

Cells are filled from abstracts only. "yes"/"no" only where the abstract settles it; otherwise n/v.

| Prior work | Multi-asset? | Regime/crisis-conditional evaluation? | Noise ceiling (real-vs-real)? | Held-out crisis test? | Downstream VaR/ES validation of metrics? | Emerging-market data? |
|---|---|---|---|---|---|---|
| Tail-GAN (cont2022tailgan) | yes | n/v | n/v | n/v | VaR/ES used as training objective; validation of other metrics n/v | n/v |
| VaR review (ericson2024var) | n/v (yield curve) | n/v | n/v (historical simulation is a competitor, not a ceiling) | n/v | VaR backtesting is a KPI; not used to validate other metrics | no (USD yield curve) |
| Flaig and Junike (flaig2022scenario) | yes (many risk factors) | n/v | n/v | n/v | compared with approved internal models; n/v as metric validation | n/v |
| CTBench (ang2025ctbench) | yes (452 tokens) | partly: "four distinct market regimes", unnamed | n/v | n/v | "risk assessment" dimension; VaR/ES n/v | no (crypto) |
| Caulfield and Gleeson (caulfield2024systematic) | yes | n/v | n/v | n/v | no VaR/ES in abstract (IV trading task) | n/v |
| MarketGANs (huh2026marketgans) | yes | n/v | n/v | n/v | no VaR/ES in abstract (covariance estimation) | no (US equities) |
| MARCD (alzahrani2025marcd) | yes (ETFs) | regime-conditioned MODEL; regime-split evaluation n/v | n/v | 2020-2025 out of sample, not framed as a crisis test | CVaR in the allocator; n/v as validation | n/v |
| Cetingoz and Lehalle (cetingoz2025synthetic) | yes (US equities) | n/v | n/v | n/v | no VaR/ES in abstract (strategy-based test) | no (US equities) |
| Masi et al. 2026 (masi2026gandiffusion) | yes | n/v | n/v | n/v | n/v | n/v |
| SFAG (zhang2026sfag) | no (one index) | n/v | n/v | n/v | no VaR in abstract (momentum backtest) | yes (Shanghai Composite) |
| Hounwanou et al. (hounwanou2025synthetic) | no (S&P 500 index) | n/v | n/v | n/v | no VaR in abstract (portfolio, vol forecasting) | no |
| TSGBench (ang2023tsgbench) | n/v | n/v | n/v | n/v | n/v | n/v |

TSGBench was verified bibliographically only (first pass); its design was not checked in this pass.

Reading of the table: no abstract mentions a real-vs-real noise ceiling, and none describes a crisis-labelled
held-out test of cross-asset dependence. The nearest neighbours are CTBench (regime-wise evaluation plus a risk
dimension, crypto), Tail-GAN (multi-asset tail risk via VaR/ES), the VaR review (backtesting as a KPI) and
Cetingoz and Lehalle (standard metrics are not enough for portfolio use).

## Task 4. Is the novelty sentence defensible?

Sentence under test: "prior work has not evaluated crisis-time cross-asset dependence of synthetic return
generators against a noise ceiling."

Verdict: NEEDS WEAKENING. Nothing found contradicts it, but (a) the search was two queries plus abstract reads,
not a systematic review; (b) full texts were not read, and real-vs-real baselines are the kind of detail that
lives in a methods section, not an abstract; (c) adjacent work exists on each ingredient separately (regime-wise
evaluation in CTBench, VaR/ES-based assessment in Tail-GAN and Ericson et al., tail co-movement in MarketGANs),
so an unqualified "has not" invites a reviewer counter-example.

Proposed wording:

> "To our knowledge, no existing benchmark of synthetic return generators evaluates cross-asset dependence
> separately in crisis and calm regimes, normalises generator error by a real-versus-real noise ceiling, and
> checks the resulting ranking against an out-of-sample VaR/ES backtest. Existing evaluations address these
> elements separately: regime-wise evaluation \citep{ang2025ctbench}, tail-risk scoring through VaR and ES
> \citep{cont2022tailgan,ericson2024var}, and cross-sectional tail co-movement \citep{huh2026marketgans}."

Do not write "first" without "to our knowledge". Before submission, read the evaluation sections of CTBench,
Tail-GAN and MarketGANs in full: those three are the most likely to contain a counter-example.

## Task 5. NSE / SEBI market-structure facts

Primary source actually read: SEBI master circular chapter "CHAPTER 1: TRADING" (PDF, text extracted), section 2
"Circuit Breaker / Price Bands": https://www.sebi.gov.in/sebi_data/commondocs/oct-2023/Chapter-1-Trading_p.pdf
The NSE pages themselves (nseindia.com circuit-breaker and price-band pages) timed out on every direct fetch;
what they say is known only from search-result excerpts and is marked as such.

Index-based market-wide circuit breakers (SEBI 2.1, 2.2, 2.4)
- Three stages, either direction: 10%, 15%, 20%; triggered by BSE Sensex or NSE Nifty 50, whichever is breached
  first. Limits are translated daily from the previous day's closing index level.
- 10%: before 1 PM, 1 hour halt; 1 PM to before 2:30 PM, half-hour halt; at or after 2:30 PM, no halt.
- 15%: before 1 PM, 2 hour halt; 1 PM to before 2 PM, 1 hour halt; at or after 2 PM, halt for the rest of the day.
- 20%: halt for the remainder of the day.
- Trading resumes with a 15-minute pre-open call auction, and "the extant duration of the market halt shall be
  suitably reduced by fifteen minutes" (2.4.3). Effective halts are therefore 45 min / 15 min at 10% and
  1 h 45 min / 45 min at 15%, each followed by the 15-minute auction. This matches the table reported in the
  search excerpt of the NSE page (https://www.nseindia.com/products-services/equity-market-circuit-breakers;
  page not opened).
- Circular references given in the SEBI text: SMDRPD/Policy/Cir-37/2001 (28 June 2001); CIR/MRD/DP/25/2013
  (3 Sept 2013); CIR/MRD/DP/02/2015 (12 Jan 2015).

Price bands for individual securities (SEBI 2.3, 2.5)
- 2.3.1: scrip-wise price bands "up to 20% either way, for all scrips in the rolling settlement except for the
  scrips on which derivatives products are available."
- 2.5.1-2.5.2: scrips excluded from price bands instead have DYNAMIC price bands (operating range) that reject
  orders outside the range. The dynamic band is set at 10% of the previous day's closing price for (i) stocks on
  which derivatives are available, (ii) stocks in indices on which derivatives are available, (iii) index
  futures, (iv) stock futures.
- 2.5.3: in a market trend the dynamic band is relaxed "in increments of 5% only after a cooling-off period of
  15 minutes" from the time the exchange's flexing criteria are met (footnote: SEBI press release, 20 March 2020).
- FINDING: stocks with derivatives, which covers the NIFTY 50 constituents, have NO fixed daily price band. They
  have only a dynamic 10% band that is flexed in 5% steps. Daily returns of these names are therefore not
  hard-truncated at +/-10% or +/-20%; a move can exceed 10% in a day after flexing. The paper should say
  "dynamic price bands (10%, relaxed in 5% steps after a 15-minute cooling-off)" and must not describe a fixed
  circuit limit for F&O stocks. Any index-level truncation comes from the market-wide breaker only.
- Not verified on an opened page: the 2%/5%/10%/20% tiering of fixed bands for non-F&O scrips, and the exchange's
  exact flexing criteria (a search excerpt of an NSE circular mentions last trade at 9.90% or more of base price
  with minimum trade and client-code counts). Do not state these without opening the NSE circular.
- Whether every current NIFTY 50 constituent is in the F&O segment was not checked; "essentially all" is the
  brief's assumption, not a verified fact.

T+1 settlement
- SEBI allowed T+1 on an optional basis by circular dated 7 September 2021; the market infrastructure
  institutions' joint press release of 8 November 2021 set a phased roadmap "with bottom 100 stocks based on
  market capitalization being available for introduction of T+1 settlement cycle from February 25, 2022".
  Source (PDF read): SEBI board memorandum, January 2022,
  https://www.sebi.gov.in/sebi_data/meetingfiles/jan-2022/1642414627550_1.pdf
- "in 2021, T+1 settlement was introduced in a phased manner, and this was fully implemented from January 2023."
  Source (PDF read): https://www.sebi.gov.in/sebi_data/meetingfiles/may-2024/1714651311614_1.pdf ; also "moving
  the equity markets completely to T+1 settlement cycle since January 2023",
  https://www.sebi.gov.in/sebi_data/meetingfiles/nov-2024/1731577464307_1.pdf
- The exact completion day "27 January 2023" appears only in a search-result excerpt; it was NOT found in any
  document that was opened. Write "January 2023" unless the dated circular is located.
- Implication worth stating in the paper if settlement is discussed: phasing went from the smallest stocks
  upward, so large caps migrated late in the schedule. The date on which NIFTY 50 names migrated was not verified.

## Could not verify (or verified only partly)

- Exact date 27 January 2023 for full T+1 implementation (only "January 2023" confirmed).
- NSE's own circuit-breaker and price-band pages (timeouts); facts above rest on the SEBI master circular.
- Fixed-band tiers (2/5/10/20%) for non-F&O scrips; NSE's dynamic-band flexing criteria; full F&O membership of
  NIFTY 50.
- The publication date and full title/number of the SEBI master circular (the .bib year 2023 comes from the URL).
- Peer-reviewed venues for TimeVAE (cited as arXiv preprint) and for Ghosh et al. "From Variational to
  Deterministic Autoencoders" (the brief's "2020" presumably means ICLR 2020; not confirmed, cited as arXiv 2019).
- Volume/pages of Acerbi and Szekely (Risk, December 2014); end pages of Hamilton 1989 and Christoffersen 1998;
  the printed subtitle of Engle 2002; journal details of Tail-GAN (Management Science); DOI of Flaig and Junike.
- Content of Masi et al. 2023 (ICAIF) and of TSGBench beyond bibliographic data.
- "Sig-Graph GAN" as a name: mapped to arXiv 2605.22215 by the first pass, not re-checked.
- No item in the brief turned out to be non-existent. Several are mislabelled (see below).

## Corrections to the project's existing notes and claims

1. "SigCWGAN (Ni et al.)": first author is Liao (Liao, Ni, Sabate-Vidales, Szpruch, Wiese, Xiao); published as
   "Sig-Wasserstein GANs for Conditional Time Series Generation", Mathematical Finance 34(2), 2024.
2. "CoMeTS-GAN (arXiv 2605.27113)": real ID, different title (GAN-Diffusion framework paper).
3. FinStressTS is a forecasting benchmark; S3G is a prediction model. Neither is a generator-evaluation
   comparator and neither should appear in the gap table.
4. arXiv 2401.10370 is on USD yield curves; arXiv 2512.21791 and SFAG are single-index studies. None is a
   multi-asset equity benchmark, so they are weak comparators for a cross-asset claim.
5. F&O stocks: dynamic bands only, no fixed band (confirms the suspicion recorded in docs/STATE.md).
6. Circuit-breaker halt durations: quote the effective 45 min / 15 min / 1 h 45 min / 45 min plus a 15-minute
   call auction, or the nominal 1 h / 30 min / 2 h / 1 h "including" the auction, but not a mixture.
