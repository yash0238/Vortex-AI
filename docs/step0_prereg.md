# Step 0: can a cascade mechanism measured on ordinary days predict record-breaking crashes?

Frozen 2026-11-18 01:40, before any of the data below was downloaded or looked at. Changes after this
point must be listed at the end with a reason; results are reported whatever they are.

## Data
- D1: Kenneth French daily 49 industry portfolios, value-weighted, July 1926 to the latest release.
  Missing values (-99.99) are treated as missing; an industry enters once it has 250 valid days.
- D2: our NSE 47-stock panel (data/panels/nse_ext, 2010-11 to 2026-10).

## Events
- Each fit uses only data before its fit date. For industry/stock i, the threshold q_i is the 1% quantile of
  its daily returns over that fit sample. Event e_{i,t} = 1 if r_{i,t} < q_i.
- Daily count N_t = sum_i e_{i,t}. Block severity S_b = sum of N_t over non-overlapping 20-trading-day blocks.

## Models (each fitted on the fit sample only)
- M1 cascade (the mechanism): N_t | past ~ NegBin(mean lam_t, dispersion r),
  lam_t = mu + alpha * sum_{l>=1} (1-beta) beta^(l-1) N_{t-l}.  Branching ratio n = alpha.
  Same-day clustering (common shocks) is absorbed by the NegBin dispersion, not by alpha.
- M0a extreme-value: unconditional distribution of S_b from the fit sample, empirical below its 90th
  percentile and a generalised Pareto tail above it.
- M0b no-cascade: N_t iid NegBin (M1 with alpha = 0).
- Predictive distribution of S_b for M1 and M0b: 20,000 simulated blocks started from the actual state at
  the block start (M1 uses the real past counts; M0b needs none). Probabilities floored at 1e-6.

## Out-of-sample protocol
- D1: expanding window, first fit on 1926-07 to 1945-12, refit every calendar year; each fit forecasts the
  blocks of the following year.  D2: fit on 2010-11 to 2019-12, forecast 2020-01 to 2026-10.
- Severity is binned on a fixed grid (0, 1-2, 3-5, 6-10, 11-20, 21-40, 41-80, 81-160, 161-320, >320 for D1;
  same edges scaled by 47/49 for D2). Score = log predictive probability of the observed bin.
- Record-breaking block: severity above every earlier block in the data (including the burn-in).

## Primary pass rule
P1: M1 has a higher mean log score than M0a over all out-of-sample blocks of D1.
P2: on the record-breaking blocks of D1 (and the 2020 record on D2), M1 assigns a higher probability to the
    observed bin than M0a in a majority of them.
Pass = P1 and P2.

## Fingerprints (secondary; at least one must hold)
F1 precursor: the branching ratio n estimated on the 5 years before a record-breaking block is higher than
   the median of all rolling 5-year estimates.  Holds if true for a majority of record breakers.
F2 two aftermaths: after the 10 largest blocks, the decay of N_t over the next 120 days follows a power law
   whose exponent differs between blocks with a rising-n precursor and blocks without one (difference
   larger than both standard errors).
(The network fingerprint, contagion moving from sectors to market-wide, needs the full network model and
 is not part of Step 0.)

Decision: Pass + at least one fingerprint -> build the cascade generator.  Otherwise the idea is dropped
or reformulated, and that outcome is reported.

## Changes after freezing
1. 2026-11-18 02:05, before any out-of-sample block was scored: the NegBin count model is unbounded, and a
   sanity simulation on the 1926-1945 fit produced 20-day severities up to 2474, above the physical maximum
   of 49 x 20 = 980. Simulated daily counts are therefore capped at K, the number of series in the event
   set. This is a consistency fix (a count cannot exceed the number of series); the likelihood is unchanged
   because observed counts never exceed K. Seen before the change: only the 1926-1945 fit
   (n = 0.991, beta = 0.855, r = 0.131) and the M0a bin probabilities of that fit.
