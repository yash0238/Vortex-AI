# Table 1: Statistical Fidelity Gates (NSE-47 Survivor Panel, Full Sample)

| Gate | Metric | Real | Generated | Threshold / Reference | Pass |
|------|--------|------|-----------|----------------------|------|
| Kurtosis | Excess Kurtosis | 6.694 | 4.504 | Bootstrap 5–95% CI: [6.207, 7.262] | ❌ |
| ACF(r²) | Lag-1 ACF of squared returns | 0.0316 | 0.8836 | Tolerance ±0.0200 | ❌ |
| Correlation | Frobenius error vs real pooled corr | — | 19.672 | Ceiling 7.071, Limit 10.606 | ❌ |
| Discriminative | Gradient-boosting accuracy | — | 0.990 ± 0.007 | Real-vs-real floor nan, Target < 0.60 | ❌ |
| Contagion | Crisis correlation boost | 6.47% | 0.57% | Real CI ±15.00% | ✅ |
| CVaR Ratio | Crisis/Normal CVaR (α=5%) | 1.06x | 0.94x | Real CI ±0.37x | ✅ |
| Predictive | R² (synthetic→real) | — | -0.0106 | Target > -0.5 | ✅ |

**Overall: 3 / 7 gates passed**
