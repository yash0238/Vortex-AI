# Table 1: Statistical Fidelity Gates (NSE-47 Survivor Panel, Full Sample)

| Gate | Metric | Real | Generated | Threshold / Reference | Pass |
|------|--------|------|-----------|----------------------|------|
| Kurtosis | Excess Kurtosis | 6.694 | 4.432 | Bootstrap 5–95% CI: [6.207, 7.262] | ❌ |
| ACF(r²) | Lag-1 ACF of squared returns | 0.0316 | 0.8868 | Tolerance ±0.0200 | ❌ |
| Correlation | Frobenius error vs real pooled corr | — | 19.822 | Ceiling 7.071, Limit 10.606 | ❌ |
| Discriminative | Gradient-boosting accuracy | — | 0.988 ± 0.009 | Real-vs-real floor nan, Target < 0.60 | ❌ |
| Contagion | Crisis correlation boost | 6.47% | -0.10% | Real CI ±15.00% | ❌ |
| CVaR Ratio | Crisis/Normal CVaR (α=5%) | 1.06x | 0.99x | Real CI ±0.37x | ✅ |
| Predictive | R² (synthetic→real) | — | 0.0055 | Target > -0.5 | ✅ |

**Overall: 2 / 7 gates passed**
