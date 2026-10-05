"""
vortex-ai / sandbox / mf_case_study.py
======================================
Mutual Fund Case Study Evaluation Script.
Evaluates 5 Mutual Fund Portfolios against CG-NSDE Synthetic Crash Scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


class MutualFundCaseStudy:
    def __init__(self, tickers: List[str]):
        """
        Initialize MF Case Study framework.
        :param tickers: List of NIFTY stock tickers matching CG-NSDE model outputs.
        """
        self.tickers = tickers
        self.n_stocks = len(tickers)
        self.portfolios = self._build_portfolio_weights()

    def _build_portfolio_weights(self) -> Dict[str, Dict]:
        """
        Defines portfolio holdings and asset allocation for the 5 schemes.
        Weights are normalized per portfolio.
        """
        # Baseline top stock ticker index mapping
        ticker_map = {t: i for i, t in enumerate(self.tickers)}

        def make_weight_vector(weights_dict: Dict[str, float]) -> np.ndarray:
            vec = np.zeros(self.n_stocks)
            total_w = sum(weights_dict.values())
            for tk, w in weights_dict.items():
                if tk in ticker_map:
                    vec[ticker_map[tk]] = w / total_w
                else:
                    # Distribute unknown ticker weight across existing universe
                    vec += (w / total_w) / self.n_stocks
            return vec

        portfolios = {
            "Axis Multicap Fund": {
                "category": "Multicap Fund",
                "equity_weights": make_weight_vector({
                    "ICICIBANK.NS": 0.08, "HDFCBANK.NS": 0.07, "RELIANCE.NS": 0.06,
                    "AXISBANK.NS": 0.05, "BHARTIARTL.NS": 0.05, "TCS.NS": 0.04,
                    "INFY.NS": 0.04, "LT.NS": 0.04, "BAJFINANCE.NS": 0.03, "TATAMOTORS.NS": 0.03
                }),
                "debt_ratio": 0.01,
                "cash_ratio": 0.02
            },
            "Groww Multicap Fund": {
                "category": "Multicap Fund",
                "equity_weights": make_weight_vector({
                    "RELIANCE.NS": 0.06, "HDFCBANK.NS": 0.06, "ICICIBANK.NS": 0.05,
                    "M&M.NS": 0.05, "BHARTIARTL.NS": 0.04, "TITAN.NS": 0.04,
                    "TATASTEEL.NS": 0.03, "ULTRACEMCO.NS": 0.03, "SUNPHARMA.NS": 0.03
                }),
                "debt_ratio": 0.02,
                "cash_ratio": 0.03
            },
            "Sundaram Multicap Fund": {
                "category": "Multicap Fund",
                "equity_weights": make_weight_vector({
                    "ICICIBANK.NS": 0.07, "HDFCBANK.NS": 0.07, "INFY.NS": 0.05,
                    "RELIANCE.NS": 0.05, "LT.NS": 0.04, "ITC.NS": 0.04,
                    "AXISBANK.NS": 0.04, "HCLTECH.NS": 0.03, "KOTAKBANK.NS": 0.03
                }),
                "debt_ratio": 0.01,
                "cash_ratio": 0.02
            },
            "TrustMF Midcap Fund": {
                "category": "Midcap Fund",
                "equity_weights": make_weight_vector({
                    "BAJFINANCE.NS": 0.06, "TATAMOTORS.NS": 0.05, "PERSISTENT.NS": 0.05,
                    "COFORGE.NS": 0.04, "FEDERALBNK.NS": 0.04, "ASTRAL.NS": 0.04,
                    "POLYCAB.NS": 0.04, "AUROPHARMA.NS": 0.03, "ASHOKLEY.NS": 0.03
                }),
                "debt_ratio": 0.00,
                "cash_ratio": 0.04
            },
            "Bandhan Aggressive Hybrid Fund": {
                "category": "Aggressive Hybrid Fund",
                "equity_weights": make_weight_vector({
                    "ICICIBANK.NS": 0.06, "HDFCBANK.NS": 0.05, "RELIANCE.NS": 0.04,
                    "SBIN.NS": 0.03, "NTPC.NS": 0.03, "M&M.NS": 0.02, "LT.NS": 0.02
                }),
                "debt_ratio": 0.20,  # 20% Debt allocation
                "cash_ratio": 0.05
            }
        }
        return portfolios

    def simulate_fund_returns(
        self,
        gen_returns: np.ndarray,
        rf_daily: float = 0.0002
    ) -> Dict[str, np.ndarray]:
        """
        Maps stock log returns to Fund NAV daily returns.
        :param gen_returns: Generated returns shape (N_scenarios, T_days, N_stocks)
        :param rf_daily: Daily risk-free return rate for debt/cash portion (e.g. 6.5% p.a. ~ 0.02% daily)
        :return: Dict of fund return arrays shape (N_scenarios, T_days)
        """
        N_scenarios, T_days, _ = gen_returns.shape
        fund_returns = {}

        for fund_name, meta in self.portfolios.items():
            w_eq = meta["equity_weights"]
            w_debt = meta["debt_ratio"]
            w_cash = meta["cash_ratio"]
            w_eq_total = 1.0 - w_debt - w_cash

            # Compute daily equity return r_eq = sum(w_i * r_i)
            # gen_returns: (B, T, N) dot w_eq: (N,) -> (B, T)
            eq_returns = np.dot(gen_returns, w_eq) * w_eq_total

            # Debt returns modeled with yield accrual + small noise
            debt_returns = np.full((N_scenarios, T_days), rf_daily) * w_debt
            cash_returns = np.full((N_scenarios, T_days), rf_daily * 0.8) * w_cash

            # Total Fund daily return
            r_fund = eq_returns + debt_returns + cash_returns
            fund_returns[fund_name] = r_fund

        return fund_returns

    def calculate_risk_metrics(
        self,
        fund_returns: Dict[str, np.ndarray],
        regimes: np.ndarray
    ) -> pd.DataFrame:
        """
        Computes CVaR, Max Drawdown, Volatility, and Crisis Beta across regimes.
        """
        records = []
        normal_mask = (regimes == 0)
        crisis_mask = (regimes == 1)

        for fund_name, r_matrix in fund_returns.items():
            # Separate normal vs crisis scenario returns
            r_norm = r_matrix[normal_mask] if np.any(normal_mask) else r_matrix
            r_cris = r_matrix[crisis_mask] if np.any(crisis_mask) else r_matrix

            # NAV trajectories: Cumulative product of (1 + r)
            nav_norm = np.cumprod(1 + r_norm, axis=1)
            nav_cris = np.cumprod(1 + r_cris, axis=1)

            # 1. Max Drawdown (Crisis)
            peak_cris = np.maximum.accumulate(nav_cris, axis=1)
            drawdowns = (peak_cris - nav_cris) / peak_cris
            max_dd_cris = np.max(drawdowns) * 100

            # 2. Max Drawdown (Normal)
            peak_norm = np.maximum.accumulate(nav_norm, axis=1)
            max_dd_norm = np.max((peak_norm - nav_norm) / peak_norm) * 100

            # 3. 95% CVaR (Expected Shortfall) on daily crisis returns
            flat_cris_r = r_cris.flatten()
            var_95 = np.percentile(flat_cris_r, 5)
            cvar_95 = np.mean(flat_cris_r[flat_cris_r <= var_95]) * 100

            # 4. Volatilities (Annualized %)
            vol_norm = np.mean(np.std(r_norm, axis=1)) * np.sqrt(252) * 100
            vol_cris = np.mean(np.std(r_cris, axis=1)) * np.sqrt(252) * 100

            # 5. Downside Ratio (Crisis Vol / Normal Vol)
            vol_ratio = vol_cris / (vol_norm + 1e-8)

            records.append({
                "Fund Name": fund_name,
                "Category": self.portfolios[fund_name]["category"],
                "Normal Vol (%)": round(vol_norm, 2),
                "Crisis Vol (%)": round(vol_cris, 2),
                "Normal Max DD (%)": round(max_dd_norm, 2),
                "Crisis Max DD (%)": round(max_dd_cris, 2),
                "95% Crisis CVaR (%)": round(cvar_95, 2),
                "Vol Surge Ratio": round(vol_ratio, 2)
            })

        return pd.DataFrame(records)

    def plot_nav_trajectories(
        self,
        fund_returns: Dict[str, np.ndarray],
        regimes: np.ndarray,
        save_path: str = "mf_case_study_nav.png"
    ):
        """
        Generates fan charts of NAV paths during crisis regime.
        """
        crisis_mask = (regimes == 1)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()

        for idx, (fund_name, r_matrix) in enumerate(fund_returns.items()):
            ax = axes[idx]
            r_cris = r_matrix[crisis_mask] if np.any(crisis_mask) else r_matrix
            nav_paths = np.cumprod(1 + r_cris, axis=1) * 100.0  # Base NAV = 100

            # Quantiles for fan chart
            p10 = np.percentile(nav_paths, 10, axis=0)
            p50 = np.percentile(nav_paths, 50, axis=0)
            p90 = np.percentile(nav_paths, 90, axis=0)
            t = np.arange(nav_paths.shape[1])

            ax.plot(t, p50, label="Median NAV", color="navy", lw=2)
            ax.fill_between(t, p10, p90, color="skyblue", alpha=0.4, label="10th-90th Percentile")
            ax.set_title(fund_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Simulated NAV (Base 100)")
            ax.legend(loc="lower left")

        # Hide 6th empty subplot
        fig.delaxes(axes[5])
        plt.suptitle("CG-NSDE Stress Test: Mutual Fund NAV Trajectories in Crisis Regime", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to {save_path}")


# =====================================================================
# Execution Sandbox
# =====================================================================
if __name__ == "__main__":
    # Sample NIFTY universe matching download.py
    nifty_universe = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "AXISBANK.NS",
        "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS",
        "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "PERSISTENT.NS", "COFORGE.NS",
        "FEDERALBNK.NS", "ASTRAL.NS", "POLYCAB.NS", "AUROPHARMA.NS", "ASHOKLEY.NS"
    ]

    # Instantiate case study class
    case_study = MutualFundCaseStudy(tickers=nifty_universe)

    # 1. Mock CG-NSDE output for testing (replace with actual model generated tensor)
    # Shape: (100 scenarios, 60 days, 30 stocks)
    np.random.seed(42)
    N_scenarios, T_days, N_stocks = 100, 60, len(nifty_universe)

    # Generate 85 normal scenarios and 15 crisis scenarios
    regime_labels = np.array([0] * 85 + [1] * 15)

    gen_returns = np.random.normal(loc=0.0005, scale=0.012, size=(N_scenarios, T_days, N_stocks))
    # Inject crisis regime behavior (volatility jump + negative drift + high covariance)
    gen_returns[regime_labels == 1] = np.random.normal(loc=-0.004, scale=0.035, size=(15, T_days, N_stocks))

    # 2. Run simulation
    fund_returns = case_study.simulate_fund_returns(gen_returns)

    # 3. Calculate stress metrics
    metrics_df = case_study.calculate_risk_metrics(fund_returns, regime_labels)

    print("\n=======================================================")
    print("      MUTUAL FUND STRESS TEST CASE STUDY RESULTS       ")
    print("=======================================================\n")
    print(metrics_df.to_string(index=False))

    # 4. Generate NAV fan chart plot
    case_study.plot_nav_trajectories(fund_returns, regime_labels)
