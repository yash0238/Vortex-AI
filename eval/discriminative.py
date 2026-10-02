"""Discriminative score test for VORTEX-AI evaluation.

Implements discriminative test from masterplan Part F.2:
Trains a classifier to distinguish real from synthetic data.
Target: accuracy < 0.6 (ideally ~0.5, i.e., indistinguishable).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict


def discriminative_score(
    real_returns: np.ndarray,
    gen_returns: np.ndarray,
    n_samples: int = 500,
    cv_folds: int = 3,
    random_state: int = 42,
    purge: int = 60,
) -> Dict[str, float]:
    """
    Compute discriminative score using Gradient Boosting Classifier.

    Uses block-contiguous folds with purge gaps: overlapping 60-day windows
    mean window-level shuffling leaks near-duplicates across folds, flattering
    the score. Each class is split into contiguous blocks in the passed-in
    order (callers pass chronological windows); train drops windows within
    `purge` positions of the test block. Lower is better:
    - 0.5: Perfect (indistinguishable)
    - <0.6: Good (target threshold)
    - >0.7: Poor (easily distinguishable)

    Args:
        real_returns: (N_real, T, n_stocks) real returns, chronological order
        gen_returns: (N_gen, T, n_stocks) generated returns
        n_samples: number of samples to use from each class
        cv_folds: number of contiguous block folds
        random_state: random seed
        purge: windows dropped from train around each test block

    Returns:
        results: dict with discriminative scores
    """
    # Limit to n_samples from each class, preserving order (no shuffle)
    n = min(n_samples, len(real_returns), len(gen_returns))
    X_real = real_returns[:n].reshape(n, -1)  # (n, T*n_stocks)
    X_gen = gen_returns[:n].reshape(n, -1)

    rng = np.random.default_rng(random_state)
    accs = []
    edges = np.linspace(0, n, cv_folds + 1).astype(int)
    for k in range(cv_folds):
        a, b = int(edges[k]), int(edges[k + 1])
        X_te = np.concatenate([X_real[a:b], X_gen[a:b]], axis=0)
        y_te = np.array([1] * (b - a) + [0] * (b - a))
        keep = np.ones(n, dtype=bool)
        keep[max(0, a - purge):min(n, b + purge)] = False
        X_tr = np.concatenate([X_real[keep], X_gen[keep]], axis=0)
        y_tr = np.array([1] * int(keep.sum()) + [0] * int(keep.sum()))
        if len(X_te) == 0 or int(keep.sum()) < 10:
            continue
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
        clf.fit(X_tr, y_tr)
        accs.append(float(clf.score(X_te, y_te)))

    mean_score = float(np.mean(accs)) if accs else float("nan")
    std_score = float(np.std(accs)) if accs else float("nan")

    results = {
        "discriminative_score": mean_score,
        "discriminative_std": std_score,
        "discriminative_pass": bool(mean_score < 0.60),  # Target: <0.6
        "indistinguishability": 1.0 - abs(mean_score - 0.5) * 2,  # 1.0 = perfect
    }

    return results


def predictive_score(
    real_returns: np.ndarray,
    gen_returns: np.ndarray,
    n_samples: int = 500,
    forecast_horizon: int = 1,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute predictive score: train predictor on synthetic, test on real.
    
    Measures if synthetic data preserves temporal predictive patterns.
    
    Args:
        real_returns: (N, T, n_stocks) real returns
        gen_returns: (N, T, n_stocks) generated returns
        n_samples: samples to use
        forecast_horizon: steps ahead to predict
        cv_folds: CV folds
        random_state: seed
    
    Returns:
        results: dict with predictive scores
    """
    n = min(n_samples, len(real_returns), len(gen_returns))
    
    # Create lagged features and targets
    def create_forecast_data(returns: np.ndarray, h: int = 1):
        """Create (X, y) for next-step prediction."""
        N, T, n_stocks = returns.shape
        X_list, y_list = [], []
        
        for i in range(N):
            for t in range(T - h):
                X_list.append(returns[i, t].flatten())  # Current step
                y_list.append(returns[i, t + h].mean())  # Next step (mean return)
        
        return np.array(X_list), np.array(y_list)
    
    # Train on synthetic
    X_train, y_train = create_forecast_data(gen_returns[:n], h=forecast_horizon)
    
    # Test on real
    X_test, y_test = create_forecast_data(real_returns[:n], h=forecast_horizon)
    
    # Train model
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    
    # Evaluate on real data
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "predictive_mse": float(mse),
        "predictive_r2": float(r2),
        "predictive_pass": bool(r2 > -0.5),  # Should capture some signal
    }
