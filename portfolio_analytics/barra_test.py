import numpy as np
import pandas as pd

def barra_risk_breakdown_dollar(weights, X, F, spec_var, notional, factor_meta):
    """
    Calculate daily portfolio risk using Barra factor model
    Args:
        weights: np.array (n_assets,)
        X: np.array (n_assets, n_factors) exposure matrix
        F: np.array (n_factors, n_factors) factor covariance matrix (annualised)
        spec_var: np.array (n_assets,) specific variance per asset (annualised)
        notional: float (portfolio size)
        factor_meta: pd.DataFrame with columns ["factor", "category"]
    Returns:
        Dictionary with portfolio summary, asset-level, factor-level, and grouped risk
    """
    # Convert to daily
    scale = 1 / np.sqrt(252)
    F_daily = F * scale**2
    spec_var_daily = spec_var * scale**2

    weights = np.array(weights).reshape(-1, 1)  # column vector
    X = np.array(X)
    D = np.diag(spec_var_daily)

    # Covariance matrices
    sys_cov = X @ F_daily @ X.T
    total_cov = sys_cov + D

    # Portfolio variance & volatility
    port_var = float(weights.T @ total_cov @ weights)
    port_vol = np.sqrt(port_var)
    port_dollar_risk = port_vol * notional

    # Asset-level risk
    mcr = (total_cov @ weights) / port_vol
    ctr = (weights * mcr).flatten()
    ctr_dollar = ctr * notional

    # Specific and systematic risk
    sys_var = float(weights.T @ sys_cov @ weights)
    spec_var_total = float(weights.T @ D @ weights)
    sys_dollar_risk = sys_var * notional
    spec_dollar_risk = spec_var_total * notional

    # Factor CTR
    factor_exposure = X.T  # (n_factors, n_assets)
    factor_mcr = (factor_exposure @ F_daily @ factor_exposure @ weights) / port_vol
    factor_ctr = (factor_mcr.flatten() * np.sum(factor_exposure * weights.T, axis=1))
    factor_ctr_dollar = factor_ctr * notional

    # Factor table with metadata
    factor_table = pd.DataFrame({
        "factor": factor_meta["factor"],
        "category": factor_meta["category"],
        "CTR": factor_ctr,
        "DollarCTR": factor_ctr_dollar
    })

    # Grouped by category
    factor_by_category = factor_table.groupby("category", as_index=False).agg({"DollarCTR": "sum"})

    # Asset-level breakdown
    asset_table = pd.DataFrame({
        "Asset": np.arange(len(weights)),
        "Weight": weights.flatten(),
        "MCR": mcr.flatten(),
        "CTR": ctr,
        "DollarCTR": ctr_dollar
    })

    # Final Output
    return {
        "portfolio": {
            "Volatility": port_vol,
            "Variance": port_var,
            "Systematic": sys_var,
            "Specific": spec_var_total,
            "DollarRisk": port_dollar_risk,
            "Systematic$": sys_dollar_risk,
            "Specific$": spec_dollar_risk
        },
        "assetBreakdown": asset_table,
        "factorBreakdown": factor_table,
        "factorByCategory": factor_by_category
    }





weights = [0.2, 0.3, 0.5]
X = [
    [1, 0.5, 0, 0, 0],
    [0.4, 1, 0.6, 0, 0],
    [0.6, 0.3, 0, 1, 0.2]
]
F = np.zeros((5, 5))
F[0, 0] = 0.04
F[1, 1] = 0.05
F[2, 2] = 0.03
F[3, 3] = 0.02
F[4, 4] = 0.015
spec_var = [1.5, 2.0, 1.0]
notional = 100_000_000

factor_meta = pd.DataFrame({
    "factor": ["Value", "Momentum", "Tech", "UK", "France"],
    "category": ["Style", "Style", "Industry", "Country", "Country"]
})

result = barra_risk_breakdown_dollar(weights, X, F, spec_var, notional, factor_meta)
