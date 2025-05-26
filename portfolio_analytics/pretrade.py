
import numpy as np
import pandas as pd

def barra_portfolio_risk(weights, X, F, specific_var, notional):
    '''
    Computes portfolio risk, MCR, CTR, and VaR using a Barra-style factor risk model.

    Parameters:
        weights (array): Portfolio weights
        X (2D array): Factor exposure matrix (assets x factors)
        F (2D array): Factor covariance matrix (factors x factors)
        specific_var (array): Specific variance per asset
        notional (float): Portfolio value in dollars

    Returns:
        dict: Dictionary of risk outputs in raw, dollar, and bps terms
    '''
    w = np.array(weights).reshape(-1, 1)
    X = np.array(X)
    F = np.array(F)
    D = np.diag(specific_var)

    # Total covariance matrix: Σ = X F X' + D
    cov_total = X @ F @ X.T + D

    # Portfolio variance and volatility
    port_var = float(w.T @ cov_total @ w)
    port_vol = np.sqrt(port_var)

    # Systematic and specific variance
    systematic_cov = X @ F @ X.T
    systematic_var = float(w.T @ systematic_cov @ w)
    specific_var_total = float(w.T @ D @ w)

    # Percent breakdown
    systematic_pct = systematic_var / port_var
    specific_pct = specific_var_total / port_var

    # Factor exposures and contributions
    b = (w.T @ X).flatten()
    Fb = F @ b
    factor_contrib = b * Fb
    factor_contrib_pct = factor_contrib / port_var

    # Marginal Contribution to Risk (MCR)
    sigma_w = cov_total @ w
    mcr = (sigma_w / port_vol).flatten()

    # Contribution to Risk (CTR)
    ctr = weights * mcr

    # Dollar and bps conversions
    factor_contrib_usd = factor_contrib * notional
    factor_contrib_bps = factor_contrib * 10000

    port_vol_usd = port_vol * notional
    port_vol_bps = port_vol * 10000

    systematic_risk_usd = systematic_var * notional
    specific_risk_usd = specific_var_total * notional
    systematic_risk_bps = systematic_var * 10000
    specific_risk_bps = specific_var_total * 10000

    mcr_usd = mcr * notional
    mcr_bps = mcr * 10000
    ctr_usd = ctr * notional
    ctr_bps = ctr * 10000

    # Daily VaR calculations
    daily_vol = port_vol / np.sqrt(252)
    z_95 = 1.645
    z_99 = 2.326
    var_95 = z_95 * daily_vol * notional
    var_99 = z_99 * daily_vol * notional
    var_95_bps = var_95 / (notional / 10000)
    var_99_bps = var_99 / (notional / 10000)

    return {
        'Portfolio Volatility': port_vol,
        'Portfolio Volatility ($)': port_vol_usd,
        'Portfolio Volatility (bps)': port_vol_bps,
        'Systematic Risk': systematic_var,
        'Specific Risk': specific_var_total,
        'Systematic Risk ($)': systematic_risk_usd,
        'Specific Risk ($)': specific_risk_usd,
        'Systematic Risk (bps)': systematic_risk_bps,
        'Specific Risk (bps)': specific_risk_bps,
        'Factor Contribution': factor_contrib,
        'Factor Contribution (%)': factor_contrib_pct,
        'Factor Contribution ($)': factor_contrib_usd,
        'Factor Contribution (bps)': factor_contrib_bps,
        'MCR': mcr,
        'CTR': ctr,
        'MCR ($)': mcr_usd,
        'CTR ($)': ctr_usd,
        'MCR (bps)': mcr_bps,
        'CTR (bps)': ctr_bps,
        'VaR 95% ($)': var_95,
        'VaR 99% ($)': var_99,
        'VaR 95% (bps)': var_95_bps,
        'VaR 99% (bps)': var_99_bps,
    }
