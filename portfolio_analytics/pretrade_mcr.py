import numpy as np
import pandas as pd

def risk_breakdown_assets(w, Sigma_total, notional):
    """
    Risk breakdown per asset:
    - w: portfolio weights (1D array)
    - Sigma_total: total covariance matrix (systematic + specific)
    - notional: total portfolio value
    """
    w = np.array(w).reshape(-1, 1)
    sigma_p = np.sqrt(float(w.T @ Sigma_total @ w))
    mcr = (Sigma_total @ w).flatten() / sigma_p
    ctr = (w.flatten() * mcr)
    dollar_risk = ctr * sigma_p * notional

    return pd.DataFrame({
        "Weight": w.flatten(),
        "MCR": mcr,
        "CTR": ctr,
        "Risk_$": dollar_risk
    })


def risk_breakdown_factors(b, Sigma_f, notional):
    """
    Risk breakdown per factor:
    - b: factor exposures (1D array)
    - Sigma_f: factor covariance matrix
    - notional: total portfolio value
    """
    b = np.array(b).reshape(-1, 1)
    sigma_sys = np.sqrt(float(b.T @ Sigma_f @ b))
    mcr = (Sigma_f @ b).flatten() / sigma_sys
    ctr = b.flatten() * mcr
    dollar_risk = ctr * sigma_sys * notional

    return pd.DataFrame({
        "Exposure": b.flatten(),
        "MCR": mcr,
        "CTR": ctr,
        "Risk_$": dollar_risk
    })


# Dummy data
w = np.array([0.25, 0.25, 0.25, 0.25])
B = np.array([[0.5, 0.1],
              [0.4, -0.2],
              [0.6, 0.0],
              [0.3, 0.2]])
Sigma_f = np.array([[0.04, 0.01],
                    [0.01, 0.02]])
Sigma_spec_diag = np.diag([0.02, 0.015, 0.025, 0.02])
notional = 10_000_000

# Compute total covariance matrix
Sigma_total = B @ Sigma_f @ B.T + Sigma_spec_diag

# Run breakdowns
asset_risk_df = risk_breakdown_assets(w, Sigma_total, notional)
factor_exposures = w @ B  # portfolio-level exposures
factor_risk_df = risk_breakdown_factors(factor_exposures, Sigma_f, notional)

# Display results
import ace_tools as tools; tools.display_dataframe_to_user(name="Asset Risk Breakdown ($)", dataframe=asset_risk_df)
