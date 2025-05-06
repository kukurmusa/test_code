import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up asset names
assets = ['AAPL', 'MSFT', 'JPM', 'XOM']

# Simulated factor exposures (4 assets x 3 factors: Market, Size, Value)
X = np.array([
    [1.2, 0.3, 0.1],   # AAPL
    [1.1, 0.2, 0.2],   # MSFT
    [0.8, -0.1, 0.5],  # JPM
    [0.9, 0.0, 0.7]    # XOM
])

# Factor covariance matrix (3x3)
F = np.array([
    [0.05, 0.01, 0.00], 
    [0.01, 0.03, 0.01], 
    [0.00, 0.01, 0.04]
])

# Specific risk (diagonal of specific variance matrix)
specific_var = np.array([0.02, 0.015, 0.025, 0.03])
D = np.diag(specific_var)

# Portfolio weights
weights = np.array([0.3, 0.3, 0.2, 0.2])

# Calculate total covariance matrix using Barra model: Σ = XFX' + D
cov_total = X @ F @ X.T + D

# Portfolio variance and volatility
port_var = weights.T @ cov_total @ weights
port_vol = np.sqrt(port_var)

# Marginal contribution to risk (MCR): (Σw)_i / σ_p
mcr = (cov_total @ weights) / port_vol

# Total contribution to risk by asset: w_i * MCR_i
ctr = weights * mcr

# Put into a DataFrame
df_results = pd.DataFrame({
    'Weight': weights,
    'MCR': mcr,
    'Contribution_to_Risk': ctr
}, index=assets)

df_results.loc['Total'] = [np.sum(weights), '', np.sum(ctr)]

print(df_results)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Barra Risk Decomposition", dataframe=df_results)

print(port_vol)




# Compute systematic covariance matrix: XFX'
systematic_cov = X @ F @ X.T

# Compute systematic and specific risk contributions to portfolio variance
systematic_var = weights.T @ systematic_cov @ weights
specific_var_total = weights.T @ D @ weights

# Total risk check
total_var_check = systematic_var + specific_var_total

# Percentage contribution
systematic_pct = systematic_var / port_var
specific_pct = specific_var_total / port_var

# Factor exposures for the portfolio
portfolio_factor_exposure = weights @ X  # 1x3 vector

# Risk contribution from each factor
factor_risk_contributions = portfolio_factor_exposure @ F @ portfolio_factor_exposure.T
factor_contributions_pct = (portfolio_factor_exposure @ F @ portfolio_factor_exposure.T) / port_var

# Breakdown of factor contributions
factor_names = ['Market', 'Size', 'Value']
factor_contrib_values = (portfolio_factor_exposure @ F) * portfolio_factor_exposure
factor_contrib_pct = factor_contrib_values / port_var

# Assemble DataFrame
df_risk_breakdown = pd.DataFrame({
    'Risk Contribution': list(factor_contrib_values) + [specific_var_total],
    'Percent of Total Risk': list(factor_contrib_pct) + [specific_pct]
}, index=factor_names + ['Specific'])

# tools.display_dataframe_to_user(name="Systematic and Specific Risk Breakdown", dataframe=df_risk_breakdown)
print(df_risk_breakdown)

