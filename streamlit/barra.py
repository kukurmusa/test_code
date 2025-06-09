import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Simulate random data
np.random.seed(42)
num_assets = 4
num_factors = 3

weights_client = np.random.dirichlet(np.ones(num_assets))
weights_bench = np.random.dirichlet(np.ones(num_assets))
X = np.random.normal(0, 1, size=(num_assets, num_factors))
F = np.cov(np.random.normal(0, 1, size=(num_factors, 100)))
specific_var = np.abs(np.random.normal(0.02, 0.005, size=num_assets))
notional = 10_000_000

def compute_risk(weights, X, F, specific_var):
    w = np.array(weights).reshape(-1, 1)
    D = np.diag(specific_var)
    cov_total = X @ F @ X.T + D
    port_var = float(w.T @ cov_total @ w)
    port_vol = np.sqrt(port_var)
    systematic_cov = X @ F @ X.T
    sys_var = float(w.T @ systematic_cov @ w)
    spec_var = float(w.T @ D @ w)
    b = (w.T @ X).flatten()
    Fb = F @ b
    factor_contrib = b * Fb
    factor_contrib_pct = factor_contrib / port_var
    return port_vol, sys_var, spec_var, b, factor_contrib, factor_contrib_pct

# Calculate risks
port_vol_client, sys_var_client, spec_var_client, b_client, factor_contrib_client, factor_contrib_pct_client = compute_risk(weights_client, X, F, specific_var)
port_vol_bench, sys_var_bench, spec_var_bench, b_bench, factor_contrib_bench, factor_contrib_pct_bench = compute_risk(weights_bench, X, F, specific_var)
weights_active = weights_client - weights_bench
port_vol_active, sys_var_active, spec_var_active, b_active, factor_contrib_active, factor_contrib_pct_active = compute_risk(weights_active, X, F, specific_var)

# Display data
st.title("Barra Portfolio Risk (with Plotly Charts)")

st.header("Portfolio Weights")
weights_df = pd.DataFrame({
    "Client": weights_client,
    "Benchmark": weights_bench,
    "Active": weights_active
}, index=[f"Asset {i+1}" for i in range(num_assets)])
st.dataframe(weights_df.style.format("{:.3f}"))

st.header("Risk Metrics")
risk_df = pd.DataFrame({
    "Portfolio": ["Client", "Benchmark", "Active"],
    "Volatility": [port_vol_client, port_vol_bench, port_vol_active],
    "Systematic": [sys_var_client, sys_var_bench, sys_var_active],
    "Specific": [spec_var_client, spec_var_bench, spec_var_active]
})
st.dataframe(risk_df)

# Systematic vs Specific Pie Chart (Client)
pie_fig = px.pie(
    names=["Systematic", "Specific"],
    values=[sys_var_client, spec_var_client],
    title="Client: Systematic vs Specific Risk"
)
st.plotly_chart(pie_fig)

# Factor Contribution Bar Chart (Client)
factor_fig = px.bar(
    x=[f"Factor {i+1}" for i in range(num_factors)],
    y=factor_contrib_pct_client * 100,
    labels={"x": "Factor", "y": "% of Total Risk"},
    title="Client: Factor Contribution %"
)
st.plotly_chart(factor_fig)

# Active Exposures Bar Chart
active_fig = px.bar(
    x=[f"Factor {i+1}" for i in range(num_factors)],
    y=b_active,
    labels={"x": "Factor", "y": "Active Exposure"},
    title="Active Factor Exposures"
)
st.plotly_chart(active_fig)

# VaR
daily_vol = port_vol_client / np.sqrt(252)
z_95, z_99 = 1.645, 2.326
var_95 = z_95 * daily_vol * notional
var_99 = z_99 * daily_vol * notional

st.header("Value-at-Risk (Client)")
st.write(f"**1-Day 95% VaR:** ${var_95:,.0f}")
st.write(f"**1-Day 99% VaR:** ${var_99:,.0f}")


# Calculate Marginal Contribution to Risk (MCR) and Contribution to Risk (CTR)
cov_total = X @ F @ X.T + np.diag(specific_var)
sigma_w = cov_total @ weights_client.reshape(-1, 1)
mcr = (sigma_w / port_vol_client).flatten()
ctr = weights_client * mcr

# Show in a DataFrame
mcr_df = pd.DataFrame({
    "Asset": [f"Asset {i+1}" for i in range(num_assets)],
    "Weight": weights_client,
    "MCR": mcr,
    "CTR": ctr
})
st.header("Marginal & Total Contribution to Risk")
st.write("### 📊 What is MCR and CTR?")
st.markdown("""
**Marginal Contribution to Risk (MCR)** measures how much an infinitesimal increase in an asset's weight changes the total portfolio risk:
\\[
\\text{MCR}_i = \\frac{(\\Sigma w)_i}{\\sigma_p}
\\]
where \\( \\Sigma \\) is the total covariance matrix and \\( \\sigma_p \\) is the portfolio volatility.

**Total Contribution to Risk (CTR)** tells us how much each asset contributes to total portfolio risk:
\\[
\\text{CTR}_i = w_i \\times \\text{MCR}_i
\\]
""")
st.dataframe(mcr_df)

# Plot CTR with Plotly
ctr_fig = px.bar(
    x=mcr_df["Asset"],
    y=mcr_df["CTR"],
    labels={"x": "Asset", "y": "Total Contribution to Risk"},
    title="Asset Contribution to Portfolio Risk (CTR)"
)
st.plotly_chart(ctr_fig)
