import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from weasyprint import HTML
import base64
from io import BytesIO
from pathlib import Path

# Example inputs
assets = ['AAPL', 'MSFT', 'JPM', 'XOM']
weights = np.array([0.3, 0.3, 0.2, 0.2])
X = np.array([
    [1.2, 0.3, 0.1],
    [1.1, 0.2, 0.2],
    [0.8, -0.1, 0.5],
    [0.9, 0.0, 0.7]
])
F = np.array([
    [0.05, 0.01, 0.00],
    [0.01, 0.03, 0.01],
    [0.00, 0.01, 0.04]
])
specific_var = np.array([0.02, 0.015, 0.025, 0.03])
notional = 10_000_000

# Compute covariance matrices
w = weights.reshape(-1, 1)
D = np.diag(specific_var)
cov_total = X @ F @ X.T + D
systematic_cov = X @ F @ X.T

# Portfolio risk
port_var = float(w.T @ cov_total @ w)
port_vol = np.sqrt(port_var)
daily_vol = port_vol / np.sqrt(252)
z_95, z_99 = 1.645, 2.326
var_95 = z_95 * daily_vol * notional
var_99 = z_99 * daily_vol * notional

# Breakdown
sys_var = float(w.T @ systematic_cov @ w)
spec_var = float(w.T @ D @ w)
sys_pct, spec_pct = sys_var / port_var, spec_var / port_var

# Factor contribution
b = (w.T @ X).flatten()
Fb = F @ b
factor_contrib = b * Fb
factor_contrib_pct = factor_contrib / port_var
factor_names = ['Market', 'Size', 'Value']

# Asset MCR & CTR
sigma_w = cov_total @ w
mcr = (sigma_w / port_vol).flatten()
ctr = weights * mcr

# Create charts
def make_pie_chart(labels, sizes, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title(title)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

pie_risk = make_pie_chart(['Systematic', 'Specific'], [sys_pct, spec_pct], 'Systematic vs Specific Risk')
pie_factor = make_pie_chart(factor_names, factor_contrib_pct, 'Factor Risk Contribution')
pie_ctr = make_pie_chart(assets, ctr / port_vol, 'Asset Contribution to Risk')

# Generate HTML report
html_report = f"""
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; padding: 40px; color: #222; }}
h1 {{ color: #003366; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background-color: #f4f4f4; }}
img {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1>Pre-Trade Portfolio Risk Summary</h1>

<h2>Overview</h2>
<p>This report provides a risk-based snapshot of the portfolio using a Barra-style factor risk model. The analysis includes total volatility, systematic/specific risk decomposition, factor attribution, and VaR metrics.</p>

<h2>Portfolio Risk Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Annualised Volatility</td><td>{port_vol:.2%}</td></tr>
<tr><td>Daily Volatility</td><td>{daily_vol:.4%}</td></tr>
<tr><td>Systematic Risk %</td><td>{sys_pct:.2%}</td></tr>
<tr><td>Specific Risk %</td><td>{spec_pct:.2%}</td></tr>
<tr><td>VaR 95% (1-day)</td><td>${var_95:,.0f}</td></tr>
<tr><td>VaR 99% (1-day)</td><td>${var_99:,.0f}</td></tr>
</table>

<h2>Charts</h2>
<img src="data:image/png;base64,{pie_risk}" alt="Systematic vs Specific Risk"/>
<img src="data:image/png;base64,{pie_factor}" alt="Factor Risk Contribution"/>
<img src="data:image/png;base64,{pie_ctr}" alt="Asset Contribution to Risk"/>

<h2>Marginal & Total Risk Contribution</h2>
<table>
<tr><th>Asset</th><th>Weight</th><th>MCR</th><th>CTR</th></tr>
""" + "\n".join(
    f"<tr><td>{a}</td><td>{w:.1%}</td><td>{m:.2%}</td><td>{c:.2%}</td></tr>"
    for a, w, m, c in zip(assets, weights, mcr, ctr)
) + """
</table>

<h2>Factor Contribution Detail</h2>
<table>
<tr><th>Factor</th><th>Contribution</th><th>% of Total Risk</th></tr>
""" + "\n".join(
    f"<tr><td>{f}</td><td>{v:.4f}</td><td>{p:.2%}</td></tr>"
    for f, v, p in zip(factor_names, factor_contrib, factor_contrib_pct)
) + """
</table>

</body>
</html>
"""

# Save as PDF
pdf_path = Path("/mnt/data/pretrade_risk_summary.pdf")
HTML(string=html_report).write_pdf(pdf_path)

pdf_path.name
