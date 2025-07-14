import plotly.graph_objects as go

# Factor contributions in bps (exclude specific risk if it's separate)
factor_names = ["Value", "Momentum", "Size", "Volatility"]
contributions = [25, -10, 15, 5]

# Optional specific (idiosyncratic) risk
specific = 65

# Combine all and compute total
all_contribs = contributions + [specific]
total_risk = sum(all_contribs)

# Labels and bar types
labels = factor_names + ["Specific", "Portfolio Risk"]
measures = ["relative"] * len(all_contribs) + ["total"]
y_values = all_contribs + [total_risk]

fig = go.Figure(go.Waterfall(
    name="Risk Attribution",
    orientation="v",
    measure=measures,
    x=labels,
    y=y_values,
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    text=[f"{v} bps" for v in y_values],
    textposition="outside"
))

fig.update_layout(
    title="Factor Contributions to Total Portfolio Risk",
    yaxis_title="Risk Contribution (bps)",
    xaxis_title="Source",
    showlegend=False
)

fig.show()
