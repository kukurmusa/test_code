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


#################


# import plotly.graph_objects as go
#
# # Factor contributions in bps (exclude specific risk if it's separate)
# factor_names = ["Value", "Momentum", "Size", "Volatility"]
# contributions = [25, -10, 15, 5]
#
# # Optional specific (idiosyncratic) risk
# specific = 65
#
# # Combine all and compute total
# all_contribs = contributions + [specific]
# total_risk = sum(all_contribs)
#
# # Labels and bar types
# labels = factor_names + ["Specific", "Portfolio Risk"]
# measures = ["relative"] * len(all_contribs) + ["total"]
# y_values = all_contribs + [total_risk]
#
# fig = go.Figure(go.Waterfall(
#     name="Risk Attribution",
#     orientation="v",
#     measure=measures,
#     x=labels,
#     y=y_values,
#     connector={"line": {"color": "rgb(63, 63, 63)"}},
#     text=[f"{v} bps" for v in y_values],
#     textposition="outside"
# ))
#
# fig.update_layout(
#     title="Factor Contributions to Total Portfolio Risk",
#     yaxis_title="Risk Contribution (bps)",
#     xaxis_title="Source",
#     showlegend=False
# )
#
# fig.show()


import plotly.graph_objects as go

factors = ["Value", "Momentum", "Size", "Volatility", "Specific"]
contributions = [25, -10, 15, 5, 65]  # Total = 100 bps risk

# SHAP-style bar chart
base_risk = 0  # like SHAP base value
total_risk = sum(contributions)

colors = ["red" if v > 0 else "green" for v in contributions]

fig = go.Figure()

# Add base
fig.add_trace(go.Bar(
    x=[base_risk],
    y=["Base"],
    orientation='h',
    marker=dict(color="lightgrey"),
    showlegend=False,
    hoverinfo='skip'
))

# Add factor contributions
fig.add_trace(go.Bar(
    x=contributions,
    y=factors,
    orientation='h',
    marker_color=colors,
    text=[f"{v:+} bps" for v in contributions],
    textposition="outside",
    name="Factor Contributions"
))

# Add total bar
fig.add_trace(go.Bar(
    x=[total_risk],
    y=["Total"],
    orientation='h',
    marker=dict(color="blue"),
    text=[f"{total_risk} bps"],
    textposition="outside",
    showlegend=False
))

fig.update_layout(
    title="SHAP-style Portfolio Risk Attribution (BPS)",
    xaxis_title="Risk Contribution (bps)",
    yaxis=dict(showgrid=False),
    barmode='stack',
    height=400
)

fig.show()
