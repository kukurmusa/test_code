import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------
# 1) Create sample inputs: basket + ETF returns
# ---------------------------------------------------

np.random.seed(42)

T = 250        # number of days
M = 4          # number of hedge ETFs

dates = pd.bdate_range("2024-01-01", periods=T)

# Common market factor
common_factor = np.random.normal(0, 0.01, size=T)

# Basket returns: mostly driven by common factor + small idiosyncratic noise
basket_ret = 0.8 * common_factor + np.random.normal(0, 0.005, size=T)
basket_returns = pd.Series(basket_ret, index=dates, name="Basket")

# ETF returns: different loadings to the same factor + idiosyncratic noise
etf_data = {}
loadings = [0.9, 0.7, 0.4, -0.2]   # ETF_1 most similar, ETF_4 negatively correlated
for i in range(M):
    etf_ret = loadings[i] * common_factor + np.random.normal(0, 0.008, size=T)
    etf_data[f"ETF_{i+1}"] = etf_ret

etf_returns = pd.DataFrame(etf_data, index=dates)

# Liquidity / ADV info (sample numbers in notional terms)
portfolio_notional = 100_000_000.0       # e.g. £100m basket
adv_notional = np.array([25e6, 30e6, 15e6, 10e6])  # ADV for each ETF in £
adv_cap = 0.3                             # max 30% of ADV allowed for hedge

# ---------------------------------------------------
# 2) Prepare matrices for the QP: min ||r_b - R w||^2
# ---------------------------------------------------

# Align data just in case (here it is already aligned)
df = pd.concat([basket_returns, etf_returns], axis=1, join="inner")
y = df.iloc[:, 0].values           # basket returns (T,)
R = df.iloc[:, 1:].values          # ETF returns (T, M)

T_eff, m_futures = R.shape         # m_futures should be M

# Quadratic objective components:
#   min_w ||y - R w||^2
# = w^T (R^T R) w - 2 (R^T y)^T w + y^T y
Q = R.T @ R                        # (M × M)
c_vec = -2.0 * (R.T @ y)           # (M,)

# Constant term y^T y is irrelevant for optimisation (just shifts objective)

# ---------------------------------------------------
# 3) Gurobi model with Barra-style constraints
# ---------------------------------------------------

model = gp.Model("return_based_hedge")

# Hedge weights for each ETF (can be long or short)
w = model.addVars(m_futures, lb=-GRB.INFINITY, name="w")

# Absolute value |w_j| captured via auxiliary variables
u = model.addVars(m_futures, lb=0.0, name="u")

# Binary selection variables: z_j = 1 if ETF j is used in the hedge
z = model.addVars(m_futures, vtype=GRB.BINARY, name="z")

# Big-M for |w_j| <= M_big * z_j
M_big = 5.0        # max absolute weight per ETF (e.g. 5× notional)

# Max L1 leverage: sum |w_j|
max_leverage = 5.0  # e.g. total gross hedge <= 5× notional

# Per-instrument ADV-based cap on |w_j|
# max_allowed_weight_j = min(M_big, adv_cap * ADV_j / portfolio_notional)
max_allowed_weight = np.minimum(M_big, adv_cap * adv_notional / portfolio_notional)

# ---------------------------------------------------
# Constraints
# ---------------------------------------------------

# 1) u_j >= |w_j|  (absolute-value linearisation)
for j in range(m_futures):
    model.addConstr(u[j] >=  w[j],   name=f"abs_pos_{j}")
    model.addConstr(u[j] >= -w[j],   name=f"abs_neg_{j}")

# 2) Big-M link: |w_j| <= M_big * z_j
for j in range(m_futures):
    model.addConstr(u[j] <= M_big * z[j], name=f"bigM_{j}")

# 3) ADV-based liquidity cap: |w_j| <= max_allowed_weight_j * z_j
for j in range(m_futures):
    model.addConstr(
        u[j] <= max_allowed_weight[j] * z[j],
        name=f"adv_cap_{j}"
    )

# 4) Cardinality constraint: use at most 2 ETFs in the hedge
model.addConstr(gp.quicksum(z[j] for j in range(m_futures)) <= 2,
                name="max_2_hedge_ETFs")

# 5) L1 leverage constraint: sum_j |w_j| <= max_leverage
model.addConstr(gp.quicksum(u[j] for j in range(m_futures)) <= max_leverage,
                name="leverage_cap")

# (Optional) You could add a net exposure constraint too:
# model.addConstr(gp.quicksum(w[j] for j in range(m_futures)) == 0, name="net_flat")

# ---------------------------------------------------
# 4) Objective: minimise tracking error ||y - R w||^2
# ---------------------------------------------------

quad_obj = gp.QuadExpr()

# Quadratic term: w^T Q w
for i in range(m_futures):
    for j in range(m_futures):
        if Q[i, j] != 0.0:
            quad_obj.add(Q[i, j] * w[i] * w[j])

# Linear term: c^T w
for i in range(m_futures):
    if c_vec[i] != 0.0:
        quad_obj.add(c_vec[i] * w[i])

model.setObjective(quad_obj, GRB.MINIMIZE)

# Gurobi parameters
model.Params.OutputFlag = 1   # set to 0 for silent run
model.Params.MIPGap = 1e-4
model.Params.TimeLimit = 60   # seconds

# ---------------------------------------------------
# 5) Solve
# ---------------------------------------------------

model.optimize()

print("Optimal?", model.Status == GRB.OPTIMAL)

# ---------------------------------------------------
# 6) Extract solution and analytics
# ---------------------------------------------------

if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
    w_sol = np.array([w[j].X for j in range(m_futures)])
    z_sol = np.array([z[j].X for j in range(m_futures)])
    u_sol = np.array([u[j].X for j in range(m_futures)])

    # Hedge returns from solution
    hedge_ret = R @ w_sol  # (T,)

    # Residual / tracking error
    residual = y - hedge_ret
    residual_vol = residual.std()

    # Correlation between basket and hedge
    corr = np.corrcoef(y, hedge_ret)[0, 1]

    # Beta of basket vs hedge portfolio: Cov(b, h) / Var(h)
    var_h = np.var(hedge_ret, ddof=1)
    if var_h > 0:
        cov_bh = np.cov(y, hedge_ret, ddof=1)[0, 1]
        beta = cov_bh / var_h
    else:
        beta = np.nan

    print("\n=== ETF Names ===")
    print(list(etf_returns.columns))

    print("\n=== Selected ETFs (z_j = 1) ===")
    for j, name in enumerate(etf_returns.columns):
        if z_sol[j] > 0.5:
            print(f"{name}: w = {w_sol[j]:.4f}, |w| = {u_sol[j]:.4f}")

    print("\nFull hedge weights (all ETFs):")
    for j, name in enumerate(etf_returns.columns):
        print(f"{name:6s}: w = {w_sol[j]: .4f}, z = {z_sol[j]:.0f}")

    print(f"\nResidual (hedged) volatility: {residual_vol:.6f}")
    print(f"Correlation(basket, hedge):   {corr:.4f}")
    print(f"Beta(basket vs hedge):        {beta:.4f}")

else:
    print("No feasible solution found.")
