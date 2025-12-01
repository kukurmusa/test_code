import numpy as np
import pandas as prd
# import cvxpy as cp        # <-- no longer used
import gurobipy as gp       # <-- new: use gurobipy instead
from gurobipy import GRB
import math

# -----------------------------------------------------------
# Data prep (unchanged from your structure – just as example)
# -----------------------------------------------------------
portfolio_notional = query_kdb("nsjdlpatca002", 55555, "exec sum Notional from data")

B_b = np.stack(np.array(query_kdb("nsjdlpatca002", 55555, "barraRefDict[`barraExp]")))
B_h = np.stack(np.array(query_kdb("nsjdlpatca002", 55555, "etfBasketRefDict[`barraExp]")))
specVar = np.array(query_kdb("nsjdlpatca002", 55555, "barraRefDict[`barraSpec]"))
w_b = np.stack(np.array(query_kdb("nsjdlpatca002", 55555, "barraRefDict[`barraWts]")), dtype=np.float64)
Cov_F = np.array(query_kdb("nsjdlpatca002", 55555, "barraRefDict[`barraCov]"))
adv   = np.array(query_kdb("nsjdlpatca002", 55555, "etfBasket[`advDol]"))

# Make covariance symmetric + add small ridge (unchanged)
Cov_F = (Cov_F + Cov_F.T) / 2
eps = 1e-2
Cov_F += np.eye(Cov_F.shape[0]) * eps

m_futures = B_h.shape[1]     # number of hedge instruments
M = 1.0                      # big-M used in |w_h| <= M * z

# NOTE: cvxpy variables removed – Gurobi will create its own vars
# w_h = cp.Variable(m_futures)
# z   = cp.Variable(m_futures, boolean=True)

# Pre-compute baseline exposure from portfolio only
# (this is used both before & after hedge inside the function)
exposure_base = B_b.T @ w_b

# Keep the same inputDict structure as before so callers don’t break
inputDict = dict()
inputDict["B_b"] = B_b
inputDict["B_h"] = B_h
inputDict["specVar"] = specVar
inputDict["w_b"] = w_b
inputDict["Cov_F"] = Cov_F
inputDict["adv"] = adv
inputDict["portfolio_notional"] = portfolio_notional
inputDict["m_futures"] = m_futures
inputDict["M"] = M
inputDict["adv_cap"] = 1.0   # 100% of ADV as global cap (same as before)


# -----------------------------------------------------------
# Gurobi implementation of the optimisation
# -----------------------------------------------------------
def calculate_optimal_hedge(inputDict):
    """
    Solve the same hedge optimisation problem using gurobipy
    instead of CVXPY/ECOS_BB. Structure of the function and 
    return values is kept the same.
    """

    # -----------------------------
    # Unpack data / scalar limits
    # -----------------------------
    B_b = inputDict["B_b"]
    B_h = inputDict["B_h"]
    w_b = inputDict["w_b"]
    Cov_F = inputDict["Cov_F"]
    adv = inputDict["adv"]
    portfolio_notional = inputDict["portfolio_notional"]
    m_futures = inputDict["m_futures"]
    M = inputDict["M"]

    # Per-instrument max weight from ADV cap (same logic as before)
    max_allowed_weight = np.minimum(
        M,
        inputDict["adv_cap"] * adv / portfolio_notional
    )
    max_leverage = 100.0   # L1-norm cap on hedge weights
    min_weight = 0.30      # currently unused but kept for structure

    n_factors = Cov_F.shape[0]

    # -------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------
    model = gp.Model("optimal_hedge")

    # Continuous hedge weights w_h[j] for each future
    w_h = model.addVars(m_futures, lb=-GRB.INFINITY, name="w_h")

    # Binary selection variables z[j] to control which futures are used
    z = model.addVars(m_futures, vtype=GRB.BINARY, name="z")

    # Auxiliary variables u[j] ≈ |w_h[j]|  (for L1-norm & big-M constraints)
    u = model.addVars(m_futures, lb=0.0, name="u")

    # Factor exposure variables e[k] for net exposure after hedge
    e = model.addVars(n_factors, lb=-GRB.INFINITY, name="e")

    # -------------------------------------------------------
    # Constraints (mirror CVXPY ones)
    # -------------------------------------------------------

    # 1) Tie e[k] to base exposure + futures contribution
    #    e = B_b^T w_b + B_h^T w_h
    for k in range(n_factors):
        model.addConstr(
            e[k] == exposure_base[k] + gp.quicksum(B_h[k, j] * w_h[j] for j in range(m_futures)),
            name=f"exposure_{k}"
        )

    # 2) u[j] >= |w_h[j]|  (absolute value linearisation)
    for j in range(m_futures):
        model.addConstr(u[j] >=  w_h[j],   name=f"abs_pos_{j}")
        model.addConstr(u[j] >= -w_h[j],   name=f"abs_neg_{j}")

    # 3) |w_h| <= M * z  (big-M linking hedge size to binary indicator)
    for j in range(m_futures):
        model.addConstr(u[j] <= M * z[j], name=f"bigM_{j}")

    # 4) Optional tighter instrument-specific ADV caps
    #    (same structure as your commented CVXPY code)
    for j in range(m_futures):
        model.addConstr(u[j] <= max_allowed_weight[j] * z[j],
                        name=f"adv_cap_{j}")

    # 5) Sum of selected futures <= 2  (cardinality constraint)
    model.addConstr(gp.quicksum(z[j] for j in range(m_futures)) <= 2,
                    name="max_2_instruments")

    # 6) L1-norm of hedge weights <= max_leverage
    model.addConstr(gp.quicksum(u[j] for j in range(m_futures)) <= max_leverage,
                    name="leverage_cap")

    # -------------------------------------------------------
    # Objective: minimise net factor risk
    #   risk = e^T Cov_F e
    # -------------------------------------------------------
    quad_expr = gp.QuadExpr()
    for i in range(n_factors):
        for j in range(n_factors):
            if Cov_F[i, j] != 0.0:
                quad_expr.add(e[i] * Cov_F[i, j] * e[j])

    model.setObjective(quad_expr, GRB.MINIMIZE)

    # Speed-up options (tune as needed)
    model.Params.OutputFlag = 0       # silent optimisation
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = 60       # seconds, just as a safety cap

    # -------------------------------------------------------
    # Solve
    # -------------------------------------------------------
    model.optimize()

    print("Feasible solution found?", model.Status == GRB.OPTIMAL)

    # -------------------------------------------------------
    # Risk before hedge  (unchanged calculation)
    # -------------------------------------------------------
    exposure_before = exposure_base
    risk_before = exposure_before.T @ Cov_F @ exposure_before
    risk_before = math.sqrt(risk_before / 250.0)
    print(f"Risk Before: {risk_before}")

    # Handle infeasible / non-optimal cases similar to CVXPY check
    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        return (inputDict["adv_cap"], risk_before, None)

    # -------------------------------------------------------
    # Extract hedge weights from solution
    # -------------------------------------------------------
    w_h_sol = np.array([w_h[j].X for j in range(m_futures)])

    # -------------------------------------------------------
    # Risk after hedge  (same formula as before)
    # -------------------------------------------------------
    exposure_after = exposure_before + B_h.T @ w_h_sol
    risk_after = exposure_after.T @ Cov_F @ exposure_after
    risk_after = math.sqrt(risk_after / 250.0)
    print(f"Risk After: {risk_after}")

    return (inputDict["adv_cap"], risk_before, risk_after, w_h_sol)


# Run optimisation
calculate_optimal_hedge(inputDict)
