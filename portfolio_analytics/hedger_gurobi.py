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
    Solve the hedge optimisation problem using gurobipy instead of CVXPY.
    The overall structure, inputs and outputs are kept the same.
    """

    # -----------------------------
    # Unpack inputs
    # -----------------------------
    B_b = inputDict["B_b"]               # factor exposures of portfolio (K × N)
    B_h = inputDict["B_h"]               # factor exposures of futures/ETFs (K × M)
    w_b = inputDict["w_b"]               # portfolio weights (N,)
    Cov_F = inputDict["Cov_F"]           # factor covariance matrix (K × K)
    adv = inputDict["adv"]               # ADV in notional for each hedge instrument (M,)
    portfolio_notional = inputDict["portfolio_notional"]
    m_futures = inputDict["m_futures"]   # number of hedge instruments (M)
    M = inputDict["M"]                   # big-M constant for |w_h| <= M z

    # Factor dimension K from covariance (should match B_b/B_h)
    n_factors = Cov_F.shape[0]

    # Base factor exposure from the portfolio only: b_base = B_b @ w_b  (K × 1)
    exposure_base = B_b @ w_b

    # -----------------------------
    # ADV-based max weight per hedge
    # -----------------------------
    max_allowed_weight = np.minimum(
        M,
        inputDict["adv_cap"] * adv / portfolio_notional
    )
    max_leverage = 100.0   # L1-norm cap on hedge weights
    min_weight = 0.30      # kept for structure; not used explicitly

    # -----------------------------
    # Pre-compute quadratic pieces
    # -----------------------------
    # Risk(b_net) = (b_base + B_h w)^T Cov_F (b_base + B_h w)
    #            = w^T Q w + c^T w + const
    #
    # Q = B_h^T Cov_F B_h   (M × M)
    # c = 2 * B_h^T Cov_F b_base  (M,)
    Q = B_h.T @ Cov_F @ B_h
    c_vec = 2.0 * (B_h.T @ (Cov_F @ exposure_base))

    # -----------------------------
    # Build Gurobi model
    # -----------------------------
    model = gp.Model("optimal_hedge")

    # Hedge weights (continuous)
    w_h = model.addVars(m_futures, lb=-GRB.INFINITY, name="w_h")

    # Binary selection flags (instrument used or not)
    z = model.addVars(m_futures, vtype=GRB.BINARY, name="z")

    # Auxiliary abs-value vars u[j] ≈ |w_h[j]| for L1 norm and big-M
    u = model.addVars(m_futures, lb=0.0, name="u")

    # -----------------------------
    # Constraints (mirror original CVXPY logic)
    # -----------------------------

    # 1) u[j] >= |w_h[j]|  (absolute-value linearisation)
    for j in range(m_futures):
        model.addConstr(u[j] >=  w_h[j],   name=f"abs_pos_{j}")
        model.addConstr(u[j] >= -w_h[j],   name=f"abs_neg_{j}")

    # 2) |w_h[j]| <= M * z[j]  (link weight to binary decision)
    for j in range(m_futures):
        model.addConstr(u[j] <= M * z[j], name=f"bigM_{j}")

    # 3) Optional per-instrument ADV cap:
    #    |w_h[j]| <= max_allowed_weight[j] * z[j]
    for j in range(m_futures):
        model.addConstr(
            u[j] <= max_allowed_weight[j] * z[j],
            name=f"adv_cap_{j}"
        )

    # 4) At most 2 hedge instruments can be active (same as cp.sum(z) <= 2)
    model.addConstr(gp.quicksum(z[j] for j in range(m_futures)) <= 2,
                    name="max_2_instruments")

    # 5) L1 leverage cap: sum_j |w_h[j]| <= max_leverage
    model.addConstr(gp.quicksum(u[j] for j in range(m_futures)) <= max_leverage,
                    name="leverage_cap")

    # -----------------------------
    # Objective: minimise factor risk
    #   Minimise w^T Q w + c^T w  (constant term dropped)
    # -----------------------------
    quad_obj = gp.QuadExpr()

    # Quadratic term w^T Q w
    for i in range(m_futures):
        for j in range(m_futures):
            if Q[i, j] != 0.0:
                quad_obj.add(Q[i, j] * w_h[i] * w_h[j])

    # Linear term c^T w
    for i in range(m_futures):
        if c_vec[i] != 0.0:
            quad_obj.add(c_vec[i] * w_h[i])

    model.setObjective(quad_obj, GRB.MINIMIZE)

    # Solver parameters (can tune further)
    model.Params.OutputFlag = 0   # suppress verbose output
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = 60   # seconds safety cap

    # -----------------------------
    # Solve
    # -----------------------------
    model.optimize()

    print("Feasible solution found?", model.Status == GRB.OPTIMAL)

    # -----------------------------
    # Risk before hedge
    # -----------------------------
    exposure_before = exposure_base
    risk_before = float(exposure_before.T @ Cov_F @ exposure_before)
    risk_before = math.sqrt(risk_before / 250.0)
    print(f"Risk Before: {risk_before}")

    # Handle infeasible / no-solution cases like before
    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        return (inputDict["adv_cap"], risk_before, None)

    # -----------------------------
    # Extract hedge solution
    # -----------------------------
    w_h_sol = np.array([w_h[j].X for j in range(m_futures)])

    # -----------------------------
    # Risk after hedge
    # -----------------------------
    exposure_after = exposure_before + B_h @ w_h_sol
    risk_after = float(exposure_after.T @ Cov_F @ exposure_after)
    risk_after = math.sqrt(risk_after / 250.0)
    print(f"Risk After: {risk_after}")

    return (inputDict["adv_cap"], risk_before, risk_after, w_h_sol)



# Run optimisation
calculate_optimal_hedge(inputDict)
