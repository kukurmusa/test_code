from qpython import qconnection
from qpython.qtype import QException
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# =========================================================
# 1. Simple continuous hedge: solveHedgeMinWeight via Gurobi
#    Minimise (b_p - X_f^T h)' F (b_p - X_f^T h)
#    s.t. h_i >= 0.10, sum h_i = 1
# =========================================================
def solveHedgeMinWeight(hedgeInput):
    """
    Gurobi-only version of the simple continuous hedge problem.

    hedgeInput: q dict with keys `b_p`, `X_f`, `F` mapped to:
      - b_p : portfolio factor exposure vector (K,)
      - X_f : futures factor exposures (M x K)
      - F   : factor covariance matrix (K x K)
    """

    try:
        # Extract input from q dict (keys are bytes from qpython)
        b_p = np.array(hedgeInput[b"b_p"]).flatten()     # (K,)
        X_f = np.array(hedgeInput[b"X_f"])               # (M, K)
        F   = np.array(hedgeInput[b"F"])                 # (K, K)

        K = b_p.shape[0]
        M = X_f.shape[0]

        # Regularise / symmetrise covariance
        F = 0.5 * (F + F.T)
        F = F + np.eye(K) * 1e-8

        # We'll write the objective as:
        #   minimise (b_p - X_f^T h)' F (b_p - X_f^T h)
        # Let X_T = X_f.T
        X_T = X_f.T  # (K, M)

        # Expand objective: || F^(1/2) (b_p - X_T h) ||^2
        # More directly: y = b_p - X_T h
        # objective = y' F y = (b_p - X_T h)' F (b_p - X_T h)
        # In standard QP form: 0.5 h' Q h + c' h + const
        # Derivation:
        #   = h' (X_T' F X_T) h - 2 b_p' F X_T h + b_p' F b_p
        # So:
        #   Q = 2 * X_T' F X_T
        #   c = -2 * X_T' F b_p

        Q = 2.0 * (X_T.T @ F @ X_T)        # (M, M)
        c = -2.0 * (X_T.T @ F @ b_p)       # (M,)

        # Build Gurobi model
        model = gp.Model("solveHedgeMinWeight")
        model.Params.OutputFlag = 0  # set to 1 for debug

        # Variables: h_i, no upper bounds (only via constraints)
        h = model.addVars(M, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="h")

        # Constraints: h_i >= 0.10, sum h_i = 1
        for i in range(M):
            model.addConstr(h[i] >= 0.10, name=f"min_weight_{i}")

        model.addConstr(gp.quicksum(h[i] for i in range(M)) == 1.0, name="sum_to_one")

        # Objective: 0.5 h' Q h + c' h
        obj = gp.QuadExpr()

        # Quadratic part
        for i in range(M):
            for j in range(M):
                if Q[i, j] != 0.0:
                    obj += 0.5 * Q[i, j] * h[i] * h[j]

        # Linear part
        for i in range(M):
            if c[i] != 0.0:
                obj += c[i] * h[i]

        model.setObjective(obj, GRB.MINIMIZE)

        # Solve
        model.optimize()

        if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return f"Error: Gurobi status {model.Status}"

        # Extract solution
        h_opt = [h[i].X for i in range(M)]

        # Return hedge weights to q (as simple list)
        return h_opt

    except Exception as e:
        return f"Error: {e}"


# =========================================================
# 2. MIQP version (with binaries z) in Gurobi only
#    – equivalent to your second cvxpy example
# =========================================================
def solveHedgeMIQP(b_p, X_f, F, max_futs=5, big_M=1.0):
    """
    Gurobi-only version of the MIQP hedge:

    Minimise (b_p - X_f^T h)' F (b_p - X_f^T h)
    s.t. -M z_i <= h_i <= M z_i
         sum z_i <= max_futs
         sum |h_i| >= 0.2   (linearised)
         sum h_i == 0
    """

    b_p = np.asarray(b_p).flatten()         # (K,)
    X_f = np.asarray(X_f)                   # (M, K)
    F   = np.asarray(F)                     # (K, K)

    K = b_p.shape[0]
    M = X_f.shape[0]

    # Regularise / symmetrise covariance
    F = 0.5 * (F + F.T)
    F = F + np.eye(K) * 1e-8

    X_T = X_f.T  # (K, M)

    # Quadratic form in h:
    # same derivation as above
    Q = 2.0 * (X_T.T @ F @ X_T)        # (M, M)
    c = -2.0 * (X_T.T @ F @ b_p)       # (M,)

    model = gp.Model("HedgeMIQP")
    model.Params.OutputFlag = 1

    # Variables
    h = model.addVars(M, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="h")
    z = model.addVars(M, vtype=GRB.BINARY, name="z")
    # auxiliary for |h_i|
    a = model.addVars(M, lb=0.0, name="a")

    # Constraints:
    for i in range(M):
        # -M z_i <= h_i <= M z_i
        model.addConstr(h[i] <=  big_M * z[i], name=f"ub_{i}")
        model.addConstr(h[i] >= -big_M * z[i], name=f"lb_{i}")

        # a_i >= |h_i|
        model.addConstr(a[i] >=  h[i], name=f"a_pos_{i}")
        model.addConstr(a[i] >= -h[i], name=f"a_neg_{i}")

    # sum z_i <= max_futs
    model.addConstr(gp.quicksum(z[i] for i in range(M)) <= max_futs, name="max_futures")

    # sum |h_i| >= 0.2   ->   sum a_i >= 0.2
    model.addConstr(gp.quicksum(a[i] for i in range(M)) >= 0.2, name="min_activity")

    # dollar-neutral: sum h_i == 0
    model.addConstr(gp.quicksum(h[i] for i in range(M)) == 0.0, name="dollar_neutral")

    # Objective: 0.5 h' Q h + c' h
    obj = gp.QuadExpr()
    for i in range(M):
        for j in range(M):
            if Q[i, j] != 0.0:
                obj += 0.5 * Q[i, j] * h[i] * h[j]

    for i in range(M):
        if c[i] != 0.0:
            obj += c[i] * h[i]

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi status: {model.Status}")

    h_opt = np.array([h[i].X for i in range(M)])
    z_opt = np.array([z[i].X for i in range(M)])

    return h_opt, z_opt, model.ObjVal


# =========================================================
# 3. q IPC server – unchanged, now using Gurobi backend
# =========================================================
def run_server():
    with qconnection.QConnection(host="localhost", port=5001) as q:
        print("Connected to q. Awaiting requests...")
        while True:
            try:
                data = q.receive(data_only=False)
                fn, args = data
                if fn == b"solveHedgeMinWeight":
                    result = solveHedgeMinWeight(args)
                    q.send(result)
                else:
                    q.send(f"Unknown function: {fn}")
            except QException as e:
                print(f"QException: {e}")
            except Exception as e:
                print(f"Error: {e}")


# =========================================================
# 4. Standalone test for the MIQP version (optional)
# =========================================================
if __name__ == "__main__":
    # If you want to test the MIQP bit directly in Python:
    b_p = np.array([0.6, 0.8, 0.3])
    X_f = np.random.randn(8, 3) * 0.5 + 0.1
    F = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05],
    ])

    h_opt, z_opt, obj = solveHedgeMIQP(b_p, X_f, F, max_futs=5, big_M=1.0)
    print("MIQP objective:", obj)
    print("Selected weights:", np.round(h_opt, 4))
    print("Used futures:", np.where(np.abs(h_opt) > 1e-4)[0])

    # Or start the q server instead:
    # run_server()
