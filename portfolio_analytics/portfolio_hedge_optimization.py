"""
/--------------------------------------
/ Prepare mock data in kdb+
/--------------------------------------

/ Portfolio factor exposures (K=3)
b_p: enlist each 0.6 0.8 0.3

/ Futures factor exposures (M=3 × K=3)
X_fut: flip (1.0 0.3 0.0; 0.2 1.1 0.1; 0.4 0.6 0.8)

/ Factor covariance matrix (K × K)
F: 0.04 0.01 0.00
   0.01 0.03 0.01
   0.00 0.01 0.05

/ Bundle input
hedgeInput: (`b_p`X_f`F)!(b_p; X_fut; F)

/--------------------------------------
/ Call Python via IPC (port 5001)
/--------------------------------------
pyRes: ("localhost";5001) ("solveHedgeMinWeight"; hedgeInput)

/ View hedge weights returned
pyRes

"""






from qpython import qconnection
from qpython.qtype import QException
import numpy as np
import cvxpy as cp

def solveHedgeMinWeight(hedgeInput):
    try:
        # Extract input from q dict
        b_p = np.array(hedgeInput[b'b_p']).flatten()
        X_f = np.array(hedgeInput[b'X_f'])
        F   = np.array(hedgeInput[b'F'])

        M = X_f.shape[0]  # number of futures

        # Define variable
        h = cp.Variable(M)

        # Objective: minimise systematic risk after hedge
        residual = b_p - X_f.T @ h
        objective = cp.Minimize(cp.quad_form(residual, F))

        # Constraints: min weight 10%, weights sum to 1
        constraints = [h >= 0.10, cp.sum(h) == 1]

        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Return hedge weights to q
        return list(h.value)

    except Exception as e:
        return f"Error: {e}"


# Start Python server on port 5001
def run_server():
    with qconnection.QConnection(host='localhost', port=5001) as q:
        print('Connected to q. Awaiting requests...')
        while True:
            try:
                data = q.receive(data_only=False)
                fn, args = data
                if fn == b'solveHedgeMinWeight':
                    result = solveHedgeMinWeight(args)
                    q.send(result)
                else:
                    q.send(f"Unknown function: {fn}")
            except QException as e:
                print(f"QException: {e}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    run_server()


import numpy as np
import cvxpy as cp

# Dummy data (K=3 factors, M=8 futures)
b_p = np.array([0.6, 0.8, 0.3])
X_f = np.random.randn(8, 3) * 0.5 + 0.1  # 8 futures × 3 factors
F = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.03, 0.01],
    [0.00, 0.01, 0.05]
])
F = 0.5 * (F + F.T)  # make symmetric

# Wrap covariance matrix as PSD
F_psd = cp.psd_wrap(F)

# Hedge variable (continuous)
M = X_f.shape[0]
h = cp.Variable(M)

# Binary inclusion variable for each future
z = cp.Variable(M, boolean=True)

# Residual factor exposure
residual = b_p - X_f.T @ h

# Objective: minimise factor risk
objective = cp.Minimize(cp.quad_form(residual, F_psd))

# Constraints
big_M = 1.0  # max absolute weight per future
constraints = [
    h <=  z * big_M,
    h >= -z * big_M,
    cp.sum(z) <= 5,         # use at most 5 futures
    cp.norm(h, 1) >= 0.2,   # optional: minimum activity
    cp.sum(h) == 0          # optional: dollar-neutral
]

# Solve with ECOS_BB (MIQP)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS_BB, verbose=True)

# Output
print("Solver status:", problem.status)
print("Selected weights:", h.value.round(4))
print("Used futures:", np.where(np.abs(h.value) > 1e-4)[0])
