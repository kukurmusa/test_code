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

# Dummy data (M futures, K factors)
X_f = np.array([
    [1.0, 0.3, 0.0],
    [0.2, 1.1, 0.1],
    [0.4, 0.6, 0.8]
])
b_p = np.array([0.6, 0.8, 0.3])
F = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.03, 0.01],
    [0.00, 0.01, 0.05]
])

M = X_f.shape[0]
h = cp.Variable(M)

residual = b_p - X_f.T @ h
objective = cp.Minimize(cp.quad_form(residual, F))
constraints = [h >= 0.10, cp.sum(h) == 1.0]

problem = cp.Problem(objective, constraints)
problem.solve()

# Validate solution
if h.value is None:
    raise ValueError("Optimisation failed or returned no solution.")

h_val = np.array(h.value).flatten()
print("h_val:", h_val)
print("residual:", b_p - X_f.T @ h_val)

