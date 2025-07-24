import cvxpy as cp
import numpy as np

# Assume you already have:
# B_b, w_b, B_h, Cov_F

# Inputs
k = Cov_F.shape[0]
m = B_h.shape[0]

# Decision variable: weights of hedge futures
w_h = cp.Variable(m)

# Total factor exposure after hedge
net_exposure = B_b.T @ w_b + B_h.T @ w_h

# Objective: Minimise factor risk (quadratic form)
risk = cp.quad_form(net_exposure, Cov_F)

# Constraint: Only 3 non-zero hedge weights
# Use L0 approximation: sum of binary indicators
z = cp.Variable(m, boolean=True)
M = 10  # big-M constant
constraints = [
    cp.abs(w_h) <= M * z,  # if z[i]=0, then w_h[i]=0
    cp.sum(z) <= 3
]

# Optional: dollar-neutral
# constraints += [cp.sum(w_h) == 0]

# Solve
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve(solver=cp.ECOS_BB)

print("Optimal hedge weights:", w_h.value)
