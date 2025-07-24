import cvxpy as cp
import numpy as np
import pandas as pd

# Simulate data
P, H, K, k = 7, 23, 5, 5
N = P + H
M = 1e4
np.random.seed(42)

# Mock data
X = np.random.randn(N, K)
F = np.diag(np.random.uniform(0.01, 0.05, K))
specVar = np.random.uniform(0.01, 0.03, N)
w = np.zeros(N)
w[:P] = np.random.dirichlet(np.ones(P))

# Covariance
D = np.diag(specVar)
cov_total = X @ F @ X.T + D

# Optimisation
h = cp.Variable(N)
z = cp.Variable(H, boolean=True)

constraints = [h[i] == 0 for i in range(P)]
constraints += [h[P+i] <= z[i]*M for i in range(H)]
constraints += [h[P+i] >= -z[i]*M for i in range(H)]
constraints.append(cp.sum(z) <= k)

total_w = w + h
objective = cp.Minimize(cp.quad_form(total_w, cov_total))
prob = cp.Problem(objective, constraints)
prob.solve(solver="ECOS_BB")

# Output
hedge_assets = [f"Hedge_{i+1}" for i in range(H)]
selected_hedges = [hedge_assets[i] for i in range(H) if round(z.value[i]) == 1]
weights = {hedge_assets[i]: round(h.value[P+i], 6) for i in range(H) if round(z.value[i]) == 1}
print("Selected Hedge Instruments:", selected_hedges)
print("Hedge Weights:", weights)
print("Risk Before:", np.sqrt(w.T @ cov_total @ w))
print("Risk After :", np.sqrt(total_w.value.T @ cov_total @ total_w.value))