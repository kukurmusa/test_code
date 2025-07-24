import numpy as np
import pandas as pd
import cvxpy as cp

# Parameters
P, H, K = 573, 20, 5
N = P + H
k = 5
M = 1e4
adv_limit = 0.2

np.random.seed(42)
X = np.random.randn(N, K)
F = np.diag(np.random.uniform(0.01, 0.05, K))
specVar = np.random.uniform(0.01, 0.03, N)
w = np.zeros(N)
w[:P] = np.random.dirichlet(np.ones(P))
adv_notional = np.random.uniform(1e5, 5e6, size=H)
max_hedge_size = adv_limit * adv_notional
min_hedge_size = 0.05 * adv_notional

D = np.diag(specVar)
cov_total = X @ F @ X.T + D

# Step 1: Greedy pre-selection based on exposure similarity
b_p = w @ X[:P]
scores = np.array([np.abs(np.dot(X[P+i], b_p)) for i in range(H)])
top_indices = scores.argsort()[::-1][:8]
selected_idxs = [P + i for i in top_indices]

# Optimisation
z = cp.Variable(len(selected_idxs), boolean=True)
h = cp.Variable(N)
constraints = [h[i] == 0 for i in range(P)]

for j, idx in enumerate(selected_idxs):
    constraints += [
        h[idx] <= z[j] * max_hedge_size[top_indices[j]],
        h[idx] >= -z[j] * max_hedge_size[top_indices[j]],
        cp.abs(h[idx]) >= z[j] * min_hedge_size[top_indices[j]]
    ]

constraints.append(cp.sum(z) <= k)

total_w = w + h
objective = cp.Minimize(cp.quad_form(total_w, cov_total))
prob = cp.Problem(objective, constraints)
prob.solve(solver="ECOS_BB")

# Output
selected_hedges = [f"Hedge_{top_indices[i]+1}" for i in range(len(top_indices)) if round(z.value[i]) == 1]
weights = {f"Hedge_{top_indices[i]+1}": round(h.value[selected_idxs[i]], 4)
           for i in range(len(top_indices)) if round(z.value[i]) == 1}
risk_before = np.sqrt(w.T @ cov_total @ w)
risk_after = np.sqrt(total_w.value.T @ cov_total @ total_w.value)

print("Selected Hedge Instruments:", selected_hedges)
print("Hedge Weights:", weights)
print("Risk Before:", round(risk_before, 4))
print("Risk After :", round(risk_after, 4))