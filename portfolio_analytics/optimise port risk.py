import numpy as np
import pandas as pd
import cvxpy as cp

# Parameters
P, H, K = 573, 20, 5
N = P + H
lambda_l1 = 10
adv_limit = 0.2

np.random.seed(42)
X = np.random.randn(N, K)
F = np.diag(np.random.uniform(0.01, 0.05, K))
specVar = np.random.uniform(0.01, 0.03, N)
w = np.zeros(N)
w[:P] = np.random.dirichlet(np.ones(P))

adv_notional = np.random.uniform(1e5, 5e6, H)
max_hedge_size = adv_limit * adv_notional

D = np.diag(specVar)
cov_total = X @ F @ X.T + D

h = cp.Variable(N)
constraints = [h[:P] == 0]
for i in range(H):
    idx = P + i
    constraints += [
        h[idx] <= max_hedge_size[i],
        h[idx] >= -max_hedge_size[i]
    ]

portfolio_w = w + h
objective = cp.Minimize(cp.quad_form(portfolio_w, cov_total) + lambda_l1 * cp.norm1(h))
prob = cp.Problem(objective, constraints)
prob.solve()

# Output
hedge_weights = {f"Hedge_{i+1}": round(h.value[P + i], 4) for i in range(H) if abs(h.value[P + i]) > 1e-5}
risk_before = np.sqrt(w.T @ cov_total @ w)
risk_after = np.sqrt(portfolio_w.value.T @ cov_total @ portfolio_w.value)

print("Selected Hedge Weights:", hedge_weights)
print("Risk Before:", round(risk_before, 4))
print("Risk After :", round(risk_after, 4))