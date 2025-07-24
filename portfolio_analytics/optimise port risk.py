import cvxpy as cp
import numpy as np
import pandas as pd

np.random.seed(42)

# --------------------------
# Dummy Input Data
# --------------------------

# Assume 5 assets in basket and 6 futures in hedge
n_assets = 5
m_futures = 6
k_factors = 4  # number of Barra factors

# Random basket weights (long-only)
w_b = np.array([0.2, 0.15, 0.25, 0.2, 0.2])  # sum to 1

# Dummy Barra exposures for basket and hedge
B_b = np.random.randn(n_assets, k_factors)   # 5 x 4
B_h = np.random.randn(m_futures, k_factors)  # 6 x 4

# Dummy factor covariance matrix (positive definite)
A = np.random.randn(k_factors, k_factors)
Cov_F = A.T @ A  # 4 x 4 positive semi-definite

# --------------------------
# Optimization
# --------------------------

w_h = cp.Variable(m_futures)  # hedge weights
z = cp.Variable(m_futures, boolean=True)  # binary selection vars
M = 10  # big-M for sparsity

# Net factor exposure (after hedge)
net_exposure = B_b.T @ w_b + B_h.T @ w_h

# Objective: Minimise factor risk
risk = cp.quad_form(net_exposure, Cov_F)

# Constraints
constraints = [
    cp.abs(w_h) <= M * z,  # link weights to binary variables
    cp.sum(z) <= 3,        # select at most 3 hedge instruments
    # cp.sum(w_h) == 0,    # optional: dollar neutral
]

# Solve
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve(solver=cp.ECOS_BB)

# --------------------------
# Output
# --------------------------

# Show hedge weights and selected instruments
df_result = pd.DataFrame({
    'Future': [f'H{i+1}' for i in range(m_futures)],
    'Weight': np.round(w_h.value, 4),
    'Selected': z.value.astype(int)
})

print("Optimized Hedge Portfolio:")
print(df_result[df_result['Selected'] == 1])
