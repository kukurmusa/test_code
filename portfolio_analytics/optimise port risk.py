import cvxpy as cp
import numpy as np
import pandas as pd

np.random.seed(1)

# --------------------------
# Dummy Input Data (Controlled)
# --------------------------

n_assets = 5
m_futures = 6
k_factors = 3

# Basket weights
w_b = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

# Basket exposures (5 assets × 3 factors)
B_b = np.array([
    [1.0, 0.5, -0.2],
    [0.8, 0.3, -0.1],
    [1.2, 0.7, -0.3],
    [0.6, 0.2, 0.0],
    [1.1, 0.4, -0.1]
])

# Futures exposures (6 contracts × 3 factors)
B_h = np.array([
    [-1.0, -0.5, 0.2],
    [-0.8, -0.3, 0.1],
    [-1.2, -0.6, 0.3],
    [0.0, 0.1, -0.5],
    [0.5, -0.1, -0.3],
    [-0.7, -0.2, 0.2]
])

# Factor covariance (positive definite 3x3)
Cov_F = np.array([
    [0.04, 0.01, 0.0],
    [0.01, 0.03, 0.0],
    [0.0,  0.0,  0.02]
])

# --------------------------
# Optimization
# --------------------------

w_h = cp.Variable(m_futures)
z = cp.Variable(m_futures, boolean=True)
M = 5

net_exposure = B_b.T @ w_b + B_h.T @ w_h
risk_expr = cp.quad_form(net_exposure, Cov_F)

min_weight = 0.05  # 5%
M = 5.0            # Max hedge notional weight

constraints = [
    cp.abs(w_h[i]) <= M * z[i] for i in range(m_futures)
] + [
    cp.abs(w_h[i]) >= min_weight * z[i] for i in range(m_futures)
] + [
    cp.sum(z) <= 3
]

problem = cp.Problem(cp.Minimize(risk_expr), constraints)
problem.solve(solver=cp.ECOS_BB)

# --------------------------
# Output
# --------------------------

# Risk before hedge
exposure_before = B_b.T @ w_b
risk_before = exposure_before.T @ Cov_F @ exposure_before

# Risk after hedge
exposure_after = exposure_before + B_h.T @ w_h.value
risk_after = exposure_after.T @ Cov_F @ exposure_after

# Results
df_result = pd.DataFrame({
    'Future': [f'H{i+1}' for i in range(m_futures)],
    'Weight': np.round(w_h.value, 4),
    'Selected': z.value.astype(int)
})

print("\nOptimized Hedge Portfolio (max 3 futures):")
print(df_result[df_result['Selected'] == 1])

print(f"\nFactor Risk Before Hedge:  {risk_before:.6f}")
print(f"Factor Risk After Hedge:   {risk_after:.6f}")
