import cvxpy as cp

# Constants
P = 7                    # Portfolio assets
H = 23                   # Hedge assets
N = P + H                # Total
K = 5                    # Number of factors
k = 5                    # Max number of hedge instruments to select
M = 1e4                  # Big-M value

# Extract numpy arrays from dataframes
X = df_exposures.values
specVar = df_specvar["specVar"].values
w = df_portfolio["weight"].values
F = df_factor_cov.values

# Covariance matrix: X F X' + D
D = np.diag(specVar)
cov_total = X @ F @ X.T + D

# Optimisation variables
h = cp.Variable(N)                  # hedge weights
z = cp.Variable(H, boolean=True)   # selection binary

# Constraints
constraints = []

# Hedge instruments are only allowed on last H assets
for i in range(P):
    constraints.append(h[i] == 0)

for i in range(H):
    idx = P + i
    constraints += [
        h[idx] <=  z[i] * M,
        h[idx] >= -z[i] * M
    ]

constraints.append(cp.sum(z) <= k)

# Objective: minimise total portfolio risk after hedge
total_w = w + h
objective = cp.Minimize(cp.quad_form(total_w, cov_total))

# Solve with ECOS_BB
prob = cp.Problem(objective, constraints)
prob.solve(solver="ECOS_BB")

# Output
selected_hedge_assets = [hedge_assets[i] for i in range(H) if round(z.value[i]) == 1]
hedge_weights = {hedge_assets[i]: round(h.value[P + i], 6) for i in range(H) if round(z.value[i]) == 1}
risk_before = np.sqrt(w.T @ cov_total @ w)
risk_after = np.sqrt(total_w.value.T @ cov_total @ total_w.value)

(selected_hedge_assets, hedge_weights, risk_before, risk_after)