import cvxpy as cp
import numpy as np

# --- Mock setup: Portfolio + 23 hedge assets
P = 7       # portfolio assets
H = 23      # hedge instruments
N = P + H   # total assets
K = 5       # factors

# Portfolio weights (first P have capital, rest are 0)
w = np.zeros(N)
w[:P] = np.random.dirichlet(np.ones(P))  # normalised example

# Factor exposures (N x K)
X = np.random.randn(N, K)

# Factor covariance
F = np.diag(np.random.uniform(0.01, 0.05, size=K))

# Specific variance (N)
specVar = np.random.uniform(0.01, 0.03, size=N)
D = np.diag(specVar)

# Total covariance matrix
cov_total = X @ F @ X.T + D

# --- Optimisation variables
h = cp.Variable(N)                  # hedge weights
z = cp.Variable(H, boolean=True)   # binary selection: only for hedge instruments

# --- Constraints
M = 1e4  # Big-M value
constraints = []

# Only hedge weights (last H elements) can be non-zero
for i in range(P):
    constraints.append(h[i] == 0)

# Link binary z to hedge weights h[P:] (Big-M)
for i in range(H):
    constraints += [
        h[P+i] <=  z