"""
Hedge basket using Barra factor exposures.

Assumptions:
- 2 positions (A and B)
- 3 hedge basket names (C, D, E)
- 3 factors (for illustration)

We solve:
  H * hedge_weights = - (w_A * X_A + w_B * X_B)
"""

import numpy as np

# -------------------------------
# INPUTS
# -------------------------------
# Positions and their weights (notionals)
w_A, w_B = 500000, 300000

# Factor exposures (3 factors each)
X_A = np.array([1.1, 0.5, -0.2])
X_B = np.array([0.9, 0.3, 0.1])

# Hedge basket exposures
X_C = np.array([1.2, 0.4, 0.0])
X_D = np.array([0.8, 0.6, -0.1])
X_E = np.array([1.0, 0.5, 0.2])

# -------------------------------
# CALCULATE TARGET EXPOSURE OFFSET
# -------------------------------
# Sum of target exposures (negative because we want to offset)
target_offset = - (w_A * X_A + w_B * X_B)

print("Target offset vector:", target_offset)

# -------------------------------
# SOLVE FOR HEDGE WEIGHTS
# -------------------------------
# Hedge basket exposures as matrix H (3 factors x 3 hedge names)
H = np.column_stack([X_C, X_D, X_E])

# Solve: H * hedge_weights = target_offset
# Use least squares (robust for over/under-determined systems)
hedge_weights, residuals, rank, s = np.linalg.lstsq(H, target_offset, rcond=None)

# -------------------------------
# RESULTS
# -------------------------------
print("Hedge notional weights (for C, D, E):", hedge_weights)
print("Residuals (unhedged exposure left):", residuals)

# Final hedge basket exposures (should be close to -target_offset)
final_exposure = H @ hedge_weights
print("Final exposure neutralised to:", final_exposure + (w_A * X_A + w_B * X_B))




########################################################################################################################################################################






import numpy as np

# Combined exposure vector
target_offset = - np.array([910000, 440000, -70000])

# Hedge basket matrix H (3x8)
H = np.array([
    [1.2, 0.8, 1.0, 0.9, 1.1, 0.85, 1.05, 0.95],
    [0.4, 0.7, 0.3, 0.5, 0.6, 0.4, 0.5, 0.45],
    [0.0, -0.1, 0.2, -0.05, 0.1, 0.0, -0.1, 0.05]
])

# Solve for hedge weights
hedge_weights, residuals, rank, s = np.linalg.lstsq(H, target_offset, rcond=None)

# Output results
hedge_names = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
for name, weight in zip(hedge_names, hedge_weights):
    print(f"Hedge Notional for {name}: {weight:.2f} EUR")

print("\nResidual exposures:", residuals)
