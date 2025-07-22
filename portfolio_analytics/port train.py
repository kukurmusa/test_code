import numpy as np
import cvxpy as cp

def hedge_with_top_k_same_side(b_p, X_f, F, k=5):
    """
    Solves hedge optimisation and returns top-k same-side futures weights.
    
    Inputs:
    - b_p: np.array of shape (K,) - portfolio factor exposure
    - X_f: np.array of shape (M, K) - futures factor exposure matrix
    - F: np.array of shape (K, K) - factor covariance matrix
    - k: max number of futures to use in hedge
    
    Returns:
    - weights_pruned: np.array of shape (M,) - hedge weights
    """

    # Step 1: Ensure F is symmetric and PSD
    F = 0.5 * (F + F.T)
    F_psd = cp.psd_wrap(F)

    M = X_f.shape[0]
    h = cp.Variable(M)

    # Step 2: Define residual exposure and objective
    residual = b_p - X_f.T @ h
    objective = cp.Minimize(cp.quad_form(residual, F_psd))

    # Step 3: Solve unconstrained long+short problem
    problem = cp.Problem(objective, [])
    problem.solve(solver=cp.SCS)

    if problem.status != "optimal":
        raise ValueError(f"Solver failed: {problem.status}")

    # Step 4: Get weights and select top-k
    weights = h.value
    idx = np.argsort(np.abs(weights))[-k:]

    # Step 5: Determine majority sign (+1 or -1)
    signs = np.sign(weights[idx])
    majority_sign = 1.0 if np.sum(signs > 0) >= np.sum(signs < 0) else -1.0

    # Step 6: Filter to only those on majority side
    same_side_idx = [i for i in idx if np.sign(weights[i]) == majority_sign]

    # If we have fewer than k same-side, fill from rest (optional fallback)
    if len(same_side_idx) < k:
        others = [i for i in np.argsort(np.abs(weights))[:-k:-1] if np.sign(weights[i]) == majority_sign]
        same_side_idx = (same_side_idx + others)[:k]

    # Step 7: Zero out everything else
    weights_pruned = np.zeros_like(weights)
    weights_pruned[same_side_idx] = weights[same_side_idx]

    # Step 8: Optional - rescale to sum to 1 or 0 (e.g., normalise)
    if majority_sign == 1.0:
        weights_pruned /= np.sum(weights_pruned)  # long-only normalisation
    else:
        weights_pruned /= np.sum(np.abs(weights_pruned))  # short basket scaled

    return weights_pruned

# ---------------------
# 🔢 Test with Dummy Data
# ---------------------
if __name__ == "__main__":
    # Portfolio factor exposure (K = 3)
    b_p = np.array([0.6, 0.8, 0.3])

    # Futures factor exposures (M = 7 futures, K = 3 factors)
    X_f = np.array([
        [1.0, 0.3, 0.0],
        [0.2, 1.1, 0.1],
        [0.4, 0.6, 0.8],
        [-0.5, -0.2, -0.1],
        [-0.8, -1.0, -0.5],
        [0.0, 0.0, 0.9],
        [-0.4, 0.1, 0.3]
    ])

    # Factor covariance matrix (K × K)
    F = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.03, 0.01],
        [0.00, 0.01, 0.05]
    ])

    # Solve and print result
    weights = hedge_with_top_k_same_side(b_p, X_f, F, k=5)
    print("Final hedge weights (same side, top 5):")
    for i, w in enumerate(weights):
        print(f"Future {i+1}: {w:.4f}")