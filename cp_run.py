# run_cp_fc.py
# Usage: python run_cp_fc.py
# Expects cp_fc_layer.mat with variable name holding the 4-way tensor, e.g., 'W4'

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cp_als import cp_als, cp_frobenius_error, cp_reconstruct

# --- Load the provided 4-way tensorized FC weights (28, 28, 16, 32) ---
# Adjust the key if your .mat file uses a different variable name.
mat = loadmat("/Users/shivanedadi/Downloads/tensorized_weights/cp_fc_layer.mat")
# Try a few likely keys; change to the correct one if needed:
X = np.asarray(mat["A"])  # <- this is the correct key in your file
print("Loaded tensor shape:", X.shape, "dtype:", X.dtype)

# --- Sweep ranks and record Frobenius error ---
Rs = list(range(20, 51))  # 20..50 inclusive
errors = []
fits = []

for R in Rs:
    factors, lambdas, info = cp_als(
        X, rank=R, max_iters=200, tol=1e-6,
        init="svd", l2_reg=1e-8, random_state=0, verbose=False
    )
    err = cp_frobenius_error(X, factors, lambdas)
    errors.append(err)
    fits.append(info["fit"])
    print(f"R={R:3d}  iter={info['iters']:3d}  fit={info['fit']:.6f}  ||X - Xhat||_F={err:.6e}")

# --- Plot Frobenius-norm error vs rank R (as requested) ---
plt.figure()
plt.plot(Rs, errors, marker="o")
plt.semilogy(Rs, errors, 'o-')
plt.xlabel("CP Rank R")
plt.ylabel("log ||X - X_hat||_F")
plt.title("CP-ALS error vs. rank on [28,28,16,32] FC weights")
plt.grid(True)
plt.tight_layout()
plt.savefig("cp_fc_error_vs_rank_log.png", dpi=160)
print("Saved plot to cp_fc_error_vs_rank_log.png")
