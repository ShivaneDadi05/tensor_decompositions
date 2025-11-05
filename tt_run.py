# tt_run.py
# Sweep tolerances for TT-SVD on your 7-way tensor (key "A") and plot S vs error.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tt import tt_svd, tt_error_and_compression, tt_params

# 1) Load tensor (key "A")
mat = loadmat("/Users/shivanedadi/Downloads/tensorized_weights/tt_embedding.mat")
if "A" not in mat:
    raise KeyError(f"Expected key 'A' in tt_embedding.mat; found: {list(mat.keys())}")
X = np.asarray(mat["A"]).astype(np.float64)
print("Loaded X shape:", X.shape, "dtype:", X.dtype)
orig_params = int(np.prod(X.shape))
print("Original parameter count:", orig_params)

# 2) Define tolerance sweep
eps_list = [1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3]

errs, comps, ranks_list, params_list = [], [], [], []
for eps in eps_list:
    cores, ranks = tt_svd(X, eps=eps)  # add max_rank=... to cap ranks if desired

    # DEBUG: print core shapes so any mismatch is obvious up front
    print(f"\n[eps={eps}] core shapes:")
    for i, G in enumerate(cores):
        print(f"  core {i}: {G.shape}")

    err, S, tt_p = tt_error_and_compression(X, cores)
    errs.append(err)
    comps.append(S)
    ranks_list.append(ranks)
    params_list.append(tt_p)
    print(f"[eps={eps}] err={err:.3e}  S={S:.2f}  TT params={tt_p}  ranks={ranks}")

# 3) Plot S vs actual error
plt.figure()
plt.semilogx(errs, comps, marker="o")
plt.xlabel(r"Actual relative error  $\varepsilon = \|X - \hat{X}\|_F / \|X\|_F$")
plt.ylabel(r"Compression ratio  $S$")
plt.title("TT-SVD compression vs error")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("tt_S_vs_eps.png", dpi=160)
print("Saved plot to tt_S_vs_eps.png")
