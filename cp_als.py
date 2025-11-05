from __future__ import annotations
import numpy as np
from numpy.linalg import solve, norm
from scipy.linalg import svd

__all__ = [
    "cp_als",
    "cp_reconstruct",
    "khatri_rao",
    "unfold",
    "cp_frobenius_error",
]

def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-n unfolding: X_(mode) has shape (I_mode, prod_{k!=mode} I_k)
    """
    X = np.asarray(X)
    order = X.ndim
    # Move desired mode to front, then flatten the rest
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))

def khatri_rao(factors: list[np.ndarray], skip: int | None = None, reverse: bool = False) -> np.ndarray:
    """
    Column-wise Kronecker (Khatri-Rao) product of factor matrices (same #columns R).
    Optionally skip one factor (for MTTKRP) and/or reverse order.
    Returns array with shape (prod of rows, R).
    """
    idx = list(range(len(factors)))
    if skip is not None:
        idx.pop(skip)
    if reverse:
        idx = idx[::-1]

    out = None
    for i, k in enumerate(idx):
        A = factors[k]
        if out is None:
            out = A
        else:
            # column-wise kron: (m x R) ⨀ (n x R) -> (mn x R)
            out = (out[:, None, :] * A[None, :, :]).reshape(out.shape[0]*A.shape[0], out.shape[1])
    return out

def _mttkrp(X: np.ndarray, factors: list[np.ndarray], mode: int) -> np.ndarray:
    """
    MTTKRP: Matricized-Tensor Times Khatri-Rao Product for mode-n.
    M = X_(n) @ KhatriRao(factors_except_n)
    Shapes:
      X_(n): (I_n, prod_{k!=n} I_k)
      KR   : (prod_{k!=n} I_k, R)
      M    : (I_n, R)
    """
    Xn = unfold(X, mode)                        # (I_n, ∏_{k≠n} I_k)
    KR = khatri_rao(factors, skip=mode)         # (∏_{k≠n} I_k, R)
    return Xn @ KR                               # (I_n, R)

def _initialize_factors_svd(X: np.ndarray, R: int, random_state: int | None = None) -> list[np.ndarray]:
    """
    SVD-based initialization: for each mode n, take top-R right singular vectors of X_(n).
    We return factors with shape (I_n, R).
    """
    rng = np.random.default_rng(random_state)
    N = X.ndim
    factors = []
    for n in range(N):
        Xn = unfold(X, n)
        # Compute top-R right-singular vectors via SVD on Xn^T (economy)
        # Using full SVD for robustness; you can switch to randomized SVD if needed.
        U, S, VT = svd(Xn, full_matrices=False)
        if VT.shape[0] >= R:
            # columns are right singular vectors of Xn: V (R x J). We want factors (I_n x R)
            # Better: use U (I_n x R) because Xn = U S V^T -> U spans rows of Xn
            An = U[:, :R]
        else:
            # fallback to random if rank < R
            An = rng.standard_normal((Xn.shape[0], R))
        # small random jitter to avoid unlucky starts
        An += 1e-6 * rng.standard_normal(An.shape)
        factors.append(An)
    return factors

def _normalize_cp(factors: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Column-normalize factors and return (normalized_factors, lambdas).
    lambdas[r] holds the product of column norms across modes for component r.
    """
    N = len(factors)
    R = factors[0].shape[1]
    lambdas = np.ones(R, dtype=factors[0].dtype)
    for r in range(R):
        for n in range(N):
            c = norm(factors[n][:, r])
            if c > 0:
                factors[n][:, r] /= c
                lambdas[r] *= c
    return factors, lambdas

def cp_reconstruct(factors: list[np.ndarray], lambdas: np.ndarray | None = None) -> np.ndarray:
    """
    Dense reconstruction from CP factors (and optional weights).
    X_hat[i1,...,iN] = sum_r λ_r ∏_n A^{(n)}[i_n, r]
    """
    shapes = [A.shape[0] for A in factors]
    R = factors[0].shape[1]
    if lambdas is None:
        lambdas = np.ones(R, dtype=factors[0].dtype)

    X_hat = np.zeros(shapes, dtype=factors[0].dtype)
    # Efficient einsum: sum_r λ_r * A1[:,r] ⊗ A2[:,r] ⊗ ... ⊗ AN[:,r]
    # Build an einsum like 'ir,jr,kr->ijk' dynamically.
    subscripts = []
    operands = []
    letters = list("abcdefghijklmnopqrstuvwxyz")
    assert len(shapes) <= len(letters), "Tensor order too large for this simple einsum builder."

    for n, A in enumerate(factors):
        subscripts.append(letters[n] + "r")
        operands.append(A)

    eins = ",".join(subscripts) + "->" + "".join(letters[:len(shapes)])
    for r in range(R):
        # form a view on the r-th column of each factor
        cols = [A[:, r][:, None] for A in factors]  # keep last axis 'r' length-1
        term = np.einsum(eins, *cols, optimize=True).squeeze()
        X_hat += lambdas[r] * term
    return X_hat

def cp_frobenius_error(X: np.ndarray, factors: list[np.ndarray], lambdas: np.ndarray) -> float:
    """
    Compute ||X - X_hat||_F using the standard CP identities:
      ||X_hat||_F^2 = sum_{r,s} λ_r λ_s prod_n <a_r^{(n)}, a_s^{(n)}>
      <X, X_hat>     = sum_r λ_r * inner(X, rank-1 outer of columns r)
    For simplicity and robustness we reconstruct approximately for error.
    (For big tensors, prefer the exact Gram-based formula.)
    """
    X_hat = cp_reconstruct(factors, lambdas)
    return norm(X - X_hat)

def cp_als(
    X: np.ndarray,
    rank: int,
    max_iters: int = 200,
    tol: float = 1e-6,
    init: str = "svd",
    l2_reg: float = 0.0,
    random_state: int | None = None,
    normalize_every: int = 1,
    verbose: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """
    Canonical Polyadic (CP) decomposition via Alternating Least Squares (ALS).

    Minimize ||X - sum_{r=1}^R λ_r a_r^(1) ∘ ... ∘ a_r^(N)||_F.

    Parameters
    ----------
    X : ndarray (I1 x I2 x ... x IN)
    rank : int, CP rank R
    max_iters : int
    tol : float, stop when |fit_{k}-fit_{k-1}| < tol
    init : "svd" or "random"
    l2_reg : float, ridge regularization on normal equations (stabilizes updates)
    random_state : int | None
    normalize_every : int, how often to column-normalize and absorb weights
    verbose : bool

    Returns
    -------
    factors : list of N arrays [A^(1), ..., A^(N)], each (I_n x R)
    lambdas : (R,)
    info : dict with keys {"fit", "fit_history", "iters"}
    """
    X = np.asarray(X)
    N = X.ndim
    R = rank

    # Initialize factors
    if init == "svd":
        factors = _initialize_factors_svd(X, R, random_state=random_state)
    elif init == "random":
        rng = np.random.default_rng(random_state)
        factors = [rng.standard_normal((X.shape[n], R)) for n in range(N)]
    else:
        raise ValueError("init must be 'svd' or 'random'.")

    # Precompute Frobenius norm of X
    Xnorm = norm(X)

    fit_history = []
    prev_fit = -np.inf

    for it in range(1, max_iters + 1):
        # Precompute Gram matrices and their Hadamard product for each mode
        Grams = [A.T @ A for A in factors]

        for n in range(N):
            # Hadamard product of Grams except mode n: H = ⊙_{k≠n} (A^(k)^T A^(k))
            H = np.ones((R, R), dtype=factors[0].dtype)
            for m in range(N):
                if m == n: 
                    continue
                H *= Grams[m]

            if l2_reg > 0.0:
                H = H + l2_reg * np.eye(R, dtype=H.dtype)

            # MTTKRP for mode n
            M = _mttkrp(X, factors, mode=n)  # (I_n, R)

            # Update A^(n): solve M = A^(n) H  =>  A^(n) = M H^{-1}
            # Prefer solve over explicit inverse
            Anew = M @ solve(H, np.eye(R, dtype=H.dtype))
            factors[n] = Anew

            # Refresh Gram for the updated factor
            Grams[n] = factors[n].T @ factors[n]

        # Normalize & compute fit
        if (it % normalize_every) == 0 or it == max_iters:
            factors, lambdas = _normalize_cp(factors)
        else:
            lambdas = np.ones(R, dtype=factors[0].dtype)

        # Estimate fit via reconstruction (robust and simple)
        # For large tensors, replace by exact CP inner-products formula.
        err = cp_frobenius_error(X, factors, lambdas)
        fit = 1.0 - (err / (Xnorm + 1e-12))
        fit_history.append(float(fit))

        if verbose:
            print(f"[ALS] iter={it:03d}  fit={fit:.8f}  delta={fit - prev_fit:.3e}")

        if abs(fit - prev_fit) < tol:
            break
        prev_fit = fit

    info = {"fit": float(fit), "fit_history": fit_history, "iters": it}
    return factors, lambdas, info
