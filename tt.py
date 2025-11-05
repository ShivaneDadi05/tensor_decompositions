# tt.py
# Tensor Train via TT-SVD (NumPy-only) + STRICT reconstruction (no silent fallbacks).

from __future__ import annotations
import numpy as np

__all__ = [
    "tt_svd",
    "tt_reconstruct",
    "tt_error_and_compression",
    "tt_params",
    "choose_rank_from_tail",
]

# ---------- Rank selection ----------

def choose_rank_from_tail(S: np.ndarray, delta: float) -> int:
    """
    Given singular values S (descending) and per-step tail budget `delta`,
    return the smallest r such that sum_{i>r} S[i]^2 <= delta^2.
    Ensures r >= 1 (prevents rank-0 cores).
    """
    S = np.asarray(S)
    if delta <= 0:
        return max(1, int(len(S)))
    delta2 = float(delta) ** 2
    tail2_from_end = np.cumsum((S[::-1] ** 2))
    nsv = len(S)
    tail2_all_r = np.empty(nsv + 1, dtype=S.dtype)
    tail2_all_r[nsv] = 0.0
    tail2_all_r[:nsv] = tail2_from_end[::-1]
    r_candidates = np.nonzero(tail2_all_r <= delta2)[0]
    r = int(r_candidates[0]) if r_candidates.size else nsv
    return max(1, r)

# ---------- TT-SVD ----------

def tt_svd(X: np.ndarray, eps: float, max_rank: int | None = None, dtype=np.float64):
    """
    TT-SVD with global relative Frobenius error target `eps`.
    Distributes truncation budget evenly across d-1 SVDs.

    Returns
    -------
    cores : list of TT cores
        G[0] : (1, n1, r1)
        G[k] : (r_{k-1}, n_{k+1}, r_{k+1}) for k=1..d-2
        G[-1]: (r_{d-1}, nd, 1)
    ranks : list of ints [1, r1, ..., r_{d-1}, 1]
    """
    X = np.asarray(X, dtype=dtype)
    n = X.shape
    d = X.ndim
    assert d >= 2, "TT requires order >= 2"

    Xnorm = np.linalg.norm(X)
    if Xnorm == 0:
        cores = []
        r_prev = 1
        for k in range(d - 1):
            cores.append(np.zeros((r_prev, n[k], 1), dtype=dtype))
            r_prev = 1
        cores.append(np.zeros((r_prev, n[-1], 1), dtype=dtype))
        return cores, [1] * (d + 1)

    delta = float(eps) * float(Xnorm) / np.sqrt(d - 1)

    cores: list[np.ndarray] = []
    r_prev = 1
    Z = X.reshape(n)

    for k in range(d - 1):
        nk = n[k]
        Z = Z.reshape(r_prev * nk, -1)               # (r_prev*nk, prod_{j>k} n_j)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)

        r_k = choose_rank_from_tail(S, delta)
        if max_rank is not None:
            r_k = min(r_k, int(max_rank))

        U = U[:, :r_k]
        S = S[:r_k]
        Vt = Vt[:r_k, :]

        Gk = U.reshape(r_prev, nk, r_k)             # STRICT axis order
        cores.append(Gk)

        Z = (S[:, None] * Vt)                       # (r_k, prod_{j>k+1} n_j)
        r_prev = r_k

    nd = n[-1]
    Gd = Z.reshape(r_prev, nd, 1)                   # STRICT axis order
    cores.append(Gd)

    ranks = [1] + [c.shape[2] for c in cores[:-1]] + [1]
    return cores, ranks

# ---------- STRICT reconstruction (no heuristics, only valid permutations) ----------

_PERMS = [
    (0, 1, 2),  # (r_prev, n, r_next)
    (1, 0, 2),  # (n, r_prev, r_next)
    (0, 2, 1),  # (r_prev, r_next, n)
    (2, 1, 0),  # (r_next, n, r_prev)
    (1, 2, 0),  # (n, r_next, r_prev)
    (2, 0, 1),  # (r_next, r_prev, n)
]

def _force_left_rank(core: np.ndarray, left_rank_expected: int, idx: int) -> np.ndarray:
    """
    Return a view of `core` whose first axis equals left_rank_expected by trying all 6 permutations.
    If none matches, raise with a clear diagnostic.
    """
    G = np.asarray(core)
    if G.ndim != 3:
        raise ValueError(f"[core {idx}] TT core must be 3D, got shape {G.shape}")

    for perm in _PERMS:
        Gp = G.transpose(perm)
        if Gp.shape[0] == left_rank_expected:
            return Gp

    # Nothing matched—explain what we saw
    details = " | ".join([f"perm {perm}->shape {G.transpose(perm).shape}" for perm in _PERMS])
    raise ValueError(
        f"[core {idx}] Cannot align left rank to {left_rank_expected}. "
        f"Original shape {G.shape}. Tried permutations: {details}"
    )

def tt_reconstruct(cores: list[np.ndarray], dtype=np.float64) -> np.ndarray:
    """
    Dense reconstruction from TT cores. STRICT chain matching:
      - first core must have left rank 1 (by permutation or error)
      - each subsequent core's left rank must match previous right rank
    No silent fallbacks.
    """
    if len(cores) == 0:
        return np.array(0.0, dtype=dtype)

    # First core: force left rank = 1
    G0 = _force_left_rank(cores[0], left_rank_expected=1, idx=0)
    if G0.shape[0] != 1:
        raise ValueError(f"[core 0] After permutation, left rank != 1: shape {G0.shape}")
    fixed = [np.asarray(G0, dtype=dtype)]

    # Remaining cores: force left rank to previous right rank
    for k in range(1, len(cores)):
        left_rank_expected = fixed[-1].shape[2]
        Gk = _force_left_rank(cores[k], left_rank_expected=left_rank_expected, idx=k)
        if Gk.shape[0] != left_rank_expected:
            raise ValueError(
                f"[core {k}] Left rank {Gk.shape[0]} != previous right rank {left_rank_expected}."
            )
        fixed.append(np.asarray(Gk, dtype=dtype))

    n_modes = [G.shape[1] for G in fixed]
    M = fixed[0][0, :, :]                           # (n1, r1)
    for k in range(1, len(fixed) - 1):
        G = fixed[k]                                # (r_{k-1}, n_k, r_k)
        # Check inner bond explicitly before einsum
        if M.shape[1] != G.shape[0]:
            raise ValueError(
                f"[contract {k}] mismatch: M last dim {M.shape[1]} vs core left rank {G.shape[0]}"
            )
        M = np.einsum("is, sjt -> ijt", M, G)       # (n_prev, n_k, r_k)
        M = M.reshape(M.shape[0] * G.shape[1], G.shape[2])  # (n_prev*n_k, r_k)

    Gd = fixed[-1]                                  # (r_{d-1}, n_d, 1)
    if M.shape[1] != Gd.shape[0]:
        raise ValueError(
            f"[final contract] mismatch: M last dim {M.shape[1]} vs last core left rank {Gd.shape[0]}"
        )
    Xhat = np.einsum("ir, rj -> ij", M, Gd[:, :, 0]).reshape(n_modes)
    return Xhat

# ---------- Metrics ----------

def tt_params(cores: list[np.ndarray]) -> int:
    """Count parameters in TT cores: sum r_{k-1} * n_k * r_k."""
    return int(sum(int(c.shape[0]) * int(c.shape[1]) * int(c.shape[2]) for c in cores))

def tt_error_and_compression(X: np.ndarray, cores: list[np.ndarray]) -> tuple[float, float, int]:
    """
    Return (relative_error, compression_ratio, tt_param_count),
    where S = (#original params) / (#TT params).
    """
    X = np.asarray(X)
    Xhat = tt_reconstruct(cores, dtype=X.dtype)
    rel_err = np.linalg.norm(X - Xhat) / (np.linalg.norm(X) + 1e-12)
    original = int(np.prod(X.shape))
    tt_p = tt_params(cores)
    S = original / tt_p if tt_p > 0 else np.inf
    return float(rel_err), float(S), tt_p
