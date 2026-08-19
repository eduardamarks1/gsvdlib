"""GSVD factorization compatible with Julia's ``svd(A, B)`` (LAPACK dggsvd3).

SciPy does not expose ``ggsvd3``, so the factorization is computed with the
classic Paige-Saunders / Van Loan construction:

1. QR with column pivoting of the stacked matrix ``M = [A; B]`` to find the
   effective rank ``t = k + l`` and the triangular factor ``R1``.
2. Full SVD of the top block ``Q11`` of the orthonormal basis: its singular
   values are the cosines (``alpha``, sorted descending, as in LAPACK).
3. The sines and the ``V`` factor are recovered from the bottom block.

Output contract mirrors Julia's GeneralizedSVD:

    A = U @ C @ R0 @ Q.T          (C is m x t,  Julia's D1)
    B = V @ S @ R0 @ Q.T          (S is p x t,  Julia's D2, LAPACK layout)

with ``C[i, i] = alpha_i`` (first ``k`` exactly 1) and
``S[i, k + i] = beta_{k+i}`` for ``i = 0 .. l-1`` (last ``l - r`` exactly 1).
Here ``Q`` is always the identity: downstream code only ever uses
``H = R0 @ Q.T``, so the factor is folded into ``R0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


@dataclass
class GeneralizedFactors:
    """Raw dggsvd3-style factors. ``t = k + l`` is the rank of [A; B]."""

    U: np.ndarray   # (m, m) orthogonal
    V: np.ndarray   # (p, p) orthogonal
    Q: np.ndarray   # (n, n) orthogonal (identity here)
    C: np.ndarray   # (m, t)  cosines on the diagonal   (Julia's D1)
    S: np.ndarray   # (p, t)  sines, LAPACK layout      (Julia's D2)
    R0: np.ndarray  # (t, n)
    k: int          # directions only in col(A)   (alpha = 1)
    l: int          # t - k

    @property
    def t(self) -> int:
        return self.k + self.l


def gsvd_factors(A, B, rank_tol=None, one_tol=1e-8) -> GeneralizedFactors:
    """Compute the generalized SVD of the pair ``(A, B)``.

    Parameters
    ----------
    A, B : arrays with the same number of columns ``n``.
    rank_tol : absolute tolerance on the pivoted-QR diagonal used to decide
        the rank of ``[A; B]``. Default: LAPACK-style ``max(m+p, n) * eps *
        |R[0, 0]|``.
    one_tol : a cosine/sine is snapped to exactly 1 (and its partner to 0)
        when within ``one_tol`` of 1, reproducing the exact 1s that LAPACK
        emits and that block-size detection relies on.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    m, n = A.shape
    p, n2 = B.shape
    if n != n2:
        raise ValueError(f"A and B must have the same column count (got {n} and {n2})")

    M = np.vstack([A, B])
    Qm, R, piv = qr(M, mode="economic", pivoting=True)
    rdiag = np.abs(np.diag(R))
    if rank_tol is None:
        rank_tol = max(m + p, n) * np.finfo(float).eps * (rdiag[0] if rdiag.size else 0.0)
    t = int(np.count_nonzero(rdiag > rank_tol))
    if t == 0:
        raise ValueError("[A; B] is (numerically) zero; GSVD is undefined")

    inv_piv = np.empty(n, dtype=int)
    inv_piv[piv] = np.arange(n)
    Q1 = Qm[:, :t]
    R1 = R[:t, :][:, inv_piv]  # M = Q1 @ R1

    Q11 = Q1[:m, :]  # (m, t)
    Q21 = Q1[m:, :]  # (p, t)

    U, alpha, Wt = np.linalg.svd(Q11, full_matrices=True)  # alpha descending
    W = Wt.T  # (t, t)
    alpha_full = np.zeros(t)
    alpha_full[: alpha.size] = np.clip(alpha, 0.0, 1.0)

    Z = Q21 @ W  # (p, t); column norms are the sines
    beta = np.linalg.norm(Z, axis=0)

    # Snap the pure blocks to exact 0/1 so that alpha^2 + beta^2 = 1 holds
    # and block detection (which looks for exact 1s) is robust.
    pure_A = beta <= one_tol            # alpha ~ 1  -> first k columns
    pure_B = alpha_full <= one_tol      # beta  ~ 1  -> last l - r columns
    alpha_full[pure_A] = 1.0
    beta[pure_A] = 0.0
    alpha_full[pure_B] = 0.0
    beta[pure_B] = 1.0

    k = int(np.count_nonzero(pure_A))
    l = t - k

    # C (m x t): alphas on the main diagonal.
    C = np.zeros((m, t))
    d = min(m, t)
    C[np.arange(d), np.arange(d)] = alpha_full[:d]

    # S (p x t): S[i, k+i] = beta[k+i], i = 0..l-1 (LAPACK layout).
    S = np.zeros((p, t))
    if l > 0:
        S[np.arange(l), k + np.arange(l)] = beta[k:]

    # V: first l columns from the normalized columns of Z; complete to an
    # orthogonal basis (re-orthonormalized via QR to absorb roundoff).
    if l > 0:
        Vl = Z[:, k:] / beta[k:]
        Qv, Rv = np.linalg.qr(Vl, mode="complete")
        signs = np.sign(np.diag(Rv[:l, :l]))
        signs[signs == 0] = 1.0
        Qv[:, :l] *= signs
        V = Qv
    else:
        V = np.eye(p)

    R0 = W.T @ R1  # (t, n);  A = U C R0,  B = V S R0
    Qfac = np.eye(n)

    return GeneralizedFactors(U=U, V=V, Q=Qfac, C=C, S=S, R0=R0, k=k, l=l)
