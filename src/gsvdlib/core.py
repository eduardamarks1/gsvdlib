"""Core GSVD pipeline: ``gsvd`` (the Julia ``Our_SVD``), block permutation and
sorting. Naming follows the paper: ``C`` is the cosine factor (Julia's D1)
and ``S`` is the sine factor (Julia's D2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .angles import get_nonzero_per_column
from .lapack_gsvd import gsvd_factors


@dataclass
class GSVDResult:
    """Result of :func:`gsvd` in the paper's blocked layout.

    Reconstruction identities::

        A = U @ C @ H
        B = V_til @ S @ H

    ``C`` is (bl+r+wl) x (bl+r+br) with I(bl) top-left and ``C_til`` in the
    middle; ``S`` is (wr+r+br) x (bl+r+br) with ``S_til`` in the middle and
    I(br) bottom-right (the "our" layout - never the raw LAPACK one).
    """

    U: np.ndarray
    V_til: np.ndarray       # V @ P  (use this with S)
    H: np.ndarray           # R0 @ Q.T
    C_til: np.ndarray       # (r, r) diagonal, cosines in (0, 1)
    S_til: np.ndarray       # (r, r) diagonal, sines in (0, 1)
    C: np.ndarray           # full blocked cosine factor  (Julia's D1)
    S: np.ndarray           # full blocked sine factor    (Julia's D2_our)
    bl: int
    br: int
    wl: int
    wr: int
    r: int
    P: np.ndarray           # right permutation applied to V
    V: np.ndarray           # original (unpermuted) V

    def sorted(self) -> "GSVDResult":
        """Return a copy with the shared block sorted by increasing sine
        (i.e. alignment angle increasing from 0 to 90 degrees)."""
        U, V, C_til, S_til, C, S, H = sort_and_rebuild(
            self.U, self.V_til, self.C, self.S, self.C_til, self.S_til,
            self.H, self.bl, self.br, self.wl, self.wr, self.r,
        )
        return replace(self, U=U, V_til=V, C_til=C_til, S_til=S_til, C=C, S=S, H=H)


def gsvd(A, B, rank_tol=None, one_tol=1e-8) -> GSVDResult:
    """Blocked GSVD of the pair ``(A, B)`` (the Julia ``Our_SVD``).

    The sine factor returned by LAPACK is not blocked the way the paper
    defines it; the permutation ``P`` fixes that and is already folded into
    ``V_til`` and ``S``. Never combine the raw LAPACK sine factor with these
    outputs.
    """
    f = gsvd_factors(A, B, rank_tol=rank_tol, one_tol=one_tol)
    m = f.U.shape[0]
    p = f.V.shape[0]

    bl, br, wl, wr, r = _wire_size_from_factors(f.C, f.S, m, p)

    P, S_our = permutation(wr, r, br, f.S)
    V_til = f.V @ P

    C_til = f.C[bl:bl + r, bl:bl + r]
    S_til = S_our[wr:wr + r, bl:bl + r]

    H = f.R0 @ f.Q.T

    return GSVDResult(U=f.U, V_til=V_til, H=H, C_til=C_til, S_til=S_til,
                      C=f.C, S=S_our, bl=bl, br=br, wl=wl, wr=wr, r=r,
                      P=P, V=f.V)


def _wire_size_from_factors(C, S, m, p, one_tol=1e-8):
    """Block sizes from already-computed LAPACK-layout factors."""
    _, k_plus_l = C.shape

    k = 0
    for i in range(min(m, k_plus_l)):
        if abs(C[i, i] - 1.0) <= one_tol:
            k += 1
        else:
            break

    l = k_plus_l - k
    r = int(np.count_nonzero([abs(S[i, k + i] - 1.0) > one_tol for i in range(l)]))

    bl = k
    br = l - r
    wl = m - k - r
    wr = p - l
    return bl, br, wl, wr, r


def wire_size(A, B):
    """Block sizes ``(bl, br, wl, wr, r)`` of the pair (API-compatible with
    the Julia function; prefer :func:`gsvd`, which computes them for free)."""
    f = gsvd_factors(A, B)
    return _wire_size_from_factors(f.C, f.S, f.U.shape[0], f.V.shape[0])


def permutation(wr, r, br, S_lapack):
    """Row permutation taking the LAPACK sine factor to the paper's layout.

    Returns ``(P, S_our)`` where ``S_our = P_til @ S_lapack`` and ``P`` is the
    matching right-permutation for ``V`` (``V_til = V @ P``).
    """
    p = wr + r + br

    P = np.zeros((p, p))
    P[0:r, wr:wr + r] = np.eye(r)
    P[r:r + br, wr + r:] = np.eye(br)
    P[r + br:, 0:wr] = np.eye(wr)

    P_til = np.zeros((p, p))
    P_til[0:wr, r + br:] = np.eye(wr)
    P_til[wr:wr + r, 0:r] = np.eye(r)
    P_til[wr + r:, r:r + br] = np.eye(br)

    S_our = P_til @ np.asarray(S_lapack)
    return P, S_our


def make_C_S(bl, br, wl, wr, r, C_til, S_til):
    """Assemble the full blocked factors from the central diagonal blocks
    (the Julia ``Make_D1_D2``)."""
    C_til = np.atleast_2d(np.asarray(C_til, dtype=float))
    S_til = np.atleast_2d(np.asarray(S_til, dtype=float))

    C = np.block([
        [np.eye(bl),         np.zeros((bl, r)), np.zeros((bl, br))],
        [np.zeros((r, bl)),  C_til,             np.zeros((r, br))],
        [np.zeros((wl, bl)), np.zeros((wl, r)), np.zeros((wl, br))],
    ])
    S = np.block([
        [np.zeros((wr, bl)), np.zeros((wr, r)), np.zeros((wr, br))],
        [np.zeros((r, bl)),  S_til,             np.zeros((r, br))],
        [np.zeros((br, bl)), np.zeros((br, r)), np.eye(br)],
    ])
    return C, S


def make_A_B(bl, br, wl, wr, r, C_til, S_til, H, U, V):
    """Rebuild ``A = U C H`` and ``B = V S H`` from the blocked factors
    (the Julia ``Make_A_B``)."""
    C, S = make_C_S(bl, br, wl, wr, r, C_til, S_til)
    return U @ C @ H, V @ S @ H


def sort_and_rebuild(U, V, C, S, C_til, S_til, H, bl, br, wl, wr, r):
    """Sort the shared block by increasing sine and rebuild every factor
    consistently (the Julia ``sort_and_rebuild``).

    Returns ``(U_sorted, V_sorted, C_til_sorted, S_til_sorted, C_sorted,
    S_sorted, H_sorted)``.
    """
    c_vals = np.asarray(get_nonzero_per_column(C_til))
    s_vals = np.asarray(get_nonzero_per_column(S_til))

    perm = np.argsort(s_vals, kind="stable")

    H_sorted = np.array(H, copy=True)
    H_sorted[bl:bl + r, :] = H[bl:bl + r, :][perm, :]

    U_sorted = np.array(U, copy=True)
    U_sorted[:, bl:bl + r] = U[:, bl:bl + r][:, perm]

    V_sorted = np.array(V, copy=True)
    V_sorted[:, wr:wr + r] = V[:, wr:wr + r][:, perm]

    C_til_sorted = np.diag(c_vals[perm])
    S_til_sorted = np.diag(s_vals[perm])

    C_sorted, S_sorted = make_C_S(bl, br, wl, wr, r, C_til_sorted, S_til_sorted)

    return U_sorted, V_sorted, C_til_sorted, S_til_sorted, C_sorted, S_sorted, H_sorted


def set_to_0_closest_to_45(C, S, k):
    """Zero the column whose angle is closest to 45 degrees plus ``k``
    symmetric pairs around it (the Julia ``set_to_0_closest_to_45``).

    Returns copies ``(C_reduced, S_reduced)``.
    """
    from .angles import get_angles_C

    angles = np.asarray(get_angles_C(C))

    idx_central = int(np.argmin(np.abs(angles - 45.0)))

    Cr = np.array(C, copy=True)
    Sr = np.array(S, copy=True)

    Cr[:, idx_central] = 0.0
    Sr[:, idx_central] = 0.0

    left = [i for i in range(angles.size) if angles[i] < 45.0]
    right = [i for i in range(angles.size) if angles[i] > 45.0]

    left.sort(key=lambda i: -angles[i])   # closest to 45 first
    right.sort(key=lambda i: angles[i])

    for i in range(min(k, min(len(left), len(right)))):
        for j in (left[i], right[i]):
            Cr[:, j] = 0.0
            Sr[:, j] = 0.0

    return Cr, Sr
