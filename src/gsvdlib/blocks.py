"""Block-structure utilities: size inference and intersection restriction."""

from __future__ import annotations

import numpy as np


def infer_block_sizes(C, S, one_tol=1e-8):
    """Recover ``(bl, br, wl, wr, r)`` from blocked ``C``, ``S`` alone.

    Relies on the layout invariants of :func:`gsvdlib.core.make_C_S`:
    ``C`` has I(bl) top-left, ``S`` has I(br) bottom-right, and the middle
    block holds strictly fractional cosines/sines.
    """
    C = np.asarray(C)
    S = np.asarray(S)
    nrows1, ncols1 = C.shape
    nrows2, ncols2 = S.shape
    if ncols1 != ncols2:
        raise ValueError("C and S must share the column count (bl + r + br)")

    bl = 0
    for i in range(min(nrows1, ncols1)):
        if abs(C[i, i] - 1.0) <= one_tol:
            bl += 1
        else:
            break

    br = 0
    for j in range(min(nrows2, ncols2)):
        if abs(S[nrows2 - 1 - j, ncols2 - 1 - j] - 1.0) <= one_tol:
            br += 1
        else:
            break

    r = ncols1 - bl - br
    wl = nrows1 - bl - r
    wr = nrows2 - br - r

    if r < 0:
        raise ValueError("Inferred r < 0; C, S are not in make_C_S layout")
    if wl < 0:
        raise ValueError("Inferred wl < 0; C row count does not match bl + r + wl")
    if wr < 0:
        raise ValueError("Inferred wr < 0; S row count does not match wr + r + br")

    return bl, br, wl, wr, r


def zero_pure_blocks(C, S, bl, br, wl, wr, r):
    """Zero the pure-A block I(bl) of ``C`` and the pure-B block I(br) of
    ``S``, preserving the layout, so theta(z) is restricted to
    col(A) intersect col(B). Returns copies ``(C_int, S_int)``.
    """
    C = np.asarray(C)
    S = np.asarray(S)
    if C.shape != (bl + r + wl, bl + r + br):
        raise ValueError(f"C shape {C.shape} != {(bl + r + wl, bl + r + br)}")
    if S.shape != (wr + r + br, bl + r + br):
        raise ValueError(f"S shape {S.shape} != {(wr + r + br, bl + r + br)}")

    C_int = np.array(C, copy=True)
    S_int = np.array(S, copy=True)

    if bl > 0:
        C_int[:bl, :bl] = 0.0
    if br > 0:
        S_int[wr + r:, bl + r:] = 0.0

    if r > 0:
        assert np.array_equal(C_int[bl:bl + r, bl:bl + r], C[bl:bl + r, bl:bl + r])
        assert np.array_equal(S_int[wr:wr + r, bl:bl + r], S[wr:wr + r, bl:bl + r])

    return C_int, S_int


def to_intersection(C, S):
    """Infer block sizes and zero the pure blocks in one call."""
    bl, br, wl, wr, r = infer_block_sizes(C, S)
    return zero_pure_blocks(C, S, bl, br, wl, wr, r)
