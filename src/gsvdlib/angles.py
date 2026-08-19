"""Angle utilities for the blocked GSVD factors."""

from __future__ import annotations

import numpy as np


def get_nonzero_per_column(matrix):
    """First nonzero entry of each column (0.0 for all-zero columns)."""
    M = np.asarray(matrix)
    out = np.zeros(M.shape[1], dtype=M.dtype)
    for j in range(M.shape[1]):
        nz = np.flatnonzero(M[:, j])
        if nz.size:
            out[j] = M[nz[0], j]
    return out


def get_angles_S(S_our):
    """Per-column alignment angles (degrees) from the sine factor: asin."""
    vals = np.clip(get_nonzero_per_column(S_our), -1.0, 1.0)
    return np.degrees(np.arcsin(vals))


def get_angles_C(C):
    """Per-column alignment angles (degrees) from the cosine factor: acos."""
    vals = np.clip(get_nonzero_per_column(C), -1.0, 1.0)
    return np.degrees(np.arccos(vals))


def highlight_closest(angles_deg, target_deg, k=20):
    """Indices of the ``k`` angles closest to ``target_deg``, closest first."""
    angles_deg = np.asarray(angles_deg)
    diffs = np.abs(angles_deg - target_deg)
    return np.argsort(diffs, kind="stable")[:k]


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def get_most_similar_row_H(x, H):
    """Cosine similarity of ``x`` against every row of ``H`` (vectorized)."""
    x = np.asarray(x, dtype=float).ravel()
    H = np.asarray(H, dtype=float)
    row_norms = np.linalg.norm(H, axis=1)
    return (H @ x) / (row_norms * np.linalg.norm(x))
