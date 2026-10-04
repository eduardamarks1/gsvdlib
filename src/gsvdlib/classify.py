"""Alignment-angle classification and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq

from .blocks import to_intersection

#: Relative cutoff of the truncated pseudo-inverse used for c = H^+ z: singular
#: values of H below ``DEFAULT_RCOND * s_max`` are dropped. Chosen on held-out
#: validation images (examples/truncation_validation.py: 8 MNIST/Fashion-MNIST
#: pairs x 5 draws); it raised test AUC in all 40 runs. Re-validate it for data
#: of a different kind.
DEFAULT_RCOND = 0.05


def theta_angles(X, C, S, H, restrict_to_intersection=True, rcond=DEFAULT_RCOND):
    """Alignment angle theta(z) (degrees) for each column of ``X``.

    Solves ``H.T @ c = x`` with a truncated pseudo-inverse for all columns at
    once, then ``theta = atan2(||S c||, ||C c||)``. ``H`` is ill-conditioned:
    features that barely vary in training give tiny singular values, and
    without truncation any energy of x there blows ``c`` up and dominates
    theta. ``rcond`` drops singular values below ``rcond * s_max``;
    ``rcond=None`` gives the plain least-squares solution of the ICLR 2026
    paper and the Julia notebook. By default ``C`` and ``S`` are first
    restricted to col(A) intersect col(B) (the thesis hypothesis).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and H.shape[1] != 1:  # a single vector passed as 1-D
        X = X.T
    if restrict_to_intersection:
        C, S = to_intersection(C, S)

    coeffs, *_ = lstsq(np.asarray(H).T, X, cond=rcond)
    wA = C @ coeffs
    wB = S @ coeffs
    nA = np.linalg.norm(wA, axis=0)
    nB = np.linalg.norm(wB, axis=0)
    return np.degrees(np.arctan2(nB, nA))


def classify(v, label_A, label_B, C, S, H, threshold_deg=45.0, rcond=DEFAULT_RCOND):
    """Classify one vector: returns ``(predicted_label, theta_deg)``."""
    theta = float(theta_angles(np.asarray(v, dtype=float).reshape(-1, 1), C, S, H,
                               rcond=rcond)[0])
    return (label_B if theta > threshold_deg else label_A), theta


@dataclass
class SetResult:
    hits: int
    total: int
    accuracy: float
    angles: np.ndarray
    mean_angle: float
    std_angle: float


def classify_set(X, expected_side, C, S, H, threshold_deg=45.0,
                 rcond=DEFAULT_RCOND) -> SetResult:
    """Classify every column of ``X`` assuming they all belong to one side.

    ``expected_side`` is ``"A"`` (predict A iff theta < threshold) or
    ``"B"`` (predict B iff theta > threshold).
    """
    if expected_side not in ("A", "B"):
        raise ValueError("expected_side must be 'A' or 'B'")
    angles = theta_angles(X, C, S, H, rcond=rcond)
    if expected_side == "A":
        hits = int(np.count_nonzero(angles < threshold_deg))
    else:
        hits = int(np.count_nonzero(angles > threshold_deg))
    total = angles.size
    return SetResult(hits=hits, total=total, accuracy=hits / total, angles=angles,
                     mean_angle=float(np.mean(angles)), std_angle=float(np.std(angles, ddof=1)))


def metrics_from_angles(angles_A, angles_B, theta_threshold_deg=45.0):
    """Precision / recall / F1 per class plus overall accuracy, treating the
    angle distributions as a binary classifier (predict A iff theta <
    threshold). Returns a dict.
    """
    angles_A = np.asarray(angles_A)
    angles_B = np.asarray(angles_B)
    nA, nB = angles_A.size, angles_B.size

    TP_A = int(np.count_nonzero(angles_A < theta_threshold_deg))
    FN_A = nA - TP_A
    TP_B = int(np.count_nonzero(angles_B >= theta_threshold_deg))
    FN_B = nB - TP_B

    FP_A = FN_B
    FP_B = FN_A

    def safe_div(a, b):
        return a / b if b else 0.0

    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    precision_A = safe_div(TP_A, TP_A + FP_A)
    recall_A = safe_div(TP_A, nA)
    precision_B = safe_div(TP_B, TP_B + FP_B)
    recall_B = safe_div(TP_B, nB)

    return {
        "accuracy": (TP_A + TP_B) / (nA + nB),
        "precision_A": precision_A, "recall_A": recall_A, "f1_A": safe_f1(precision_A, recall_A),
        "precision_B": precision_B, "recall_B": recall_B, "f1_B": safe_f1(precision_B, recall_B),
    }


def linear_cka(A, B):
    """Linear CKA between the covariance structures of ``A`` (d x nA) and
    ``B`` (d x nB), both already feature-centered:
    ``||A' B||_F^2 / (||A' A||_F ||B' B||_F)``.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    AtB = A.T @ B
    return float(np.sum(AtB * AtB) / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))
