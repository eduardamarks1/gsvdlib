"""End-to-end experiment drivers (dataset-agnostic).

``prepare_data`` builds the sorted GSVD base for a class pair;
``run_pair_experiment`` reproduces the balanced classification + metrics
pipeline from the Julia notebook for any :class:`~gsvdlib.datasets.VectorDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classify import DEFAULT_RCOND, classify_set, linear_cka, metrics_from_angles
from .core import GSVDResult, gsvd
from .datasets import VectorDataset, balanced_count, center, sample_pair


CENTERINGS = ("pooled", "per_class")


@dataclass
class PreparedPair:
    A: np.ndarray            # centered, samples as columns
    B: np.ndarray
    mean_A: np.ndarray       # raw class means
    mean_B: np.ndarray
    gsvd: GSVDResult         # already sorted
    center_A: np.ndarray     # vector subtracted from the columns of A
    center_B: np.ndarray     # ... from the columns of B
    center_test: np.ndarray  # ... from every new sample before theta(z)
    centering: str = "pooled"


def prepare_data(ds: VectorDataset, label_A, label_B, n_A=900, n_B=800,
                 split="train", seed=1234, centering="pooled") -> PreparedPair:
    """Sample, center and decompose a class pair.

    ``centering`` chooses the reference frame:

    * ``"pooled"`` (default): A, B and every new sample z are centered by the
      same vector, the mean of all columns of A and B together. One frame for
      everything, and no class information is needed to center z.
    * ``"per_class"``: A and B are each centered by their own mean and new
      samples are left as they are. This is the protocol of the ICLR 2026
      paper and the Julia notebook; keep it to reproduce the published tables.

    Note the transposition: the GSVD is computed on the co-span formulation,
    so the (samples x features) matrices ``A.T`` / ``B.T`` are what is
    decomposed, exactly as in the Julia notebook (``Our_SVD(A', B')``).
    """
    if centering not in CENTERINGS:
        raise ValueError(f"centering must be one of {CENTERINGS}, got {centering!r}")
    A, B = sample_pair(ds, label_A, label_B, n_A, n_B, split=split, seed=seed)
    mean_A, mean_B = A.mean(axis=1), B.mean(axis=1)
    if centering == "pooled":
        _, mu = center(np.hstack([A, B]))
        center_A = center_B = center_test = mu
    else:
        center_A, center_B = mean_A, mean_B
        center_test = np.zeros_like(mean_A)
    A = A - center_A[:, None]
    B = B - center_B[:, None]
    res = gsvd(A.T, B.T).sorted()
    return PreparedPair(A=A, B=B, mean_A=mean_A, mean_B=mean_B, gsvd=res,
                        center_A=center_A, center_B=center_B,
                        center_test=center_test, centering=centering)


def evaluate_pair(ds: VectorDataset, label_A, label_B, prep: PreparedPair,
                  split="test", seed=4321, threshold_deg=45.0, n_test=None,
                  verbose=True, rcond=DEFAULT_RCOND):
    """Balanced evaluation of a prepared pair on ``split``.

    Test samples are centered with ``prep.center_test``, i.e. in the same
    frame as the decomposition (the pooled mean by default). ``rcond`` is the
    pseudo-inverse truncation passed to :func:`~gsvdlib.classify.theta_angles`.

    Returns ``(angles_A, angles_B, overall_accuracy)``.
    """
    g = prep.gsvd
    if n_test is None:
        n_test = balanced_count(ds, label_A, label_B, split=split)
    X_A, X_B = sample_pair(ds, label_A, label_B, n_test, n_test,
                           split=split, seed=seed)
    X_A = X_A - prep.center_test[:, None]
    X_B = X_B - prep.center_test[:, None]

    res_A = classify_set(X_A, "A", g.C, g.S, g.H, threshold_deg=threshold_deg, rcond=rcond)
    res_B = classify_set(X_B, "B", g.C, g.S, g.H, threshold_deg=threshold_deg, rcond=rcond)

    if verbose:
        for label, res in ((label_A, res_A), (label_B, res_B)):
            print(f"=== {ds.class_name(label)} ===  "
                  f"{res.hits}/{res.total} correct "
                  f"({100 * res.accuracy:.2f}%), "
                  f"mean θ = {res.mean_angle:.2f}° "
                  f"(σ = {res.std_angle:.2f}°)")

    overall = (res_A.hits + res_B.hits) / (res_A.total + res_B.total)
    if verbose:
        print(f"Overall accuracy: {100 * overall:.2f}%")
    return res_A.angles, res_B.angles, overall


def run_pair_experiment(ds: VectorDataset, pairs, n_A=900, n_B=800,
                        base_split="train", test_split="test",
                        base_seed=1234, test_seed=4321,
                        threshold_deg=45.0, verbose=True, centering="pooled",
                        rcond=DEFAULT_RCOND):
    """Run the full balanced pipeline for several class pairs.

    ``centering`` is passed to :func:`prepare_data` and ``rcond`` to
    :func:`evaluate_pair`. Linear CKA is always
    computed on each set centered by its own mean, so it does not depend on
    the centering chosen for theta.

    Returns ``(rows, angle_data)`` where ``rows`` is a list of per-class
    metric dicts (pair, class, precision, recall, f1, accuracy_overall, cka)
    ready for ``pandas.DataFrame(rows)``, and ``angle_data`` maps each pair
    to its ``(angles_A, angles_B)``. No global accumulators.
    """
    rows = []
    angle_data = {}
    for label_A, label_B in pairs:
        if verbose:
            print(f"\n>>> Pair: {ds.class_name(label_A)} vs {ds.class_name(label_B)}")
        prep = prepare_data(ds, label_A, label_B, n_A=n_A, n_B=n_B,
                            split=base_split, seed=base_seed,
                            centering=centering)
        angles_A, angles_B, _ = evaluate_pair(
            ds, label_A, label_B, prep, split=test_split, seed=test_seed,
            threshold_deg=threshold_deg, verbose=verbose, rcond=rcond)
        angle_data[(label_A, label_B)] = (angles_A, angles_B)

        cka = linear_cka(center(prep.A)[0], center(prep.B)[0])
        m = metrics_from_angles(angles_A, angles_B,
                                theta_threshold_deg=threshold_deg)
        for cls, side in ((label_A, "A"), (label_B, "B")):
            rows.append({
                "pair": f"{label_A} vs {label_B}",
                "class": ds.class_name(cls),
                "precision": m[f"precision_{side}"],
                "recall": m[f"recall_{side}"],
                "f1": m[f"f1_{side}"],
                "accuracy_overall": m["accuracy"],
                "cka": cka,
            })
    return rows, angle_data
