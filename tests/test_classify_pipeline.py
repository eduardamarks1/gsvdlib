import numpy as np

from gsvdlib import (ArrayDataset, classify, classify_set, gsvd, linear_cka,
                     metrics_from_angles, prepare_data, run_pair_experiment,
                     theta_angles)

RNG = np.random.default_rng(7)


def synthetic_dataset(d=50, n_per_class=120):
    """Two overlapping full-rank classes with different variance profiles:
    class A has high variance in the first half of the features, class B in
    the second half, so theta(z) should separate them."""
    w_A = np.where(np.arange(d) < d // 2, 4.0, 1.0)
    w_B = np.where(np.arange(d) < d // 2, 1.0, 4.0)
    X = np.hstack([
        w_A[:, None] * RNG.standard_normal((d, n_per_class)),
        w_B[:, None] * RNG.standard_normal((d, n_per_class)),
    ])
    labels = np.array(["a"] * n_per_class + ["b"] * n_per_class)
    return ArrayDataset(X, labels, names={"a": "Class A", "b": "Class B"})


def test_theta_separates_synthetic_classes():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    g = prep.gsvd

    angles_A = theta_angles(prep.A, g.C, g.S, g.H)
    angles_B = theta_angles(prep.B, g.C, g.S, g.H)
    assert np.mean(angles_A) < 45 < np.mean(angles_B)

    m = metrics_from_angles(angles_A, angles_B)
    assert m["accuracy"] > 0.9


def test_classify_single_vector():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    g = prep.gsvd
    label, theta = classify(prep.A[:, 0], "a", "b", g.C, g.S, g.H)
    assert label in ("a", "b")
    assert 0 <= theta <= 90


def test_classify_set_counts():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    g = prep.gsvd
    res = classify_set(prep.A, "A", g.C, g.S, g.H)
    assert res.total == 80
    assert 0 <= res.hits <= 80
    assert res.accuracy == res.hits / res.total


def test_run_pair_experiment_smoke():
    ds = synthetic_dataset()
    rows, angle_data = run_pair_experiment(ds, [("a", "b")], n_A=80, n_B=80,
                                           verbose=False)
    assert len(rows) == 2
    assert set(rows[0]) == {"pair", "class", "precision", "recall", "f1",
                            "accuracy_overall", "cka"}
    angles_A, angles_B = angle_data[("a", "b")]
    assert angles_A.size == angles_B.size > 0


def test_linear_cka_bounds():
    A = RNG.standard_normal((30, 40))
    assert abs(linear_cka(A, A) - 1.0) < 1e-12
    # orthogonal column spaces -> 0
    B = np.zeros((30, 10))
    B[15:, :] = RNG.standard_normal((15, 10))
    A2 = np.zeros((30, 10))
    A2[:15, :] = RNG.standard_normal((15, 10))
    assert linear_cka(A2, B) < 1e-12


def test_gsvd_transposed_pipeline_shapes():
    """Mimic the notebook: features x samples, decompose the transposes."""
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=70, n_B=60, seed=3)
    g = prep.gsvd
    # A.T is (70, 50): U is 70x70, H maps to feature space (cols = 50)
    assert g.U.shape == (70, 70)
    assert g.V_til.shape == (60, 60)
    assert g.H.shape[1] == 50
    assert np.linalg.norm(prep.A.T - g.U @ g.C @ g.H) < 1e-8
