import numpy as np

from gsvdlib import (ArrayDataset, classify, classify_set, evaluate_pair, gsvd,
                     linear_cka, metrics_from_angles, prepare_data,
                     run_pair_experiment, theta_angles)
from gsvdlib.classify import DEFAULT_RCOND
from gsvdlib.datasets import sample_pair

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


def test_pooled_centering_is_one_frame():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=70, n_B=60, seed=3)
    assert prep.centering == "pooled"
    # A and B share the subtracted vector, and together they have zero mean
    assert np.allclose(prep.center_A, prep.center_B)
    assert np.allclose(prep.center_test, prep.center_A)
    assert np.allclose(np.hstack([prep.A, prep.B]).mean(axis=1), 0)
    # raw class means are still available
    assert np.allclose(prep.A.mean(axis=1) + prep.center_A, prep.mean_A)


def test_per_class_centering_reproduces_paper_protocol():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=70, n_B=60, seed=3, centering="per_class")
    assert np.allclose(prep.A.mean(axis=1), 0)
    assert np.allclose(prep.B.mean(axis=1), 0)
    assert np.allclose(prep.center_test, 0)


def test_evaluate_pair_centers_test_samples():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    angles_A, angles_B, _ = evaluate_pair(ds, "a", "b", prep, split="train",
                                          seed=5, n_test=20, verbose=False)
    X_A, _ = sample_pair(ds, "a", "b", 20, 20, split="train", seed=5)
    g = prep.gsvd
    expected = theta_angles(X_A - prep.center_test[:, None], g.C, g.S, g.H)
    assert np.allclose(angles_A, expected)


def test_unknown_centering_rejected():
    ds = synthetic_dataset()
    try:
        prepare_data(ds, "a", "b", n_A=20, n_B=20, centering="global")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_truncation_matches_explicit_pinv():
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    g = prep.gsvd
    X = prep.A[:, :10]
    from gsvdlib.blocks import to_intersection
    Ci, Si = to_intersection(g.C, g.S)
    c = np.linalg.pinv(g.H.T, rcond=DEFAULT_RCOND) @ X
    expected = np.degrees(np.arctan2(np.linalg.norm(Si @ c, axis=0),
                                     np.linalg.norm(Ci @ c, axis=0)))
    assert np.allclose(theta_angles(X, g.C, g.S, g.H), expected)


def test_truncation_tames_out_of_distribution_features():
    """Energy in a feature the training data never used must not take over theta."""
    ds = synthetic_dataset()
    prep = prepare_data(ds, "a", "b", n_A=80, n_B=80, seed=1)
    g = prep.gsvd
    x = prep.A[:, :1].copy()
    # direction of H's smallest singular value: almost unseen in training
    U, _, _ = np.linalg.svd(g.H.T, full_matrices=False)
    spike = U[:, -1:] * np.linalg.norm(x)
    t_clean = theta_angles(x, g.C, g.S, g.H)[0]
    t_trunc = theta_angles(x + spike, g.C, g.S, g.H)[0]
    t_plain = theta_angles(x + spike, g.C, g.S, g.H, rcond=None)[0]
    assert abs(t_trunc - t_clean) < abs(t_plain - t_clean) + 1e-9
