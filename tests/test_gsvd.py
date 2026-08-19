import numpy as np
import pytest

from gsvdlib import (gsvd, gsvd_factors, infer_block_sizes, make_A_B,
                     make_C_S, set_to_0_closest_to_45, to_intersection,
                     wire_size, zero_pure_blocks)

RNG = np.random.default_rng(42)


def random_pair(m=40, p=30, n=25):
    return RNG.standard_normal((m, n)), RNG.standard_normal((p, n))


# ---------------------------------------------------------------------------
# Raw factorization (dggsvd3 contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,p,n", [(40, 30, 25), (100, 100, 80),
                                   (100, 500, 80), (20, 15, 60)])
def test_factors_reconstruct(m, p, n):
    A = RNG.standard_normal((m, n))
    B = RNG.standard_normal((p, n))
    f = gsvd_factors(A, B)
    H = f.R0 @ f.Q.T
    assert np.linalg.norm(A - f.U @ f.C @ H) < 1e-8 * np.linalg.norm(A)
    assert np.linalg.norm(B - f.V @ f.S @ H) < 1e-8 * np.linalg.norm(B)


def test_factors_orthogonal_and_cs_identity():
    A, B = random_pair()
    f = gsvd_factors(A, B)
    t = f.t
    assert np.allclose(f.U.T @ f.U, np.eye(f.U.shape[0]), atol=1e-10)
    assert np.allclose(f.V.T @ f.V, np.eye(f.V.shape[0]), atol=1e-10)
    assert np.allclose(f.C.T @ f.C + f.S.T @ f.S, np.eye(t), atol=1e-8)


def test_rank_deficient_pair():
    # A and B share a common subspace -> nontrivial middle block after
    # projection; also [A; B] has rank < n.
    n = 30
    basis = RNG.standard_normal((n, 10))
    A = RNG.standard_normal((20, 10)) @ basis.T
    B = RNG.standard_normal((15, 10)) @ basis.T
    f = gsvd_factors(A, B)
    assert f.t == 10
    H = f.R0 @ f.Q.T
    assert np.linalg.norm(A - f.U @ f.C @ H) < 1e-8 * np.linalg.norm(A)
    assert np.linalg.norm(B - f.V @ f.S @ H) < 1e-8 * np.linalg.norm(B)


# ---------------------------------------------------------------------------
# Blocked pipeline (gsvd / Our_SVD)
# ---------------------------------------------------------------------------

def test_gsvd_blocked_reconstruction_and_layout():
    A, B = random_pair(60, 45, 35)
    res = gsvd(A, B)

    # Reconstruction with the blocked ("our") layout
    assert np.linalg.norm(A - res.U @ res.C @ res.H) < 1e-8 * np.linalg.norm(A)
    assert np.linalg.norm(B - res.V_til @ res.S @ res.H) < 1e-8 * np.linalg.norm(B)

    # Block sizes consistent
    bl, br, wl, wr, r = res.bl, res.br, res.wl, res.wr, res.r
    assert res.C.shape == (A.shape[0], bl + r + br)
    assert res.S.shape == (B.shape[0], bl + r + br)
    assert res.C_til.shape == (r, r)
    assert res.S_til.shape == (r, r)

    # S in "our" layout: zeros on top, S_til middle, I(br) bottom-right
    if wr > 0:
        assert np.allclose(res.S[:wr, :], 0)
    if br > 0:
        assert np.allclose(res.S[wr + r:, bl + r:], np.eye(br), atol=1e-10)

    # Middle blocks: strictly fractional cos/sin with c^2 + s^2 = 1
    c = np.diag(res.C[bl:bl + r, bl:bl + r])
    s = np.diag(res.S[wr:wr + r, bl:bl + r])
    assert np.all((c > 0) & (c < 1)) and np.all((s > 0) & (s < 1))
    assert np.allclose(c**2 + s**2, 1, atol=1e-8)


def test_sorted_preserves_reconstruction_and_orders_angles():
    A, B = random_pair(60, 45, 35)
    res = gsvd(A, B).sorted()
    assert np.linalg.norm(A - res.U @ res.C @ res.H) < 1e-8 * np.linalg.norm(A)
    assert np.linalg.norm(B - res.V_til @ res.S @ res.H) < 1e-8 * np.linalg.norm(B)
    s = np.diag(res.S_til)
    assert np.all(np.diff(s) >= 0)  # sines increasing -> angles 0 -> 90


def test_wire_size_matches_gsvd():
    A, B = random_pair(50, 40, 30)
    res = gsvd(A, B)
    assert wire_size(A, B) == (res.bl, res.br, res.wl, res.wr, res.r)


def test_make_A_B_roundtrip():
    A, B = random_pair(60, 45, 35)
    res = gsvd(A, B).sorted()
    A2, B2 = make_A_B(res.bl, res.br, res.wl, res.wr, res.r,
                      res.C_til, res.S_til, res.H, res.U, res.V_til)
    assert np.allclose(A, A2, atol=1e-8)
    assert np.allclose(B, B2, atol=1e-8)


# ---------------------------------------------------------------------------
# Block utilities
# ---------------------------------------------------------------------------

def test_infer_block_sizes_and_zero_pure_blocks():
    bl, br, wl, wr, r = 3, 4, 5, 6, 7
    c = np.sort(RNG.uniform(0.05, 0.95, r))[::-1]
    s = np.sqrt(1 - c**2)
    C, S = make_C_S(bl, br, wl, wr, r, np.diag(c), np.diag(s))

    assert infer_block_sizes(C, S) == (bl, br, wl, wr, r)

    C_int, S_int = zero_pure_blocks(C, S, bl, br, wl, wr, r)
    assert C_int.shape == C.shape and S_int.shape == S.shape
    assert np.all(C_int[:bl, :bl] == 0)
    assert np.all(S_int[wr + r:, bl + r:] == 0)
    # only the bl + br identity entries changed
    changed = np.count_nonzero(C - C_int) + np.count_nonzero(S - S_int)
    assert changed == bl + br

    C_int2, S_int2 = to_intersection(C, S)
    assert np.array_equal(C_int, C_int2) and np.array_equal(S_int, S_int2)


def test_set_to_0_closest_to_45():
    bl, br, wl, wr = 2, 2, 1, 1
    angles = np.deg2rad([10, 30, 44, 46, 60, 80])
    r = len(angles)
    C, S = make_C_S(bl, br, wl, wr, r, np.diag(np.cos(angles)), np.diag(np.sin(angles)))

    def zeroed_cols(Cr, Sr):
        return [j for j in range(C.shape[1])
                if np.all(Cr[:, j] == 0) and np.all(Sr[:, j] == 0)]

    # Julia semantics: the central column (44 deg, closest to 45) is zeroed
    # and ALSO heads the "left" list, so the first symmetric pair is
    # (44, 46); the second is (30, 60), and so on.
    Cr, Sr = set_to_0_closest_to_45(C, S, k=1)
    assert zeroed_cols(Cr, Sr) == [bl + 2, bl + 3]          # 44 and 46 deg

    Cr, Sr = set_to_0_closest_to_45(C, S, k=2)
    assert zeroed_cols(Cr, Sr) == [bl + 1, bl + 2, bl + 3, bl + 4]  # +30, 60
