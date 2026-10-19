"""Verification tests: is the code solving the equations of elasticity correctly?"""
import numpy as np
import pytest

from fea import problems, analytic, mesh
from fea.elements import shape, NODE_XI
from fea.solver import Model


@pytest.mark.parametrize("etype", ["Q4", "Q8"])
def test_shape_functions_partition_of_unity_and_kronecker(etype):
    for xi, eta in [(-0.3, 0.7), (0.0, 0.0), (0.9, -0.2)]:
        N, dN = shape(etype, xi, eta)
        assert abs(N.sum() - 1) < 1e-14
        assert np.allclose(dN.sum(axis=1), 0, atol=1e-14)
    for i, (xi, eta) in enumerate(NODE_XI[etype]):
        N, _ = shape(etype, xi, eta)
        assert np.allclose(N, np.eye(len(N))[i], atol=1e-14)


@pytest.mark.parametrize("etype", ["Q4", "Q8"])
def test_shape_derivatives_match_finite_differences(etype):
    xi, eta, h = 0.31, -0.47, 1e-6
    _, dN = shape(etype, xi, eta)
    fd_xi = (shape(etype, xi + h, eta)[0] - shape(etype, xi - h, eta)[0]) / (2 * h)
    fd_eta = (shape(etype, xi, eta + h)[0] - shape(etype, xi, eta - h)[0]) / (2 * h)
    assert np.allclose(dN[0], fd_xi, atol=1e-8) and np.allclose(dN[1], fd_eta, atol=1e-8)


@pytest.mark.parametrize("etype", ["Q4", "Q8"])
def test_rigid_body_modes_give_zero_force(etype):
    P, T = mesh.rect_mesh(0, 2, 0, 1, 3, 2, etype)
    m = Model(P, T, etype)
    K = m.stiffness()
    x, y = P[:, 0], P[:, 1]
    for u in (np.c_[np.ones_like(x), 0 * x], np.c_[0 * x, np.ones_like(x)], np.c_[-y, x]):
        assert np.max(np.abs(K @ u.ravel())) < 1e-6 * np.abs(K).max()


@pytest.mark.parametrize("etype", ["Q4", "Q8"])
def test_patch_test_distorted_mesh(etype):
    r = problems.patch_test(etype)
    assert r["disp_err"] < 1e-12 and r["stress_err"] < 1e-12


def test_q8_reproduces_pure_bending_exactly():
    r = problems.pure_bending("Q8")
    assert abs(r["tip_ratio"] - 1) < 1e-9 and r["stress_err"] < 1e-9


def test_q4_shear_locks_in_bending():
    # Bilinear Q4 elements cannot bend without also shearing, so with two elements through the
    # depth the beam is far too stiff. This is a known property (shear locking), not a bug.
    assert problems.pure_bending("Q4")["tip_ratio"] < 0.8


def test_kirsch_field_is_in_equilibrium_and_has_peak_3():
    a, s0, h = 1.0, 1.0, 1e-5
    x, y = 1.7, 0.9
    dsxx = (analytic.kirsch(x + h, y, a, s0)[0] - analytic.kirsch(x - h, y, a, s0)[0]) / (2 * h)
    dsxy_y = (analytic.kirsch(x, y + h, a, s0)[2] - analytic.kirsch(x, y - h, a, s0)[2]) / (2 * h)
    dsxy_x = (analytic.kirsch(x + h, y, a, s0)[2] - analytic.kirsch(x - h, y, a, s0)[2]) / (2 * h)
    dsyy = (analytic.kirsch(x, y + h, a, s0)[1] - analytic.kirsch(x, y - h, a, s0)[1]) / (2 * h)
    assert abs(dsxx + dsxy_y) < 1e-8 and abs(dsxy_x + dsyy) < 1e-8
    assert abs(analytic.kirsch(a, 0.0, a, s0)[1] - 3 * s0) < 1e-12
    far = analytic.kirsch(0.0, 1e4, a, s0)
    assert np.allclose(far, [0, s0, 0], atol=1e-6)
    hole = analytic.kirsch(0.0, a, a, s0)  # top of the hole, along the load
    assert abs(hole[0] + s0) < 1e-12  # hoop stress is -s0 there


@pytest.mark.parametrize("etype,order", [("Q4", 1.0), ("Q8", 2.0)])
def test_convergence_order_on_exact_kirsch_problem(etype, order):
    errs = [problems.kirsch_exact_traction(etype, n) for n in (8, 16, 32)]
    rate = np.log(errs[-2]["energy_err"] / errs[-1]["energy_err"]) / np.log(errs[-2]["h"] / errs[-1]["h"])
    assert abs(rate - order) < 0.1


def test_global_equilibrium_on_finite_plate():
    r = problems.finite_plate(0.5, n=16)
    assert r["equilibrium_err"] < 1e-10


def test_peak_stress_is_at_hole_edge():
    r = problems.finite_plate(0.3, n=16)
    assert abs(r["peak"] - r["max_syy_on_model"]) < 1e-9 * r["peak"]


def test_reference_formulas_agree_at_small_hole():
    # all three finite-width formulas must tend to the Kirsch value of 3 as d/W -> 0
    for f in (analytic.ktn_howland, analytic.ktn_heywood, analytic.ktn_poly):
        assert abs(f(0.0) - 3) < 0.01
