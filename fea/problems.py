"""Standard problems built on the solver. Each returns plain numbers so tests and studies share them."""
import time
import numpy as np

from .solver import Model
from . import mesh, analytic

E_STEEL, NU_STEEL = 210e3, 0.3  # MPa, structural steel; stress results for traction loading do not depend on E
TOL = 1e-9


def patch_test(etype):
    """Prescribe u = linear field on the boundary of a distorted patch; interior must match exactly."""
    P, T = mesh.distorted_patch(etype)
    m = Model(P, T, etype, E_STEEL, NU_STEEL)
    ux = lambda x, y: 1e-3 * (1 + 2 * x + 3 * y)
    uy = lambda x, y: 1e-3 * (-1 + 0.5 * x - 1.5 * y)
    bnd = np.where((np.abs(P[:, 0]) < TOL) | (np.abs(P[:, 0] - 2) < TOL) |
                   (np.abs(P[:, 1]) < TOL) | (np.abs(P[:, 1] - 2) < TOL))[0]
    m.fix(bnd, 0, ux(*P[bnd].T))
    m.fix(bnd, 1, uy(*P[bnd].T))
    m.solve()
    u_exact = np.c_[ux(*P.T), uy(*P.T)].ravel()
    strain = 1e-3 * np.array([2, -1.5, 3 + 0.5])
    s_exact = m.D @ strain
    s_nodes = m.nodal_stress()
    return {
        "disp_err": float(np.max(np.abs(m.u - u_exact)) / np.max(np.abs(u_exact))),
        "stress_err": float(np.max(np.abs(s_nodes - s_exact)) / np.max(np.abs(s_exact))),
    }


def pure_bending(etype, nx=10, ny=2, L=10.0, h=1.0, k=10.0):
    """Cantilever under end moment, sxx = k y. Fixed ux on x = 0, uy at the origin."""
    P, T = mesh.rect_mesh(0, L, -h / 2, h / 2, nx, ny, etype)
    m = Model(P, T, etype, E_STEEL, NU_STEEL)
    left = np.where(np.abs(P[:, 0]) < TOL)[0]
    m.fix(left, 0, 0.0)
    origin = np.where((np.abs(P[:, 0]) < TOL) & (np.abs(P[:, 1]) < TOL))[0]
    m.fix(origin, 1, 0.0)
    m.add_traction(m.edges_where(lambda x, y: abs(x - L) < TOL), lambda x, y, nx_, ny_: (k * y, 0.0))
    m.solve()
    tip = np.where((np.abs(P[:, 0] - L) < TOL) & (np.abs(P[:, 1]) < TOL))[0][0]
    v_fe = m.u[2 * tip + 1]
    _, v_ex = analytic.bending_exact(L, 0.0, k, m.E, m.nu)
    s = m.nodal_stress()
    return {
        "tip_fe": float(v_fe), "tip_exact": float(v_ex),
        "tip_ratio": float(v_fe / v_ex),
        "stress_err": float(np.max(np.abs(s[:, 0] - k * P[:, 1])) / (k * h / 2)),
    }


def _hole_node(P, a):
    return int(np.argmin(np.hypot(P[:, 0] - a, P[:, 1])))


def kirsch_exact_traction(etype, n, a=1.0, L=5.0, beta=2.5, s0=100.0):
    """Method of exact solutions: Kirsch field applied as tractions on x = L and y = L."""
    P, T = mesh.plate_with_hole(2 * L, 2 * L, a, None, n, beta, etype)
    m = Model(P, T, etype, E_STEEL, NU_STEEL)
    m.fix(np.where(np.abs(P[:, 0]) < TOL)[0], 0, 0.0)
    m.fix(np.where(np.abs(P[:, 1]) < TOL)[0], 1, 0.0)

    def trac(x, y, nx_, ny_):
        s = analytic.kirsch(x, y, a, s0)
        return s[0] * nx_ + s[2] * ny_, s[2] * nx_ + s[1] * ny_

    m.add_traction(m.edges_where(lambda x, y: abs(x - L) < TOL), trac)
    m.add_traction(m.edges_where(lambda x, y: abs(y - L) < TOL), trac)
    t0 = time.perf_counter()
    m.solve()
    t_solve = time.perf_counter() - t0
    err = m.energy_error(lambda x, y: analytic.kirsch(x, y, a, s0))
    s = m.nodal_stress()
    k = _hole_node(P, a)
    area = L * L - np.pi * a * a / 4
    return {"etype": etype, "n": n, "elements": len(T), "dofs": m.ndof,
            "h": float(np.sqrt(area / len(T))), "energy_err": float(err),
            "kt": float(s[k, 1] / s0), "solve_s": t_solve, "model": m}


def finite_plate(dW, ab=1.0, n=32, W=100.0, HW=4.0, beta=None, etype="Q8", s0=100.0, keep=False):
    """Plate W wide and HW * W long, central hole of width d = dW * W across the load, a/b = ab.

    Returns net and gross stress concentration factors from the nodal-averaged peak sigma_yy at the
    hole edge, plus a global equilibrium check (reaction on the symmetry plane against applied load).
    """
    a = dW * W / 2
    b = a / ab
    P, T = mesh.plate_with_hole(W, HW * W, a, b, n, beta, etype)
    m = Model(P, T, etype, E_STEEL, NU_STEEL)
    sym_x = np.where(np.abs(P[:, 0]) < TOL)[0]
    sym_y = np.where(np.abs(P[:, 1]) < TOL)[0]
    m.fix(sym_x, 0, 0.0)
    m.fix(sym_y, 1, 0.0)
    top = HW * W / 2
    m.add_traction(m.edges_where(lambda x, y: abs(y - top) < TOL * W), lambda x, y, nx_, ny_: (0.0, s0))
    t0 = time.perf_counter()
    m.stiffness()
    m.solve()
    t_solve = time.perf_counter() - t0
    s = m.nodal_stress()
    k = _hole_node(P, a)
    peak = s[k, 1]
    applied = s0 * W / 2
    reaction = -m.reactions[2 * sym_y + 1].sum()
    out = {"dW": dW, "ab": ab, "n": n, "elements": len(T), "dofs": m.ndof,
           "peak": float(peak), "ktg": float(peak / s0), "ktn": float(peak * (1 - dW) / s0),
           "max_syy_on_model": float(s[:, 1].max()), "equilibrium_err": float(abs(reaction - applied) / applied),
           "solve_s": t_solve}
    if keep:
        out["model"] = m
    return out
