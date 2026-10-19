"""Isoparametric quadrilateral elements: Q4 (bilinear) and Q8 (serendipity).

Plain-English note
------------------
Every element is drawn in a perfect square of "natural" coordinates (xi, eta) running from -1 to 1.
Shape functions N_i(xi, eta) blend the node values into a field inside the element. The same
functions also map the square onto the real, possibly curved, element in x-y space, which is what
"isoparametric" means. The Jacobian J of that map converts derivatives from (xi, eta) to (x, y) and
scales areas (det J), so integrals over the real element become integrals over the square, which
Gauss quadrature evaluates exactly for polynomials up to a known degree.

Local node order (counter-clockwise):
    Q4: corners 0 (-1,-1), 1 (1,-1), 2 (1,1), 3 (-1,1)
    Q8: the same corners, then mid-sides 4 (0,-1), 5 (1,0), 6 (0,1), 7 (-1,0)
"""
import numpy as np

NODE_XI = {
    "Q4": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float),
    "Q8": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                    [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float),
}

# Edges as (start, mid, end) local nodes, counter-clockwise. Q4 edges have no mid node.
EDGES = {
    "Q4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Q8": [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],
}

NEN = {"Q4": 4, "Q8": 8}


def shape(etype, xi, eta):
    """Shape functions and their natural derivatives at one point.

    Returns N with shape (nen,) and dN with shape (2, nen), rows d/dxi and d/deta.
    """
    if etype == "Q4":
        xs, es = NODE_XI["Q4"].T
        N = 0.25 * (1 + xi * xs) * (1 + eta * es)
        dxi = 0.25 * xs * (1 + eta * es)
        deta = 0.25 * es * (1 + xi * xs)
        return N, np.vstack([dxi, deta])
    if etype == "Q8":
        N = np.zeros(8)
        dN = np.zeros((2, 8))
        for i, (xs, es) in enumerate(NODE_XI["Q8"]):
            if i < 4:  # corner
                N[i] = 0.25 * (1 + xi * xs) * (1 + eta * es) * (xi * xs + eta * es - 1)
                dN[0, i] = 0.25 * xs * (1 + eta * es) * (2 * xi * xs + eta * es)
                dN[1, i] = 0.25 * es * (1 + xi * xs) * (xi * xs + 2 * eta * es)
            elif xs == 0:  # mid-side on eta = +-1
                N[i] = 0.5 * (1 - xi ** 2) * (1 + eta * es)
                dN[0, i] = -xi * (1 + eta * es)
                dN[1, i] = 0.5 * es * (1 - xi ** 2)
            else:  # mid-side on xi = +-1
                N[i] = 0.5 * (1 + xi * xs) * (1 - eta ** 2)
                dN[0, i] = 0.5 * xs * (1 - eta ** 2)
                dN[1, i] = -eta * (1 + xi * xs)
        return N, dN
    raise ValueError(etype)


def gauss_1d(n):
    return np.polynomial.legendre.leggauss(n)


def gauss_2d(etype):
    """Tensor Gauss rule. 2x2 integrates the Q4 stiffness exactly on parallelograms, 3x3 for Q8."""
    n = 2 if etype == "Q4" else 3
    p, w = gauss_1d(n)
    pts = np.array([(a, b) for a in p for b in p])
    wts = np.array([wa * wb for wa in w for wb in w])
    return pts, wts


def edge_shape(nen_edge, s):
    """1D shape functions on an edge, s in [-1, 1]. Order is (start, [mid,] end)."""
    if nen_edge == 2:
        return np.array([(1 - s) / 2, (1 + s) / 2]), np.array([-0.5, 0.5])
    return (np.array([s * (s - 1) / 2, 1 - s ** 2, s * (s + 1) / 2]),
            np.array([s - 0.5, -2 * s, s + 0.5]))
