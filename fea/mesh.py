"""Structured meshers written from scratch.

plate_with_hole builds a quarter-symmetry model of a plate of width W and length H with a central
hole of semi-axes a (across the plate, perpendicular to the load) and b (along the load).

Block 1 is the square [0, W/2] x [0, W/2] around the hole. Straight lines join points on the hole
to points on the square's outer edges, and nodes are spaced along each line with a smooth
exponential grading, so elements are small at the hole where stress changes fastest. The hole
corner at the diagonal maps to the square's corner. Hole nodes, including Q8 mid-side nodes, sit
exactly on the hole curve.

Block 2 is the rectangle [0, W/2] x [W/2, H/2] above it, a plain structured grid that carries the
load from the loaded end down to the hole region. Shared nodes are merged.
"""
import numpy as np


def _merge(nodes, elems, tol=1e-9):
    scale = max(1.0, np.abs(nodes).max())
    key = np.round(nodes / (tol * scale)).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return nodes[idx], inv[elems]


def _drop_unused(nodes, elems):
    used = np.unique(elems)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[elems]


def _orient(nodes, elems, etype):
    """Reorder any clockwise element to counter-clockwise."""
    p = nodes[elems[:, :4]]
    area2 = sum(p[:, i, 0] * p[:, (i + 1) % 4, 1] - p[:, (i + 1) % 4, 0] * p[:, i, 1] for i in range(4))
    bad = area2 < 0
    if etype == "Q4":
        elems[bad] = elems[bad][:, [0, 3, 2, 1]]
    else:
        elems[bad] = elems[bad][:, [0, 3, 2, 1, 7, 6, 5, 4]]
    return elems


def _grid_elems(ni, nj, etype, offset=0):
    """Element connectivity on a structured point grid indexed [i, j] -> i * npj + j.

    Q4 uses an (ni+1) x (nj+1) grid. Q8 uses a (2ni+1) x (2nj+1) grid and skips element centres.
    """
    if etype == "Q4":
        npj = nj + 1
        g = lambda i, j: offset + i * npj + j
        return np.array([[g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)]
                         for i in range(ni) for j in range(nj)])
    npj = 2 * nj + 1
    g = lambda i, j: offset + i * npj + j
    out = []
    for i in range(ni):
        for j in range(nj):
            I, J = 2 * i, 2 * j
            out.append([g(I, J), g(I + 2, J), g(I + 2, J + 2), g(I, J + 2),
                        g(I + 1, J), g(I + 2, J + 1), g(I + 1, J + 2), g(I, J + 1)])
    return np.array(out)


def _params(n, etype):
    return np.linspace(0, 1, (n + 1) if etype == "Q4" else (2 * n + 1))


def rect_mesh(x0, x1, y0, y1, nx, ny, etype):
    xs = x0 + (x1 - x0) * _params(nx, etype)
    ys = y0 + (y1 - y0) * _params(ny, etype)
    P = np.array([(x, y) for x in xs for y in ys])
    E = _grid_elems(nx, ny, etype)
    P, E = _drop_unused(P, E)
    return P, _orient(P, E, etype)


def distorted_patch(etype):
    """2 x 2 elements on [0, 2]^2 with the centre node moved, the standard patch-test mesh."""
    C = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1.25, 0.8], [2, 1], [0, 2], [1, 2], [2, 2]], float)
    quads = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    if etype == "Q4":
        return C, np.array(quads)
    nodes = list(map(tuple, C))
    elems = []
    for q in quads:
        mids = []
        for k in range(4):
            m = tuple((C[q[k]] + C[q[(k + 1) % 4]]) / 2)
            if m not in nodes:
                nodes.append(m)
            mids.append(nodes.index(m))
        elems.append(q + mids)
    return np.array(nodes), np.array(elems)


def plate_with_hole(W, H, a, b=None, n_theta=16, beta=None, etype="Q8", aspect=2.0):
    """Quarter model of a W x H plate with a central a x b elliptical hole (circle if b is None).

    n_theta  elements around the quarter hole (even); the same number are used radially.
    beta     radial grading, 0 is uniform, larger packs elements towards the hole. None picks it
             from the size of the block relative to the hole, log(c / min(a, b)) + 1, at least 2.5.
    aspect   target element aspect ratio in block 2.
    """
    b = a if b is None else b
    c = W / 2
    if not (a < c and b < c):
        raise ValueError("hole must fit inside the W/2 square block")
    if n_theta % 2:
        raise ValueError("n_theta must be even")
    n_r = n_theta
    if beta is None:
        beta = max(2.5, np.log(c / min(a, b)) + 1)

    s = _params(n_theta, etype)
    tau = _params(n_r, etype)
    t = tau if beta == 0 else (np.exp(beta * tau) - 1) / (np.exp(beta) - 1)

    # hole parameter phi, piecewise linear in s with the diagonal corner at s = 0.5
    phic = np.arctan(a / b)  # ellipse point whose direction is 45 degrees
    phi = np.where(s <= 0.5, 2 * s * phic, phic + (2 * s - 1) * (np.pi / 2 - phic))
    hole = np.c_[a * np.cos(phi), b * np.sin(phi)]
    outer = np.where((s <= 0.5)[:, None], np.c_[np.full_like(s, c), 2 * s * c],
                     np.c_[c * (2 - 2 * s), np.full_like(s, c)])
    P1 = (hole[:, None, :] + t[None, :, None] * (outer - hole)[:, None, :]).reshape(-1, 2)
    E1 = _grid_elems(n_theta, n_r, etype)

    blocks_P, blocks_E = [P1], [E1]
    half_len = H / 2
    if half_len > c * (1 + 1e-12):
        nx = n_theta // 2
        hx = c / nx
        ny = max(1, int(np.ceil((half_len - c) / (aspect * hx))))
        P2, E2 = rect_mesh(0, c, c, half_len, nx, ny, etype)
        blocks_E.append(E2 + len(P1))
        blocks_P.append(P2)

    P = np.vstack(blocks_P)
    E = np.vstack(blocks_E)
    P, E = _drop_unused(P, E)
    P, E = _merge(P, E)
    E = _orient(P, E, etype)
    return P, E
