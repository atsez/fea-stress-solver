"""Plane-stress linear elastic finite element model with sparse assembly.

Flow: element stiffness (vectorised over all elements at each Gauss point) -> sparse global K ->
loads from edge tractions -> displacement boundary conditions -> sparse direct solve -> stresses.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elements import shape, gauss_2d, gauss_1d, edge_shape, NODE_XI, EDGES, NEN


def plane_stress_D(E, nu):
    """Stress = D @ strain, with strain = (exx, eyy, gamma_xy)."""
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


class Model:
    def __init__(self, nodes, elems, etype, E=210e3, nu=0.3, thickness=1.0):
        self.X = np.asarray(nodes, float)
        self.T = np.asarray(elems, int)
        self.etype = etype
        self.nen = NEN[etype]
        assert self.T.shape[1] == self.nen
        self.E, self.nu, self.t = E, nu, thickness
        self.D = plane_stress_D(E, nu)
        self.ndof = 2 * len(self.X)
        self.f = np.zeros(self.ndof)
        self.fixed = {}
        self.u = None

    # ---------- element kinematics ----------
    def _B_at(self, xi, eta):
        """B matrices (ne, 3, 2*nen) and det J (ne,) for every element at one natural point."""
        _, dN = shape(self.etype, xi, eta)
        Xe = self.X[self.T]                              # (ne, nen, 2)
        J = np.einsum("an,enb->eab", dN, Xe)             # (ne, 2, 2)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        invJ = np.empty_like(J)
        invJ[:, 0, 0] = J[:, 1, 1] / detJ
        invJ[:, 1, 1] = J[:, 0, 0] / detJ
        invJ[:, 0, 1] = -J[:, 0, 1] / detJ
        invJ[:, 1, 0] = -J[:, 1, 0] / detJ
        dNdx = np.einsum("eab,bn->ean", invJ, dN)        # (ne, 2, nen)
        B = np.zeros((len(self.T), 3, 2 * self.nen))
        B[:, 0, 0::2] = dNdx[:, 0]
        B[:, 1, 1::2] = dNdx[:, 1]
        B[:, 2, 0::2] = dNdx[:, 1]
        B[:, 2, 1::2] = dNdx[:, 0]
        return B, detJ

    def _edofs(self):
        e = np.empty((len(self.T), 2 * self.nen), int)
        e[:, 0::2] = 2 * self.T
        e[:, 1::2] = 2 * self.T + 1
        return e

    # ---------- assembly ----------
    def stiffness(self):
        pts, wts = gauss_2d(self.etype)
        Ke = np.zeros((len(self.T), 2 * self.nen, 2 * self.nen))
        for (xi, eta), w in zip(pts, wts):
            B, detJ = self._B_at(xi, eta)
            if np.any(detJ <= 0):
                raise ValueError("Element with non-positive Jacobian: mesh is inverted or badly distorted")
            Ke += np.einsum("eia,ij,ejb->eab", B, self.D, B) * (w * detJ * self.t)[:, None, None]
        ed = self._edofs()
        rows = np.repeat(ed, 2 * self.nen, axis=1).ravel()
        cols = np.tile(ed, (1, 2 * self.nen)).ravel()
        self.K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        return self.K

    # ---------- boundary ----------
    def boundary_edges(self):
        """Element edges that belong to only one element, as node tuples in counter-clockwise order."""
        seen = {}
        for e, conn in enumerate(self.T):
            for loc in EDGES[self.etype]:
                nodes = tuple(conn[list(loc)])
                key = (min(nodes[0], nodes[-1]), max(nodes[0], nodes[-1]))
                seen.setdefault(key, []).append(nodes)
        return [v[0] for v in seen.values() if len(v) == 1]

    def edges_where(self, predicate):
        """Boundary edges whose every node satisfies predicate(x, y)."""
        out = []
        for ed in self.boundary_edges():
            xy = self.X[list(ed)]
            if all(predicate(x, y) for x, y in xy):
                out.append(ed)
        return np.array(out, int)

    def add_traction(self, edges, traction):
        """Consistent nodal forces from a traction t(x, y, nx, ny) -> (tx, ty) on boundary edges.

        Integrated with 4-point Gauss along each edge, enough for a quadratic edge carrying a
        smoothly varying traction. The outward normal comes from the counter-clockwise edge order.
        """
        s_pts, s_w = gauss_1d(4)
        for ed in edges:
            xy = self.X[list(ed)]
            for s, w in zip(s_pts, s_w):
                N, dN = edge_shape(len(ed), s)
                p = N @ xy
                tang = dN @ xy
                ds = np.hypot(*tang)
                n = np.array([tang[1], -tang[0]]) / ds
                tx, ty = traction(p[0], p[1], n[0], n[1])
                for k, node in enumerate(ed):
                    self.f[2 * node] += N[k] * tx * ds * w * self.t
                    self.f[2 * node + 1] += N[k] * ty * ds * w * self.t

    def fix(self, nodes, comp, value=0.0):
        """Prescribe displacement component comp (0 = x, 1 = y) at nodes. value may be an array."""
        nodes = np.atleast_1d(nodes)
        vals = np.broadcast_to(value, nodes.shape)
        for nd, v in zip(nodes, vals):
            self.fixed[2 * int(nd) + comp] = float(v)

    # ---------- solve ----------
    def solve(self):
        if not hasattr(self, "K"):
            self.stiffness()
        c = np.fromiter(self.fixed.keys(), int)
        uc = np.fromiter(self.fixed.values(), float)
        free = np.setdiff1d(np.arange(self.ndof), c)
        u = np.zeros(self.ndof)
        u[c] = uc
        rhs = self.f[free] - self.K[free][:, c] @ uc
        u[free] = spla.spsolve(self.K[free][:, free].tocsc(), rhs)
        self.u = u
        self.reactions = self.K @ u - self.f
        return u

    # ---------- post-processing ----------
    def stress_at(self, xi, eta):
        """Stress (ne, 3) at one natural point in every element."""
        B, _ = self._B_at(xi, eta)
        ue = self.u[self._edofs()]
        return np.einsum("ij,ejk,ek->ei", self.D, B, ue)

    def nodal_stress(self):
        """Element stresses evaluated at each element's own nodes, averaged over elements sharing a node."""
        acc = np.zeros((len(self.X), 3))
        cnt = np.zeros(len(self.X))
        for k, (xi, eta) in enumerate(NODE_XI[self.etype]):
            s = self.stress_at(xi, eta)
            np.add.at(acc, self.T[:, k], s)
            np.add.at(cnt, self.T[:, k], 1)
        return acc / cnt[:, None]

    def strain_energy(self):
        return 0.5 * self.u @ (self.K @ self.u)

    def energy_error(self, exact_stress):
        """Relative energy-norm error sqrt(int (s - s_h)' C (s - s_h) / int s' C s) against an exact field.

        exact_stress(x, y) -> (n, 3) array. C = D^-1 is the compliance matrix.
        """
        C = np.linalg.inv(self.D)
        pts, wts = gauss_2d("Q8")  # 3x3 rule for both element types, so the error is integrated accurately
        num = den = 0.0
        ue = self.u[self._edofs()]
        for (xi, eta), w in zip(pts, wts):
            N, _ = shape(self.etype, xi, eta)
            B, detJ = self._B_at(xi, eta)
            sh = np.einsum("ij,ejk,ek->ei", self.D, B, ue)
            xy = np.einsum("n,enb->eb", N, self.X[self.T])
            se = exact_stress(xy[:, 0], xy[:, 1])
            d = se - sh
            num += np.sum(np.einsum("ei,ij,ej->e", d, C, d) * detJ * w)
            den += np.sum(np.einsum("ei,ij,ej->e", se, C, se) * detJ * w)
        return np.sqrt(num / den)


def von_mises(s):
    sxx, syy, sxy = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
