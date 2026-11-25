"""Independent beam analysis (stiffness method) used to draw the shear, moment and
deflection diagrams and to cross-check the Table 3-23 formulas.

Any consistent units work (e.g. kip & in, or N & mm).

Sign conventions (same as the AISC diagrams):
  * loads: downward positive
  * shear V(x): resultant of the upward forces to the left of the section
  * moment M(x): sagging (tension at the bottom) positive
  * deflection: downward positive
  * applied couples: clockwise positive (a clockwise couple raises M to its right)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- model
@dataclass
class Point:
    x: float
    P: float            # downward force


@dataclass
class Dist:
    x1: float
    x2: float
    w1: float           # downward intensity at x1
    w2: float           # downward intensity at x2


@dataclass
class Couple:
    x: float
    M: float            # clockwise moment


@dataclass
class Model:
    length: float
    supports: list[tuple[float, str]]           # (x, "pin" | "fixed" | "guided")
    loads: list = field(default_factory=list)   # Point | Dist | Couple

    def key_points(self) -> list[float]:
        pts = {0.0, self.length}
        pts.update(x for x, _ in self.supports)
        for ld in self.loads:
            if isinstance(ld, Dist):
                pts.update((ld.x1, ld.x2))
            else:
                pts.add(ld.x)
        return sorted(p for p in pts if -1e-12 <= p <= self.length * (1 + 1e-12))


@dataclass
class Reaction:
    x: float
    kind: str
    F: float            # upward force on the beam
    M: float            # counter-clockwise moment on the beam (fixed / guided supports)


# ---------------------------------------------------------------- solution
class Solution:
    def __init__(self, model: Model, EI: float, n_elem: int = 240):
        self.model = model
        self.EI = EI
        L = model.length
        keys = model.key_points()
        nodes = np.union1d(np.linspace(0.0, L, n_elem + 1), keys)
        # drop near-duplicates created by floating point
        keep = [0]
        for i in range(1, len(nodes)):
            if nodes[i] - nodes[keep[-1]] > 1e-9 * L:
                keep.append(i)
        nodes = nodes[keep]
        nodes[-1] = L
        self.nodes = nodes
        self._solve()

    # -- helpers
    def _node(self, x: float) -> int:
        i = int(np.argmin(np.abs(self.nodes - x)))
        if abs(self.nodes[i] - x) > 1e-7 * self.model.length:
            raise ValueError(f"no node at x={x}")
        return i

    def _solve(self):
        n = len(self.nodes)
        K = np.zeros((2 * n, 2 * n))
        F = np.zeros(2 * n)
        EI = self.EI
        for e in range(n - 1):
            h = self.nodes[e + 1] - self.nodes[e]
            k = EI / h**3 * np.array([[12, 6 * h, -12, 6 * h],
                                      [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                      [-12, -6 * h, 12, -6 * h],
                                      [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(idx, idx)] += k

        for ld in self.model.loads:
            if isinstance(ld, Point):
                F[2 * self._node(ld.x)] -= ld.P
            elif isinstance(ld, Couple):
                F[2 * self._node(ld.x) + 1] -= ld.M
            elif isinstance(ld, Dist):
                for e in range(n - 1):
                    xa, xb = self.nodes[e], self.nodes[e + 1]
                    if xa < ld.x1 - 1e-12 or xb > ld.x2 + 1e-12:
                        continue
                    qa, qb = -self._q(ld, xa), -self._q(ld, xb)       # upward intensity
                    h = xb - xa
                    F[2 * e] += h * (7 * qa + 3 * qb) / 20
                    F[2 * e + 1] += h * h * (3 * qa + 2 * qb) / 60
                    F[2 * e + 2] += h * (3 * qa + 7 * qb) / 20
                    F[2 * e + 3] -= h * h * (2 * qa + 3 * qb) / 60

        fixed_dofs = []
        for x, kind in self.model.supports:
            i = self._node(x)
            if kind in ("pin", "fixed"):
                fixed_dofs.append(2 * i)
            if kind in ("fixed", "guided"):
                fixed_dofs.append(2 * i + 1)
        free = np.setdiff1d(np.arange(2 * n), fixed_dofs)
        u = np.zeros(2 * n)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        r = K @ u - F
        self.u = u
        self.reactions = []
        for x, kind in self.model.supports:
            i = self._node(x)
            Fy = r[2 * i] if kind in ("pin", "fixed") else 0.0
            Mz = r[2 * i + 1] if kind in ("fixed", "guided") else 0.0
            self.reactions.append(Reaction(x, kind, Fy, Mz))

    @staticmethod
    def _q(ld: Dist, s: float) -> float:
        if ld.x2 == ld.x1:
            return ld.w1
        return ld.w1 + (ld.w2 - ld.w1) * (s - ld.x1) / (ld.x2 - ld.x1)

    def _left(self, xi: float, x: float, side: int) -> bool:
        tol = 1e-9 * self.model.length
        return xi < x - tol or (abs(xi - x) <= tol and side > 0)

    # -- results
    def shear(self, x: float, side: int = 1) -> float:
        """Shear just left (side=-1) or right (side=+1) of x."""
        V = 0.0
        for rc in self.reactions:
            if self._left(rc.x, x, side):
                V += rc.F
        for ld in self.model.loads:
            if isinstance(ld, Point) and self._left(ld.x, x, side):
                V -= ld.P
            elif isinstance(ld, Dist) and x > ld.x1:
                xe = min(x, ld.x2)
                V -= 0.5 * (ld.w1 + self._q(ld, xe)) * (xe - ld.x1)
        return V

    def moment(self, x: float, side: int = 1) -> float:
        """Bending moment (sagging +) just left or right of x."""
        M = 0.0
        for rc in self.reactions:
            if self._left(rc.x, x, side):
                M += rc.F * (x - rc.x) - rc.M
        for ld in self.model.loads:
            if isinstance(ld, Point) and self._left(ld.x, x, side):
                M -= ld.P * (x - ld.x)
            elif isinstance(ld, Couple) and self._left(ld.x, x, side):
                M += ld.M
            elif isinstance(ld, Dist) and x > ld.x1:
                xe = min(x, ld.x2)
                s = np.array([ld.x1, 0.5 * (ld.x1 + xe), xe])
                f = np.array([self._q(ld, si) * (x - si) for si in s])
                M -= (xe - ld.x1) / 6 * (f[0] + 4 * f[1] + f[2])       # Simpson: exact here
        return M

    def deflection(self, x: float) -> float:
        """Deflection, downward positive."""
        nodes = self.nodes
        e = int(np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2))
        h = nodes[e + 1] - nodes[e]
        t = (x - nodes[e]) / h
        v1, r1, v2, r2 = self.u[2 * e: 2 * e + 4]
        H = (1 - 3 * t * t + 2 * t**3, h * (t - 2 * t * t + t**3),
             3 * t * t - 2 * t**3, h * (t**3 - t * t))
        return -(H[0] * v1 + H[1] * r1 + H[2] * v2 + H[3] * r2)

    # -- sampled diagrams
    def diagrams(self, n: int = 400):
        """Return dict of x/V/M/D arrays for plotting (jumps shown as vertical steps)."""
        L = self.model.length
        xs = np.union1d(np.linspace(0, L, n + 1), self.model.key_points())
        xv, V, M = [], [], []
        for x in xs:
            for side in (-1, 1):
                if (x == 0 and side < 0) or (x == L and side > 0):
                    continue
                xv.append(x)
                V.append(self.shear(x, side))
                M.append(self.moment(x, side))
        # close the diagrams to zero at the ends where the value jumps from 0
        xv = [0.0] + xv + [L]
        V = [0.0] + V + [0.0]
        M = [0.0] + M + [0.0]
        xd = np.union1d(np.linspace(0, L, 4 * n + 1), self.model.key_points())
        D = [self.deflection(x) for x in xd]

        def clean(a):          # remove floating-point noise (e.g. 1e-10 at a pinned end)
            a = np.array(a, dtype=float)
            a[np.abs(a) < 1e-7 * (np.abs(a).max() or 1.0)] = 0.0
            return a

        return {"xv": np.array(xv), "V": clean(V), "M": clean(M), "xd": np.array(xd), "D": clean(D)}

    def extremes(self):
        d = self.diagrams()

        def ext(xs, ys):
            i_max, i_min = int(np.argmax(ys)), int(np.argmin(ys))
            return (ys[i_max], xs[i_max]), (ys[i_min], xs[i_min])

        return {"V": ext(d["xv"][1:-1], d["V"][1:-1]),
                "M": ext(d["xv"][1:-1], d["M"][1:-1]),
                "D": ext(d["xd"], d["D"])}
