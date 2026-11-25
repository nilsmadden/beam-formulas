"""Check every Table 3-23 formula in beams.py against the independent stiffness analysis.

Run:  .venv\\Scripts\\python.exe -m pytest -q
"""

import itertools

import numpy as np
import pytest

from beams import CASES, PARAMS
from solver import Model, Point, Solution

E, I = 29000.0, 800.0          # ksi, in^4
EI = E * I
TO_IN = {"length": 12.0, "force": 1.0, "dist": 1 / 12.0, "moment": 12.0}   # ft/kip -> kip/in


def base_params(c, scale=None):
    p = {}
    for k in c.params:
        imp = c.defaults.get(k, (PARAMS[k].default_imp, None))[0]
        p[k] = imp * TO_IN[PARAMS[k].kind]
    p["EI"] = EI
    if scale:
        for k, f in scale.items():
            if k in p:
                p[k] *= f
    return p


# a second geometry per case to exercise the other branches (a > b, etc.)
VARIANTS = [None, {"a": 2.0, "b": 0.5, "c": 1.5, "M2": 0.2, "P2": 0.5, "w2": 3.0}]


def cases_and_params():
    for num, c in CASES.items():
        for v in VARIANTS:
            p = base_params(c, v)
            if c.validate(p):
                continue
            yield pytest.param(num, p, id=f"case{num}-{'base' if v is None else 'alt'}")


def x_samples(c, p):
    l = p["l"]
    xs = [None] if c.x_name is None else [0.13 * l, 0.31 * l, 0.47 * l, 0.62 * l, 0.88 * l]
    x1s = [None]
    if c.has_x1:
        oh = p.get("loh", p.get("a"))
        x1s = [0.3 * oh, 0.8 * oh]
    return list(itertools.product(xs, x1s))


def check(sol, model, res, p):
    chk = res.chk
    kind = chk[0]
    v = res.value
    if kind == "R":
        i = chk[1]
        tol = chk[2] if len(chk) > 2 else 1e-6
        ref = sol.reactions[i].F
        assert abs(abs(v) - abs(ref)) <= tol * max(abs(ref), 1e-9) + 1e-9, (res.name, v, ref)
        return
    if kind in ("V", "Vneg"):
        ref = sol.shear(chk[1], chk[2] if len(chk) > 2 else 1)
        assert abs(abs(v) - abs(ref)) <= 1e-6 * max(abs(ref), 1) , (res.name, v, ref)
        return
    if kind in ("M", "Msigned"):
        if kind == "M":
            side = chk[2] if len(chk) > 2 else 1
            ref = sol.moment(chk[1], side)
            assert abs(abs(v) - abs(ref)) <= 1e-6 * max(abs(ref), 1), (res.name, v, ref)
        else:
            ref = sol.moment(chk[1], 1)
            assert abs(v - ref) <= chk[2] * abs(v) + 1e-6, (res.name, v, ref)
        return
    if kind == "Mzero":
        assert abs(sol.moment(chk[1])) <= 1e-6 * p["l"] * p.get("w", 1) * p["l"], res.name
        return
    if kind in ("D", "Dsigned"):
        tol = chk[2] if len(chk) > 2 else 1e-5
        ref = sol.deflection(chk[1])
        if kind == "Dsigned":
            assert abs(v - ref) <= 1e-5 * max(abs(ref), 1e-6), (res.name, v, ref)
        else:
            assert abs(abs(v) - abs(ref)) <= tol * abs(ref) + 1e-9, (res.name, v, ref)
        return
    if kind in ("Rmove", "Mmove"):
        loads = [ld for ld in model.loads if isinstance(ld, Point)]
        # rebuild the moving train from the case parameters
        l = p["l"]
        if "P1" in p:
            train = [(0.0, p["P1"]), (p["a"], p["P2"])]
        elif "a" in p:
            train = [(0.0, p["P"]), (p["a"], p["P"])]
        else:
            train = [(0.0, p["P"])]
        best_R, best_M = 0.0, 0.0
        span = sum(abs(t[0]) for t in train)
        for s in np.linspace(-span, l, 4001):
            pts = [(s + d, P) for d, P in train if 0 <= s + d <= l]
            R1 = sum(P * (l - xx) / l for xx, P in pts)
            best_R = max(best_R, R1)
            for xx, _ in pts:
                M = R1 * xx - sum(P * (xx - xj) for xj, P in pts if xj < xx)
                best_M = max(best_M, M)
        ref = best_R if kind == "Rmove" else best_M
        assert abs(v - ref) <= 2e-4 * ref, (res.name, v, ref)
        return
    raise AssertionError(f"unknown check {chk}")


@pytest.mark.parametrize("num,p", list(cases_and_params()))
def test_case(num, p):
    c = CASES[num]
    model = c.model(p)
    sol = Solution(model, EI)
    # equilibrium of the analysis itself
    total = sum(r.F for r in sol.reactions)
    applied = sum(ld.P for ld in model.loads if isinstance(ld, Point)) + sum(
        0.5 * (ld.w1 + ld.w2) * (ld.x2 - ld.x1) for ld in model.loads if hasattr(ld, "w1"))
    assert abs(total - applied) <= 1e-6 * max(applied, 1.0)

    n_checked = 0
    for x, x1 in x_samples(c, p):
        for res in c.results(p, x, x1):
            assert np.isfinite(res.value), res.name
            if res.chk:
                check(sol, model, res, p)
                n_checked += 1
    assert n_checked > 0


def test_all_45_cases_present():
    assert sorted(CASES) == list(range(1, 46))
