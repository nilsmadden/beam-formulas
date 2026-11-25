"""AISC Steel Construction Manual, Table 3-23: Shears, Moments and Deflections.

The science only, no web code. Every formula is evaluated in whatever consistent
units the caller supplies (app.py uses kip & inch, or N & mm).

Each case provides:
  * params   - the inputs it needs (keys of PARAMS)
  * validate - geometry checks, returns an error message or None
  * model    - the beam for the independent stiffness analysis (solver.py)
  * results  - the Table 3-23 formulas (LaTeX) and their values

`python -m pytest` checks every formula against the stiffness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable

from solver import Couple, Dist, Model, Point

# result kinds (control the display units)
FORCE, MOMENT, DEFL, POS, COEF = "force", "moment", "defl", "pos", "coef"


@dataclass
class Param:
    symbol: str          # LaTeX
    label: str
    kind: str            # "length" | "force" | "dist" | "moment"
    default_imp: float   # ft, kips, kip/ft, kip-ft
    default_si: float    # m, kN, kN/m, kN-m


PARAMS = {
    "l":   Param("l", "Span length, l", "length", 20.0, 6.0),
    "a":   Param("a", "Distance, a", "length", 6.0, 2.0),
    "b":   Param("b", "Distance, b", "length", 6.0, 2.0),
    "c":   Param("c", "Distance, c", "length", 6.0, 2.0),
    "loh": Param("l_{oh}", "Overhang length", "length", 5.0, 1.5),
    "w":   Param("w", "Uniform load, w", "dist", 2.0, 30.0),
    "w1":  Param("w_1", "Uniform load at R₁, w₁", "dist", 2.0, 30.0),
    "w2":  Param("w_2", "Uniform load at R₂, w₂", "dist", 1.0, 15.0),
    "W":   Param("W", "Total load, W", "force", 20.0, 90.0),
    "P":   Param("P", "Concentrated load, P", "force", 10.0, 45.0),
    "P1":  Param("P_1", "Concentrated load, P₁", "force", 10.0, 45.0),
    "P2":  Param("P_2", "Concentrated load, P₂", "force", 6.0, 25.0),
    "M":   Param("M", "Concentrated moment, M", "moment", 20.0, 30.0),
    "M1":  Param("M_1", "Left end moment (hogging), M₁", "moment", 60.0, 80.0),
    "M2":  Param("M_2", "Right end moment (hogging), M₂", "moment", 30.0, 40.0),
}


@dataclass
class Result:
    name: str                 # LaTeX, e.g. "R_1 = V_1"
    where: str                # plain-text condition / location
    formula: str              # LaTeX right-hand side
    value: float
    kind: str
    chk: tuple | None = None  # what the stiffness analysis should reproduce
    at_x: bool = False        # evaluated at the user's x (or x1)


class Results(list):
    def add(self, name, where, formula, value, kind, chk=None, at_x=False):
        self.append(Result(name, where, formula, value, kind, chk, at_x))


@dataclass
class Case:
    num: int
    title: str
    category: str
    params: list[str]
    model: Callable[[dict], Model]
    results: Callable[[dict, float | None, float | None], Results]
    validate: Callable[[dict], str | None] = lambda p: None
    derived: Callable[[dict], list[tuple[str, float, str]]] = lambda p: []
    x_name: str | None = "x"          # symbol of the position variable (None = no x formulas)
    x_note: str = "measured from the left end"
    has_x1: bool = False               # overhang cases: x1 measured from R2
    defaults: dict = field(default_factory=dict)   # per-case defaults {key: (imp, si)}
    note: str = ""
    moving: bool = False
    labels: dict = field(default_factory=dict)  # per-case input labels {key: text}

    def param_label(self, k: str) -> str:
        return self.labels.get(k, PARAMS[k].label)

    @property
    def label(self) -> str:
        return f"{self.num}. {self.title}"


CASES: dict[int, Case] = {}


def case(**kw):
    def deco(fn):
        c = Case(results=fn, **kw)
        CASES[c.num] = c
        return fn
    return deco


def _lt(x, v):
    return x is not None and x <= v


def _gt(x, v):
    return x is not None and x >= v


def _between(x, lo, hi):
    return x is not None and lo <= x <= hi


def need(cond: bool, msg: str):
    return None if cond else msg


TEUL = r"\text{Total equiv. uniform load}"

# =====================================================================================
# 1-11  SIMPLE BEAMS
# =====================================================================================


@case(num=1, title="Simple beam — uniformly distributed load", category="Simple beams",
      params=["l", "w"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Dist(0, p["l"], p["w"], p["w"])]))
def _c1(p, x, x1):
    l, w = p["l"], p["w"]
    r = Results()
    r.add(TEUL, "", r"wl", w * l, FORCE)
    r.add("R = V", "", r"\frac{wl}{2}", w * l / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at center", r"\frac{wl^2}{8}", w * l**2 / 8, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{5wl^4}{384EI}", 5 * w * l**4 / (384 * p["EI"]), DEFL, ("D", l / 2))
    if x is not None:
        r.add("V_x", "", r"w\left(\frac{l}{2}-x\right)", w * (l / 2 - x), FORCE, ("V", x), True)
        r.add("M_x", "", r"\frac{wx}{2}(l-x)", w * x / 2 * (l - x), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{wx}{24EI}\left(l^3-2lx^2+x^3\right)",
              w * x / (24 * p["EI"]) * (l**3 - 2 * l * x**2 + x**3), DEFL, ("D", x), True)
    return r


@case(num=2, title="Simple beam — load increasing uniformly to one end", category="Simple beams",
      params=["l", "W"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Dist(0, p["l"], 0, 2 * p["W"] / p["l"])]),
      note="Triangular load, zero at R₁ and maximum at R₂; W is the total load.")
def _c2(p, x, x1):
    l, W, EI = p["l"], p["W"], p["EI"]
    xm = l / sqrt(3)
    xd = l * sqrt(1 - sqrt(8 / 15))
    r = Results()
    r.add(TEUL, "", r"\frac{16W}{9\sqrt{3}} = 1.03W", 16 * W / (9 * sqrt(3)), FORCE)
    r.add("R_1 = V_1", "", r"\frac{W}{3}", W / 3, FORCE, ("R", 0))
    r.add("R_2 = V_2 = V_{max}", "", r"\frac{2W}{3}", 2 * W / 3, FORCE, ("R", 1))
    r.add(r"M_{max}", r"at x = l/√3 = 0.577 l", r"\frac{2Wl}{9\sqrt{3}} = 0.128\,Wl",
          2 * W * l / (9 * sqrt(3)), MOMENT, ("M", xm))
    r.add(r"\Delta_{max}", "at x = l√(1 − √(8/15)) = 0.519 l", r"0.0130\,\frac{Wl^3}{EI}",
          W * xd / (180 * EI * l**2) * (3 * xd**4 - 10 * l**2 * xd**2 + 7 * l**4), DEFL, ("D", xd))
    if x is not None:
        r.add("V_x", "", r"\frac{W}{3}-\frac{Wx^2}{l^2}", W / 3 - W * x**2 / l**2, FORCE, ("V", x), True)
        r.add("M_x", "", r"\frac{Wx}{3l^2}\left(l^2-x^2\right)", W * x / (3 * l**2) * (l**2 - x**2), MOMENT,
              ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{Wx}{180EIl^2}\left(3x^4-10l^2x^2+7l^4\right)",
              W * x / (180 * EI * l**2) * (3 * x**4 - 10 * l**2 * x**2 + 7 * l**4), DEFL, ("D", x), True)
    return r


@case(num=3, title="Simple beam — load increasing uniformly to center", category="Simple beams",
      params=["l", "W"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Dist(0, p["l"] / 2, 0, 2 * p["W"] / p["l"]),
                             Dist(p["l"] / 2, p["l"], 2 * p["W"] / p["l"], 0)]),
      note="Triangular load, maximum at center; W is the total load.")
def _c3(p, x, x1):
    l, W, EI = p["l"], p["W"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{4W}{3}", 4 * W / 3, FORCE)
    r.add("R = V", "", r"\frac{W}{2}", W / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at center", r"\frac{Wl}{6}", W * l / 6, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{Wl^3}{60EI}", W * l**3 / (60 * EI), DEFL, ("D", l / 2))
    if _lt(x, l / 2):
        r.add("V_x", "when x < l/2", r"\frac{W}{2l^2}\left(l^2-4x^2\right)", W / (2 * l**2) * (l**2 - 4 * x**2),
              FORCE, ("V", x), True)
        r.add("M_x", "when x < l/2", r"Wx\left(\frac{1}{2}-\frac{2x^2}{3l^2}\right)",
              W * x * (0.5 - 2 * x**2 / (3 * l**2)), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2", r"\frac{Wx}{480EIl^2}\left(5l^2-4x^2\right)^2",
              W * x / (480 * EI * l**2) * (5 * l**2 - 4 * x**2) ** 2, DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Distance from R₁ to load, a', 'b': 'Loaded length, b'}, num=4, title="Simple beam — uniform load partially distributed", category="Simple beams",
      params=["l", "w", "a", "b"],
      validate=lambda p: need(p["a"] + p["b"] <= p["l"] * (1 + 1e-9), "a + b must not exceed l."),
      derived=lambda p: [("c = l - a - b", p["l"] - p["a"] - p["b"], "length")],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Dist(p["a"], p["a"] + p["b"], p["w"], p["w"])]),
      note="Load w over length b, starting a from R₁.")
def _c4(p, x, x1):
    l, w, a, b = p["l"], p["w"], p["a"], p["b"]
    c = l - a - b
    R1 = w * b / (2 * l) * (2 * c + b)
    R2 = w * b / (2 * l) * (2 * a + b)
    r = Results()
    r.add("R_1 = V_1", "max. when a < c", r"\frac{wb}{2l}(2c+b)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "max. when a > c", r"\frac{wb}{2l}(2a+b)", R2, FORCE, ("R", 1))
    r.add(r"M_{max}", "at x = a + R₁/w", r"R_1\left(a+\frac{R_1}{2w}\right)", R1 * (a + R1 / (2 * w)), MOMENT,
          ("M", a + R1 / w))
    if _between(x, a, a + b):
        r.add("V_x", "when a < x < a + b", r"R_1 - w(x-a)", R1 - w * (x - a), FORCE, ("V", x), True)
    if _lt(x, a):
        r.add("M_x", "when x < a", r"R_1 x", R1 * x, MOMENT, ("M", x), True)
    if _between(x, a, a + b):
        r.add("M_x", "when a < x < a + b", r"R_1x-\frac{w}{2}(x-a)^2", R1 * x - w / 2 * (x - a) ** 2, MOMENT,
              ("M", x), True)
    if _gt(x, a + b):
        r.add("M_x", "when x > a + b", r"R_2(l-x)", R2 * (l - x), MOMENT, ("M", x), True)
    return r


@case(labels={'a': 'Loaded length from R₁, a'}, num=5, title="Simple beam — uniform load partially distributed at one end", category="Simple beams",
      params=["l", "w", "a"],
      validate=lambda p: need(p["a"] <= p["l"], "a must not exceed l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Dist(0, p["a"], p["w"], p["w"])]),
      note="Load w from R₁ over length a.")
def _c5(p, x, x1):
    l, w, a, EI = p["l"], p["w"], p["a"], p["EI"]
    R1 = w * a / (2 * l) * (2 * l - a)
    R2 = w * a**2 / (2 * l)
    r = Results()
    r.add("R_1 = V_1 = V_{max}", "", r"\frac{wa}{2l}(2l-a)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{wa^2}{2l}", R2, FORCE, ("R", 1))
    r.add(r"M_{max}", "at x = R₁/w", r"\frac{R_1^2}{2w}", R1**2 / (2 * w), MOMENT, ("M", R1 / w))
    if _lt(x, a):
        r.add("V_x", "when x < a", r"R_1 - wx", R1 - w * x, FORCE, ("V", x), True)
        r.add("M_x", "when x < a", r"R_1x-\frac{wx^2}{2}", R1 * x - w * x**2 / 2, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a",
              r"\frac{wx}{24EIl}\left(a^2(2l-a)^2-2ax^2(2l-a)+lx^3\right)",
              w * x / (24 * EI * l) * (a**2 * (2 * l - a) ** 2 - 2 * a * x**2 * (2 * l - a) + l * x**3),
              DEFL, ("D", x), True)
    if _gt(x, a):
        r.add("M_x", "when x > a", r"R_2(l-x)", R2 * (l - x), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x > a", r"\frac{wa^2(l-x)}{24EIl}\left(4xl-2x^2-a^2\right)",
              w * a**2 * (l - x) / (24 * EI * l) * (4 * x * l - 2 * x**2 - a**2), DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Loaded length at R₁, a', 'c': 'Loaded length at R₂, c'}, num=6, title="Simple beam — uniform load partially distributed at each end", category="Simple beams",
      params=["l", "w1", "a", "w2", "c"],
      validate=lambda p: need(p["a"] + p["c"] <= p["l"] * (1 + 1e-9), "a + c must not exceed l."),
      derived=lambda p: [("b = l - a - c", p["l"] - p["a"] - p["c"], "length")],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Dist(0, p["a"], p["w1"], p["w1"]), Dist(p["l"] - p["c"], p["l"], p["w2"], p["w2"])]),
      note="Load w₁ over length a at R₁, and w₂ over length c at R₂.")
def _c6(p, x, x1):
    l, w1, a, w2, c = p["l"], p["w1"], p["a"], p["w2"], p["c"]
    b = l - a - c
    R1 = (w1 * a * (2 * l - a) + w2 * c**2) / (2 * l)
    R2 = (w2 * c * (2 * l - c) + w1 * a**2) / (2 * l)
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{w_1a(2l-a)+w_2c^2}{2l}", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{w_2c(2l-c)+w_1a^2}{2l}", R2, FORCE, ("R", 1))
    if R1 < w1 * a:
        r.add(r"M_{max}", "at x = R₁/w₁ (R₁ < w₁a)", r"\frac{R_1^2}{2w_1}", R1**2 / (2 * w1), MOMENT,
              ("M", R1 / w1))
    if R2 < w2 * c:
        r.add(r"M_{max}", "at x = l − R₂/w₂ (R₂ < w₂c)", r"\frac{R_2^2}{2w_2}", R2**2 / (2 * w2), MOMENT,
              ("M", l - R2 / w2))
    if _lt(x, a):
        r.add("V_x", "when x < a", r"R_1 - w_1x", R1 - w1 * x, FORCE, ("V", x), True)
        r.add("M_x", "when x < a", r"R_1x-\frac{w_1x^2}{2}", R1 * x - w1 * x**2 / 2, MOMENT, ("M", x), True)
    if _between(x, a, a + b):
        r.add("V_x", "when a < x < a + b", r"R_1 - w_1a", R1 - w1 * a, FORCE, ("V", x), True)
        r.add("M_x", "when a < x < a + b", r"R_1x-\frac{w_1a}{2}(2x-a)", R1 * x - w1 * a / 2 * (2 * x - a),
              MOMENT, ("M", x), True)
    if _gt(x, a + b):
        r.add("V_x", "when x > a + b", r"R_2 - w_2(l-x)", R2 - w2 * (l - x), FORCE, ("Vneg", x), True)
        r.add("M_x", "when x > a + b", r"R_2(l-x)-\frac{w_2(l-x)^2}{2}", R2 * (l - x) - w2 * (l - x) ** 2 / 2,
              MOMENT, ("M", x), True)
    return r


@case(num=7, title="Simple beam — concentrated load at center", category="Simple beams",
      params=["l", "P"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Point(p["l"] / 2, p["P"])]))
def _c7(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", "2P", 2 * P, FORCE)
    r.add("R = V", "", r"\frac{P}{2}", P / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at point of load", r"\frac{Pl}{4}", P * l / 4, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at point of load", r"\frac{Pl^3}{48EI}", P * l**3 / (48 * EI), DEFL, ("D", l / 2))
    if _lt(x, l / 2):
        r.add("M_x", "when x < l/2", r"\frac{Px}{2}", P * x / 2, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2", r"\frac{Px}{48EI}\left(3l^2-4x^2\right)",
              P * x / (48 * EI) * (3 * l**2 - 4 * x**2), DEFL, ("D", x), True)
    return r


def _c8_core(p, x, r):
    """Formulas shared by case 8 and case 28 (load between supports)."""
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    b = l - a
    r.add(TEUL, "", r"\frac{8Pab}{l^2}", 8 * P * a * b / l**2, FORCE)
    r.add("R_1 = V_1", "= V_max when a < b", r"\frac{Pb}{l}", P * b / l, FORCE, ("R", 0))
    r.add("R_2 = V_2", "= V_max when a > b", r"\frac{Pa}{l}", P * a / l, FORCE, ("R", 1))
    r.add(r"M_{max}", "at point of load", r"\frac{Pab}{l}", P * a * b / l, MOMENT, ("M", a))
    if a >= b:
        xm = sqrt(a * (a + 2 * b) / 3)
        r.add(r"\Delta_{max}", f"at x = √(a(a+2b)/3), when a > b",
              r"\frac{Pab(a+2b)\sqrt{3a(a+2b)}}{27EIl}",
              P * a * b * (a + 2 * b) * sqrt(3 * a * (a + 2 * b)) / (27 * EI * l), DEFL, ("D", xm))
    else:   # by symmetry: swap a and b, measure from R2
        xm = l - sqrt(b * (b + 2 * a) / 3)
        r.add(r"\Delta_{max}", "a < b: a and b interchanged, x measured from R₂",
              r"\frac{Pba(b+2a)\sqrt{3b(b+2a)}}{27EIl}",
              P * a * b * (b + 2 * a) * sqrt(3 * b * (b + 2 * a)) / (27 * EI * l), DEFL, ("D", xm))
    r.add(r"\Delta_a", "at point of load", r"\frac{Pa^2b^2}{3EIl}", P * a**2 * b**2 / (3 * EI * l), DEFL, ("D", a))
    if _lt(x, a):
        r.add("M_x", "when x < a", r"\frac{Pbx}{l}", P * b * x / l, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a", r"\frac{Pbx}{6EIl}\left(l^2-b^2-x^2\right)",
              P * b * x / (6 * EI * l) * (l**2 - b**2 - x**2), DEFL, ("D", x), True)
    return b


@case(labels={'a': 'Load position from R₁, a'}, num=8, title="Simple beam — concentrated load at any point", category="Simple beams",
      params=["l", "P", "a"],
      validate=lambda p: need(0 < p["a"] < p["l"], "a must be between 0 and l."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Point(p["a"], p["P"])]),
      note="Load P at distance a from R₁.")
def _c8(p, x, x1):
    r = Results()
    _c8_core(p, x, r)
    return r


@case(labels={'a': 'Load distance from each support, a'}, num=9, title="Simple beam — two equal concentrated loads symmetrically placed", category="Simple beams",
      params=["l", "P", "a"],
      validate=lambda p: need(0 < p["a"] <= p["l"] / 2, "a must be between 0 and l/2."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Point(p["a"], p["P"]), Point(p["l"] - p["a"], p["P"])]),
      note="Two loads P, each a from its nearest support.")
def _c9(p, x, x1):
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{8Pa}{l}", 8 * P * a / l, FORCE)
    r.add("R = V", "", "P", P, FORCE, ("R", 0))
    r.add(r"M_{max}", "between loads", "Pa", P * a, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{Pa}{24EI}\left(3l^2-4a^2\right)",
          P * a / (24 * EI) * (3 * l**2 - 4 * a**2), DEFL, ("D", l / 2))
    if _lt(x, a):
        r.add("M_x", "when x < a", "Px", P * x, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a", r"\frac{Px}{6EI}\left(3la-3a^2-x^2\right)",
              P * x / (6 * EI) * (3 * l * a - 3 * a**2 - x**2), DEFL, ("D", x), True)
    if _between(x, a, l - a):
        r.add(r"\Delta_x", "when a < x < l − a", r"\frac{Pa}{6EI}\left(3lx-3x^2-a^2\right)",
              P * a / (6 * EI) * (3 * l * x - 3 * x**2 - a**2), DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Load 1 distance from R₁, a', 'b': 'Load 2 distance from R₂, b'}, num=10, title="Simple beam — two equal concentrated loads unsymmetrically placed", category="Simple beams",
      params=["l", "P", "a", "b"],
      validate=lambda p: need(p["a"] + p["b"] <= p["l"], "a + b must not exceed l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Point(p["a"], p["P"]), Point(p["l"] - p["b"], p["P"])]),
      note="Loads P at a from R₁ and at b from R₂.")
def _c10(p, x, x1):
    l, P, a, b = p["l"], p["P"], p["a"], p["b"]
    R1 = P / l * (l - a + b)
    R2 = P / l * (l - b + a)
    r = Results()
    r.add("R_1 = V_1", "= V_max when a < b", r"\frac{P}{l}(l-a+b)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "= V_max when a > b", r"\frac{P}{l}(l-b+a)", R2, FORCE, ("R", 1))
    r.add("V_x", "when a < x < l − b", r"\frac{P}{l}(b-a)", P / l * (b - a), FORCE, ("V", (a + l - b) / 2))
    r.add("M_1", "= M_max when a > b", r"R_1a", R1 * a, MOMENT, ("M", a))
    r.add("M_2", "= M_max when a < b", r"R_2b", R2 * b, MOMENT, ("M", l - b))
    if _lt(x, a):
        r.add("M_x", "when x < a", r"R_1x", R1 * x, MOMENT, ("M", x), True)
    if _between(x, a, l - b):
        r.add("M_x", "when a < x < l − b", r"R_1x-P(x-a)", R1 * x - P * (x - a), MOMENT, ("M", x), True)
    return r


@case(labels={'a': 'P₁ distance from R₁, a', 'b': 'P₂ distance from R₂, b'}, num=11, title="Simple beam — two unequal concentrated loads unsymmetrically placed",
      category="Simple beams",
      params=["l", "P1", "a", "P2", "b"],
      validate=lambda p: need(p["a"] + p["b"] <= p["l"], "a + b must not exceed l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Point(p["a"], p["P1"]), Point(p["l"] - p["b"], p["P2"])]),
      note="P₁ at a from R₁; P₂ at b from R₂.")
def _c11(p, x, x1):
    l, P1, P2, a, b = p["l"], p["P1"], p["P2"], p["a"], p["b"]
    R1 = (P1 * (l - a) + P2 * b) / l
    R2 = (P1 * a + P2 * (l - b)) / l
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{P_1(l-a)+P_2b}{l}", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{P_1a+P_2(l-b)}{l}", R2, FORCE, ("R", 1))
    r.add("V_x", "when a < x < l − b", r"R_1-P_1", R1 - P1, FORCE, ("V", (a + l - b) / 2))
    r.add("M_1", "= M_max when R₁ < P₁", r"R_1a", R1 * a, MOMENT, ("M", a))
    r.add("M_2", "= M_max when R₂ < P₂", r"R_2b", R2 * b, MOMENT, ("M", l - b))
    if _lt(x, a):
        r.add("M_x", "when x < a", r"R_1x", R1 * x, MOMENT, ("M", x), True)
    if _between(x, a, l - b):
        r.add("M_x", "when a < x < l − b", r"R_1x-P_1(x-a)", R1 * x - P1 * (x - a), MOMENT, ("M", x), True)
    return r


# =====================================================================================
# 12-17  FIXED / PROPPED BEAMS
# =====================================================================================

def _propped(p):
    return [(0, "pin"), (p["l"], "fixed")]


@case(num=12, title="Beam fixed at one end, supported at other — uniformly distributed load",
      category="Fixed and propped beams", params=["l", "w"],
      model=lambda p: Model(p["l"], _propped(p), [Dist(0, p["l"], p["w"], p["w"])]),
      note="Simply supported at R₁ (left), fixed at R₂ (right).")
def _c12(p, x, x1):
    l, w, EI = p["l"], p["w"], p["EI"]
    R1 = 3 * w * l / 8
    xd = l / 16 * (1 + sqrt(33))
    r = Results()
    r.add(TEUL, "", "wl", w * l, FORCE)
    r.add("R_1 = V_1", "", r"\frac{3wl}{8}", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2 = V_{max}", "", r"\frac{5wl}{8}", 5 * w * l / 8, FORCE, ("R", 1))
    r.add(r"M_{max}", "at fixed end", r"\frac{wl^2}{8}", w * l**2 / 8, MOMENT, ("M", l, -1))
    r.add("M_1", "at x = 3l/8", r"\frac{9}{128}wl^2", 9 * w * l**2 / 128, MOMENT, ("M", 3 * l / 8))
    r.add(r"\Delta_{max}", "at x = (l/16)(1 + √33) = 0.4215 l", r"\frac{wl^4}{185EI}",
          w * l**4 / (185 * EI), DEFL, ("D", xd, 0.002))
    if x is not None:
        r.add("V_x", "", r"R_1 - wx", R1 - w * x, FORCE, ("V", x), True)
        r.add("M_x", "", r"R_1x-\frac{wx^2}{2}", R1 * x - w * x**2 / 2, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{wx}{48EI}\left(l^3-3lx^2+2x^3\right)",
              w * x / (48 * EI) * (l**3 - 3 * l * x**2 + 2 * x**3), DEFL, ("D", x), True)
    return r


@case(num=13, title="Beam fixed at one end, supported at other — concentrated load at center",
      category="Fixed and propped beams", params=["l", "P"],
      model=lambda p: Model(p["l"], _propped(p), [Point(p["l"] / 2, p["P"])]),
      note="Simply supported at R₁ (left), fixed at R₂ (right).")
def _c13(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{3P}{2}", 3 * P / 2, FORCE)
    r.add("R_1 = V_1", "", r"\frac{5P}{16}", 5 * P / 16, FORCE, ("R", 0))
    r.add("R_2 = V_2 = V_{max}", "", r"\frac{11P}{16}", 11 * P / 16, FORCE, ("R", 1))
    r.add(r"M_{max}", "at fixed end", r"\frac{3Pl}{16}", 3 * P * l / 16, MOMENT, ("M", l, -1))
    r.add("M_1", "at point of load", r"\frac{5Pl}{32}", 5 * P * l / 32, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at x = l/√5 = 0.447 l", r"\frac{Pl^3}{48EI\sqrt{5}} = 0.00932\,\frac{Pl^3}{EI}",
          P * l**3 / (48 * EI * sqrt(5)), DEFL, ("D", l / sqrt(5)))
    r.add(r"\Delta_x", "at point of load", r"\frac{7Pl^3}{768EI}", 7 * P * l**3 / (768 * EI), DEFL, ("D", l / 2))
    if _lt(x, l / 2):
        r.add("M_x", "when x < l/2", r"\frac{5Px}{16}", 5 * P * x / 16, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2", r"\frac{Px}{96EI}\left(3l^2-5x^2\right)",
              P * x / (96 * EI) * (3 * l**2 - 5 * x**2), DEFL, ("D", x), True)
    if _gt(x, l / 2):
        r.add("M_x", "when x > l/2", r"P\left(\frac{l}{2}-\frac{11x}{16}\right)", P * (l / 2 - 11 * x / 16),
              MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x > l/2", r"\frac{P}{96EI}(x-l)^2(11x-2l)",
              P / (96 * EI) * (x - l) ** 2 * (11 * x - 2 * l), DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Load position from R₁, a'}, num=14, title="Beam fixed at one end, supported at the other — concentrated load at any point",
      category="Fixed and propped beams", params=["l", "P", "a"],
      validate=lambda p: need(0 < p["a"] < p["l"], "a must be between 0 and l."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(p["l"], _propped(p), [Point(p["a"], p["P"])]),
      note="Simply supported at R₁ (left), fixed at R₂ (right); P at a from R₁.")
def _c14(p, x, x1):
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    b = l - a
    R1 = P * b**2 / (2 * l**3) * (a + 2 * l)
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{Pb^2}{2l^3}(a+2l)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{Pa}{2l^3}\left(3l^2-a^2\right)", P * a / (2 * l**3) * (3 * l**2 - a**2), FORCE,
          ("R", 1))
    r.add("M_1", "at point of load", r"R_1a", R1 * a, MOMENT, ("M", a))
    r.add("M_2", "at fixed end", r"\frac{Pab}{2l^2}(a+l)", P * a * b / (2 * l**2) * (a + l), MOMENT, ("M", l, -1))
    if a < (sqrt(2) - 1) * l:
        xm = l * (l**2 + a**2) / (3 * l**2 - a**2)
        r.add(r"\Delta_{max}", "a < 0.414 l, at x = l(l² + a²)/(3l² − a²)",
              r"\frac{Pa}{3EI}\frac{\left(l^2-a^2\right)^3}{\left(3l^2-a^2\right)^2}",
              P * a / (3 * EI) * (l**2 - a**2) ** 3 / (3 * l**2 - a**2) ** 2, DEFL, ("D", xm))
    else:
        xm = l * sqrt(a / (2 * l + a))
        r.add(r"\Delta_{max}", "a > 0.414 l, at x = l√(a/(2l + a))",
              r"\frac{Pab^2}{6EI}\sqrt{\frac{a}{2l+a}}", P * a * b**2 / (6 * EI) * sqrt(a / (2 * l + a)), DEFL,
              ("D", xm))
    r.add(r"\Delta_a", "at point of load", r"\frac{Pa^2b^3}{12EIl^3}(3l+a)",
          P * a**2 * b**3 / (12 * EI * l**3) * (3 * l + a), DEFL, ("D", a))
    if _lt(x, a):
        r.add("M_x", "when x < a", r"R_1x", R1 * x, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a", r"\frac{Pb^2x}{12EIl^3}\left(3al^2-2lx^2-ax^2\right)",
              P * b**2 * x / (12 * EI * l**3) * (3 * a * l**2 - 2 * l * x**2 - a * x**2), DEFL, ("D", x), True)
    if _gt(x, a):
        r.add("M_x", "when x > a", r"R_1x-P(x-a)", R1 * x - P * (x - a), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x > a", r"\frac{Pa}{12EIl^3}(l-x)^2\left(3l^2x-a^2x-2a^2l\right)",
              P * a / (12 * EI * l**3) * (l - x) ** 2 * (3 * l**2 * x - a**2 * x - 2 * a**2 * l), DEFL,
              ("D", x), True)
    return r


@case(num=15, title="Beam fixed at both ends — uniformly distributed load", category="Fixed and propped beams",
      params=["l", "w"],
      model=lambda p: Model(p["l"], [(0, "fixed"), (p["l"], "fixed")], [Dist(0, p["l"], p["w"], p["w"])]))
def _c15(p, x, x1):
    l, w, EI = p["l"], p["w"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{2wl}{3}", 2 * w * l / 3, FORCE)
    r.add("R = V", "", r"\frac{wl}{2}", w * l / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at ends", r"\frac{wl^2}{12}", w * l**2 / 12, MOMENT, ("M", 0, 1))
    r.add("M_1", "at center", r"\frac{wl^2}{24}", w * l**2 / 24, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{wl^4}{384EI}", w * l**4 / (384 * EI), DEFL, ("D", l / 2))
    if x is not None:
        r.add("V_x", "", r"w\left(\frac{l}{2}-x\right)", w * (l / 2 - x), FORCE, ("V", x), True)
        r.add("M_x", "", r"\frac{w}{12}\left(6lx-l^2-6x^2\right)", w / 12 * (6 * l * x - l**2 - 6 * x**2), MOMENT,
              ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{wx^2}{24EI}(l-x)^2", w * x**2 / (24 * EI) * (l - x) ** 2, DEFL, ("D", x),
              True)
    return r


@case(num=16, title="Beam fixed at both ends — concentrated load at center", category="Fixed and propped beams",
      params=["l", "P"],
      model=lambda p: Model(p["l"], [(0, "fixed"), (p["l"], "fixed")], [Point(p["l"] / 2, p["P"])]))
def _c16(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", "P", P, FORCE)
    r.add("R = V", "", r"\frac{P}{2}", P / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at center and ends", r"\frac{Pl}{8}", P * l / 8, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{Pl^3}{192EI}", P * l**3 / (192 * EI), DEFL, ("D", l / 2))
    if _lt(x, l / 2):
        r.add("M_x", "when x < l/2", r"\frac{P}{8}(4x-l)", P / 8 * (4 * x - l), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2", r"\frac{Px^2}{48EI}(3l-4x)", P * x**2 / (48 * EI) * (3 * l - 4 * x),
              DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Load position from R₁, a'}, num=17, title="Beam fixed at both ends — concentrated load at any point", category="Fixed and propped beams",
      params=["l", "P", "a"],
      validate=lambda p: need(0 < p["a"] < p["l"], "a must be between 0 and l."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(p["l"], [(0, "fixed"), (p["l"], "fixed")], [Point(p["a"], p["P"])]),
      note="P at a from R₁.")
def _c17(p, x, x1):
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    b = l - a
    R1 = P * b**2 / l**3 * (3 * a + b)
    r = Results()
    r.add("R_1 = V_1", "= V_max when a < b", r"\frac{Pb^2}{l^3}(3a+b)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2", "= V_max when a > b", r"\frac{Pa^2}{l^3}(a+3b)", P * a**2 / l**3 * (a + 3 * b), FORCE,
          ("R", 1))
    r.add("M_1", "at R₁ (= M_max when a < b)", r"\frac{Pab^2}{l^2}", P * a * b**2 / l**2, MOMENT, ("M", 0, 1))
    r.add("M_2", "at R₂ (= M_max when a > b)", r"\frac{Pa^2b}{l^2}", P * a**2 * b / l**2, MOMENT, ("M", l, -1))
    r.add("M_a", "at point of load", r"\frac{2Pa^2b^2}{l^3}", 2 * P * a**2 * b**2 / l**3, MOMENT, ("M", a))
    if a >= b:
        r.add(r"\Delta_{max}", "when a > b, at x = 2al/(3a + b)", r"\frac{2Pa^3b^2}{3EI(3a+b)^2}",
              2 * P * a**3 * b**2 / (3 * EI * (3 * a + b) ** 2), DEFL, ("D", 2 * a * l / (3 * a + b)))
    else:
        r.add(r"\Delta_{max}", "a < b: a and b interchanged, x = 2bl/(3b + a) from R₂",
              r"\frac{2Pb^3a^2}{3EI(3b+a)^2}",
              2 * P * b**3 * a**2 / (3 * EI * (3 * b + a) ** 2), DEFL, ("D", l - 2 * b * l / (3 * b + a)))
    r.add(r"\Delta_a", "at point of load", r"\frac{Pa^3b^3}{3EIl^3}", P * a**3 * b**3 / (3 * EI * l**3), DEFL,
          ("D", a))
    if _lt(x, a):
        r.add("M_x", "when x < a", r"R_1x-\frac{Pab^2}{l^2}", R1 * x - P * a * b**2 / l**2, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a", r"\frac{Pb^2x^2}{6EIl^3}(3al-3ax-bx)",
              P * b**2 * x**2 / (6 * EI * l**3) * (3 * a * l - 3 * a * x - b * x), DEFL, ("D", x), True)
    return r


# =====================================================================================
# 18-23  CANTILEVERS (x measured from the free / guided end, fixed end on the right)
# =====================================================================================
_CANT_X = "measured from the free end"


@case(num=18, title="Cantilevered beam — load increasing uniformly to fixed end", category="Cantilevers",
      params=["l", "W"], x_note=_CANT_X,
      model=lambda p: Model(p["l"], [(p["l"], "fixed")], [Dist(0, p["l"], 0, 2 * p["W"] / p["l"])]),
      note="Free at the left, fixed at the right; W is the total load.")
def _c18(p, x, x1):
    l, W, EI = p["l"], p["W"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{8}{3}W", 8 * W / 3, FORCE)
    r.add("R = V", "", "W", W, FORCE, ("R", 0))
    r.add(r"M_{max}", "at fixed end", r"\frac{Wl}{3}", W * l / 3, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at free end", r"\frac{Wl^3}{15EI}", W * l**3 / (15 * EI), DEFL, ("D", 0))
    if x is not None:
        r.add("V_x", "", r"W\frac{x^2}{l^2}", W * x**2 / l**2, FORCE, ("Vneg", x), True)
        r.add("M_x", "", r"\frac{Wx^3}{3l^2}", W * x**3 / (3 * l**2), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{W}{60EIl^2}\left(x^5-5l^4x+4l^5\right)",
              W / (60 * EI * l**2) * (x**5 - 5 * l**4 * x + 4 * l**5), DEFL, ("D", x), True)
    return r


@case(num=19, title="Cantilevered beam — uniformly distributed load", category="Cantilevers",
      params=["l", "w"], x_note=_CANT_X,
      model=lambda p: Model(p["l"], [(p["l"], "fixed")], [Dist(0, p["l"], p["w"], p["w"])]),
      note="Free at the left, fixed at the right.")
def _c19(p, x, x1):
    l, w, EI = p["l"], p["w"], p["EI"]
    r = Results()
    r.add(TEUL, "", "4wl", 4 * w * l, FORCE)
    r.add("R = V", "", "wl", w * l, FORCE, ("R", 0))
    r.add(r"M_{max}", "at fixed end", r"\frac{wl^2}{2}", w * l**2 / 2, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at free end", r"\frac{wl^4}{8EI}", w * l**4 / (8 * EI), DEFL, ("D", 0))
    if x is not None:
        r.add("V_x", "", "wx", w * x, FORCE, ("Vneg", x), True)
        r.add("M_x", "", r"\frac{wx^2}{2}", w * x**2 / 2, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{w}{24EI}\left(x^4-4l^3x+3l^4\right)",
              w / (24 * EI) * (x**4 - 4 * l**3 * x + 3 * l**4), DEFL, ("D", x), True)
    return r


@case(num=20, title="Beam fixed at one end, free to deflect vertically but not rotate at other — "
                    "uniformly distributed load", category="Cantilevers",
      params=["l", "w"], x_note="measured from the guided (deflected) end",
      model=lambda p: Model(p["l"], [(0, "guided"), (p["l"], "fixed")], [Dist(0, p["l"], p["w"], p["w"])]),
      note="Left end guided (deflects, does not rotate); right end fixed.")
def _c20(p, x, x1):
    l, w, EI = p["l"], p["w"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{8}{3}wl", 8 * w * l / 3, FORCE)
    r.add("R = V", "", "wl", w * l, FORCE, ("R", 1))
    r.add("M_1", "at deflected end", r"\frac{wl^2}{6}", w * l**2 / 6, MOMENT, ("M", 0, 1))
    r.add(r"M_{max}", "at fixed end", r"\frac{wl^2}{3}", w * l**2 / 3, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at deflected end", r"\frac{wl^4}{24EI}", w * l**4 / (24 * EI), DEFL, ("D", 0))
    if x is not None:
        r.add("V_x", "", "wx", w * x, FORCE, ("Vneg", x), True)
        r.add("M_x", "", r"\frac{w}{6}\left(l^2-3x^2\right)", w / 6 * (l**2 - 3 * x**2), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{w\left(l^2-x^2\right)^2}{24EI}", w * (l**2 - x**2) ** 2 / (24 * EI), DEFL,
              ("D", x), True)
    return r


@case(labels={'a': 'Load position from free end, a'}, num=21, title="Cantilevered beam — concentrated load at any point", category="Cantilevers",
      params=["l", "P", "a"], x_note=_CANT_X,
      validate=lambda p: need(0 <= p["a"] < p["l"], "a must be between 0 and l."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(p["l"], [(p["l"], "fixed")], [Point(p["a"], p["P"])]),
      note="Free at the left, fixed at the right; P at a from the free end.")
def _c21(p, x, x1):
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    b = l - a
    r = Results()
    r.add(TEUL, "", r"\frac{8Pb}{l}", 8 * P * b / l, FORCE)
    r.add("R = V", "", "P", P, FORCE, ("R", 0))
    r.add(r"M_{max}", "at fixed end", "Pb", P * b, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at free end", r"\frac{Pb^2}{6EI}(3l-b)", P * b**2 / (6 * EI) * (3 * l - b), DEFL,
          ("D", 0))
    r.add(r"\Delta_a", "at point of load", r"\frac{Pb^3}{3EI}", P * b**3 / (3 * EI), DEFL, ("D", a))
    if _gt(x, a):
        r.add("M_x", "when x > a", "P(x-a)", P * (x - a), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x > a", r"\frac{P(l-x)^2}{6EI}(3b-l+x)", P * (l - x) ** 2 / (6 * EI) * (3 * b - l + x),
              DEFL, ("D", x), True)
    if _lt(x, a):
        r.add(r"\Delta_x", "when x < a", r"\frac{Pb^2}{6EI}(3l-3x-b)", P * b**2 / (6 * EI) * (3 * l - 3 * x - b),
              DEFL, ("D", x), True)
    return r


@case(num=22, title="Cantilevered beam — concentrated load at free end", category="Cantilevers",
      params=["l", "P"], x_note=_CANT_X,
      model=lambda p: Model(p["l"], [(p["l"], "fixed")], [Point(0, p["P"])]),
      note="Free at the left, fixed at the right.")
def _c22(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", "8P", 8 * P, FORCE)
    r.add("R = V", "", "P", P, FORCE, ("R", 0))
    r.add(r"M_{max}", "at fixed end", "Pl", P * l, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at free end", r"\frac{Pl^3}{3EI}", P * l**3 / (3 * EI), DEFL, ("D", 0))
    if x is not None:
        r.add("M_x", "", "Px", P * x, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{P}{6EI}\left(2l^3-3l^2x+x^3\right)",
              P / (6 * EI) * (2 * l**3 - 3 * l**2 * x + x**3), DEFL, ("D", x), True)
    return r


@case(num=23, title="Beam fixed at one end, free to deflect vertically but not rotate at other — "
                    "concentrated load at deflected end", category="Cantilevers",
      params=["l", "P"], x_note="measured from the guided (deflected) end",
      model=lambda p: Model(p["l"], [(0, "guided"), (p["l"], "fixed")], [Point(0, p["P"])]),
      note="Left end guided (deflects, does not rotate) and loaded; right end fixed.")
def _c23(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", "4P", 4 * P, FORCE)
    r.add("R = V", "", "P", P, FORCE, ("R", 1))
    r.add(r"M_{max}", "at both ends", r"\frac{Pl}{2}", P * l / 2, MOMENT, ("M", l, -1))
    r.add(r"\Delta_{max}", "at deflected end", r"\frac{Pl^3}{12EI}", P * l**3 / (12 * EI), DEFL, ("D", 0))
    if x is not None:
        r.add("M_x", "", r"P\left(\frac{l}{2}-x\right)", P * (l / 2 - x), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{P(l-x)^2}{12EI}(l+2x)", P * (l - x) ** 2 / (12 * EI) * (l + 2 * x), DEFL,
              ("D", x), True)
    return r


# =====================================================================================
# 24-28  OVERHANGING BEAMS (R1 at x = 0, R2 at x = l, overhang beyond R2, x1 from R2)
# =====================================================================================
def _oh_supports(p):
    return [(0, "pin"), (p["l"], "pin")]


_OH = dict(category="Overhanging beams", has_x1=True, x_note="between supports, measured from R₁")


@case(labels={'a': 'Overhang length, a'}, num=24, title="Beam overhanging one support — uniformly distributed load", params=["l", "a", "w"],
      defaults={"a": (5.0, 1.5)},
      validate=lambda p: need(p["a"] < p["l"], "The overhang a must be shorter than l."),
      model=lambda p: Model(p["l"] + p["a"], _oh_supports(p), [Dist(0, p["l"] + p["a"], p["w"], p["w"])]),
      note="Span l between R₁ and R₂, overhang a beyond R₂, load w over the full length. "
           "", **_OH)
def _c24(p, x, x1):
    l, a, w, EI = p["l"], p["a"], p["w"], p["EI"]
    R1 = w / (2 * l) * (l**2 - a**2)
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{w}{2l}\left(l^2-a^2\right)", R1, FORCE, ("R", 0))
    r.add("R_2 = V_2 + V_3", "", r"\frac{w}{2l}(l+a)^2", w / (2 * l) * (l + a) ** 2, FORCE, ("R", 1))
    r.add("V_2", "", "wa", w * a, FORCE, ("V", l, 1))
    r.add("V_3", "", r"\frac{w}{2l}\left(l^2+a^2\right)", w / (2 * l) * (l**2 + a**2), FORCE, ("Vneg", l, -1))
    r.add("M_1", "at x = (l/2)(1 − a²/l²)", r"\frac{w}{8l^2}(l+a)^2(l-a)^2",
          w / (8 * l**2) * (l + a) ** 2 * (l - a) ** 2, MOMENT, ("M", l / 2 * (1 - a**2 / l**2)))
    r.add("M_2", "at R₂", r"\frac{wa^2}{2}", w * a**2 / 2, MOMENT, ("M", l))
    if _lt(x, l):
        r.add("V_x", "between supports", r"R_1 - wx", R1 - w * x, FORCE, ("V", x), True)
        r.add("M_x", "between supports", r"\frac{wx}{2l}\left(l^2-a^2-xl\right)", w * x / (2 * l) * (l**2 - a**2 - x * l),
              MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "between supports",
              r"\frac{wx}{24EIl}\left(l^4-2l^2x^2+lx^3-2a^2l^2+2a^2x^2\right)",
              w * x / (24 * EI * l) * (l**4 - 2 * l**2 * x**2 + l * x**3 - 2 * a**2 * l**2 + 2 * a**2 * x**2),
              DEFL, ("D", x), True)
    if _lt(x1, a):
        r.add("V_{x_1}", "for overhang", r"w(a-x_1)", w * (a - x1), FORCE, ("V", l + x1), True)
        r.add("M_{x_1}", "for overhang", r"\frac{w}{2}(a-x_1)^2", w / 2 * (a - x1) ** 2, MOMENT, ("M", l + x1), True)
        r.add(r"\Delta_{x_1}", "for overhang", r"\frac{wx_1}{24EI}\left(4a^2l-l^3+6a^2x_1-4ax_1^2+x_1^3\right)",
              w * x1 / (24 * EI) * (4 * a**2 * l - l**3 + 6 * a**2 * x1 - 4 * a * x1**2 + x1**3), DEFL,
              ("D", l + x1), True)
    return r


@case(labels={'a': 'Overhang length, a'}, num=25, title="Beam overhanging one support — uniformly distributed load on overhang",
      params=["l", "a", "w"], defaults={"a": (5.0, 1.5)},
      model=lambda p: Model(p["l"] + p["a"], _oh_supports(p), [Dist(p["l"], p["l"] + p["a"], p["w"], p["w"])]),
      note="Span l between R₁ and R₂, load w on the overhang a only. R₁ acts downward.", **_OH)
def _c25(p, x, x1):
    l, a, w, EI = p["l"], p["a"], p["w"], p["EI"]
    r = Results()
    r.add("R_1 = V_1", "(downward)", r"\frac{wa^2}{2l}", w * a**2 / (2 * l), FORCE, ("R", 0))
    r.add("R_2 = V_1 + V_2", "", r"\frac{wa}{2l}(2l+a)", w * a / (2 * l) * (2 * l + a), FORCE, ("R", 1))
    r.add("V_2", "", "wa", w * a, FORCE, ("V", l, 1))
    r.add(r"M_{max}", "at R₂", r"\frac{wa^2}{2}", w * a**2 / 2, MOMENT, ("M", l))
    r.add(r"\Delta_{max}", "between supports at x = l/√3 (upward)",
          r"\frac{wa^2l^2}{18\sqrt{3}EI} = 0.0321\,\frac{wa^2l^2}{EI}", w * a**2 * l**2 / (18 * sqrt(3) * EI), DEFL,
          ("D", l / sqrt(3)))
    r.add(r"\Delta_{max}", "for overhang at x₁ = a", r"\frac{wa^3}{24EI}(4l+3a)", w * a**3 / (24 * EI) * (4 * l + 3 * a),
          DEFL, ("D", l + a))
    if _lt(x, l):
        r.add("M_x", "between supports", r"\frac{wa^2x}{2l}", w * a**2 * x / (2 * l), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "between supports (upward)", r"\frac{wa^2x}{12EIl}\left(l^2-x^2\right)",
              w * a**2 * x / (12 * EI * l) * (l**2 - x**2), DEFL, ("D", x), True)
    if _lt(x1, a):
        r.add("V_{x_1}", "for overhang", r"w(a-x_1)", w * (a - x1), FORCE, ("V", l + x1), True)
        r.add("M_{x_1}", "for overhang", r"\frac{w}{2}(a-x_1)^2", w / 2 * (a - x1) ** 2, MOMENT, ("M", l + x1), True)
        r.add(r"\Delta_{x_1}", "for overhang", r"\frac{wx_1}{24EI}\left(4a^2l+6a^2x_1-4ax_1^2+x_1^3\right)",
              w * x1 / (24 * EI) * (4 * a**2 * l + 6 * a**2 * x1 - 4 * a * x1**2 + x1**3), DEFL, ("D", l + x1), True)
    return r


@case(labels={'a': 'Overhang length, a'}, num=26, title="Beam overhanging one support — concentrated load at end of overhang",
      params=["l", "a", "P"], defaults={"a": (5.0, 1.5)},
      model=lambda p: Model(p["l"] + p["a"], _oh_supports(p), [Point(p["l"] + p["a"], p["P"])]),
      note="Span l between R₁ and R₂, P at the end of overhang a. R₁ acts downward.", **_OH)
def _c26(p, x, x1):
    l, a, P, EI = p["l"], p["a"], p["P"], p["EI"]
    r = Results()
    r.add("R_1 = V_1", "(downward)", r"\frac{Pa}{l}", P * a / l, FORCE, ("R", 0))
    r.add("R_2 = V_1 + V_2", "", r"\frac{P}{l}(l+a)", P / l * (l + a), FORCE, ("R", 1))
    r.add("V_2", "", "P", P, FORCE, ("V", l, 1))
    r.add(r"M_{max}", "at R₂", "Pa", P * a, MOMENT, ("M", l))
    r.add(r"\Delta_{max}", "between supports at x = l/√3 (upward)",
          r"\frac{Pal^2}{9\sqrt{3}EI} = 0.0642\,\frac{Pal^2}{EI}", P * a * l**2 / (9 * sqrt(3) * EI), DEFL,
          ("D", l / sqrt(3)))
    r.add(r"\Delta_{max}", "for overhang at x₁ = a", r"\frac{Pa^2}{3EI}(l+a)", P * a**2 / (3 * EI) * (l + a), DEFL,
          ("D", l + a))
    if _lt(x, l):
        r.add("M_x", "between supports", r"\frac{Pax}{l}", P * a * x / l, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "between supports (upward)", r"\frac{Pax}{6EIl}\left(l^2-x^2\right)",
              P * a * x / (6 * EI * l) * (l**2 - x**2), DEFL, ("D", x), True)
    if _lt(x1, a):
        r.add("M_{x_1}", "for overhang", r"P(a-x_1)", P * (a - x1), MOMENT, ("M", l + x1), True)
        r.add(r"\Delta_{x_1}", "for overhang", r"\frac{Px_1}{6EI}\left(2al+3ax_1-x_1^2\right)",
              P * x1 / (6 * EI) * (2 * a * l + 3 * a * x1 - x1**2), DEFL, ("D", l + x1), True)
    return r


@case(labels={'a': 'Overhang length, a'}, num=27, title="Beam overhanging one support — uniformly distributed load between supports",
      params=["l", "a", "w"], defaults={"a": (5.0, 1.5)},
      model=lambda p: Model(p["l"] + p["a"], _oh_supports(p), [Dist(0, p["l"], p["w"], p["w"])]),
      note="Span l between R₁ and R₂ loaded with w; overhang a unloaded (deflects upward).", **_OH)
def _c27(p, x, x1):
    l, a, w, EI = p["l"], p["a"], p["w"], p["EI"]
    r = Results()
    r.add(TEUL, "", "wl", w * l, FORCE)
    r.add("R = V", "", r"\frac{wl}{2}", w * l / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at center", r"\frac{wl^2}{8}", w * l**2 / 8, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{5wl^4}{384EI}", 5 * w * l**4 / (384 * EI), DEFL, ("D", l / 2))
    if _lt(x, l):
        r.add("V_x", "", r"w\left(\frac{l}{2}-x\right)", w * (l / 2 - x), FORCE, ("V", x), True)
        r.add("M_x", "", r"\frac{wx}{2}(l-x)", w * x / 2 * (l - x), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{wx}{24EI}\left(l^3-2lx^2+x^3\right)",
              w * x / (24 * EI) * (l**3 - 2 * l * x**2 + x**3), DEFL, ("D", x), True)
    if _lt(x1, a):
        r.add(r"\Delta_{x_1}", "for overhang (upward)", r"\frac{wl^3x_1}{24EI}", w * l**3 * x1 / (24 * EI), DEFL,
              ("D", l + x1), True)
    return r


@case(labels={'a': 'Load position from R₁, a'}, num=28, title="Beam overhanging one support — concentrated load at any point between supports",
      params=["l", "P", "a", "loh"],
      validate=lambda p: need(0 < p["a"] < p["l"], "a must be between 0 and l."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(p["l"] + p["loh"], _oh_supports(p), [Point(p["a"], p["P"])]),
      note="Span l between R₁ and R₂, P at a from R₁; the overhang is unloaded (deflects upward).",
      category="Overhanging beams", has_x1=True, x_note="between supports, measured from R₁")
def _c28(p, x, x1):
    l, P, a, EI = p["l"], p["P"], p["a"], p["EI"]
    r = Results()
    b = _c8_core(p, x if _lt(x, l) else None, r)
    if _between(x, a, l):
        r.add(r"\Delta_x", "when x > a", r"\frac{Pa(l-x)}{6EIl}\left(2lx-x^2-a^2\right)",
              P * a * (l - x) / (6 * EI * l) * (2 * l * x - x**2 - a**2), DEFL, ("D", x), True)
    if _lt(x1, p["loh"]):
        r.add(r"\Delta_{x_1}", "for overhang (upward)", r"\frac{Pabx_1}{6EIl}(l+a)",
              P * a * b * x1 / (6 * EI * l) * (l + a), DEFL, ("D", l + x1), True)
    return r


# =====================================================================================
# 29-31  CONTINUOUS BEAMS, TWO EQUAL SPANS
# =====================================================================================
def _two_span(p):
    l = p["l"]
    return [(0, "pin"), (l, "pin"), (2 * l, "pin")]


_TWO = dict(category="Continuous beams", x_note="measured from R₁")


@case(num=29, title="Continuous beam — two equal spans — uniform load on one span", params=["l", "w"],
      model=lambda p: Model(2 * p["l"], _two_span(p), [Dist(0, p["l"], p["w"], p["w"])]),
      note="Two spans of length l; load w on the first span only.", **_TWO)
def _c29(p, x, x1):
    l, w, EI = p["l"], p["w"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{49}{64}wl", 49 / 64 * w * l, FORCE)
    r.add("R_1 = V_1", "", r"\frac{7}{16}wl", 7 / 16 * w * l, FORCE, ("R", 0))
    r.add("R_2 = V_2 + V_3", "", r"\frac{5}{8}wl", 5 / 8 * w * l, FORCE, ("R", 1))
    r.add("R_3 = V_3", "", r"-\frac{1}{16}wl", -w * l / 16, FORCE, ("R", 2))
    r.add("V_2", "", r"\frac{9}{16}wl", 9 / 16 * w * l, FORCE, ("Vneg", l, -1))
    r.add(r"M_{max}", "at x = 7l/16", r"\frac{49}{512}wl^2", 49 / 512 * w * l**2, MOMENT, ("M", 7 * l / 16))
    r.add("M_1", "at support R₂", r"\frac{1}{16}wl^2", w * l**2 / 16, MOMENT, ("M", l))
    r.add(r"\Delta_{max}", "at 0.472 l from R₁", r"0.0092\,\frac{wl^4}{EI}", 0.0092 * w * l**4 / EI, DEFL,
          ("D", 0.472 * l, 0.01))
    if _lt(x, l):
        r.add("M_x", "when x < l", r"\frac{wx}{16}(7l-8x)", w * x / 16 * (7 * l - 8 * x), MOMENT, ("M", x), True)
    return r


@case(num=30, title="Continuous beam — two equal spans — concentrated load at center of one span",
      params=["l", "P"],
      model=lambda p: Model(2 * p["l"], _two_span(p), [Point(p["l"] / 2, p["P"])]),
      note="Two spans of length l; P at the center of the first span.", x_name=None, **_TWO)
def _c30(p, x, x1):
    l, P, EI = p["l"], p["P"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{13}{8}P", 13 / 8 * P, FORCE)
    r.add("R_1 = V_1", "", r"\frac{13}{32}P", 13 / 32 * P, FORCE, ("R", 0))
    r.add("R_2 = V_2 + V_3", "", r"\frac{11}{16}P", 11 / 16 * P, FORCE, ("R", 1))
    r.add("R_3 = V_3", "", r"-\frac{3}{32}P", -3 / 32 * P, FORCE, ("R", 2))
    r.add("V_2", "", r"\frac{19}{32}P", 19 / 32 * P, FORCE, ("Vneg", l, -1))
    r.add(r"M_{max}", "at point of load", r"\frac{13}{64}Pl", 13 / 64 * P * l, MOMENT, ("M", l / 2))
    r.add("M_1", "at support R₂", r"\frac{3}{32}Pl", 3 / 32 * P * l, MOMENT, ("M", l))
    r.add(r"\Delta_{max}", "at 0.480 l from R₁", r"0.015\,\frac{Pl^3}{EI}", 0.015 * P * l**3 / EI, DEFL,
          ("D", 0.480 * l, 0.02))
    return r


@case(labels={'a': 'Load position from R₁, a'}, num=31, title="Continuous beam — two equal spans — concentrated load at any point", params=["l", "P", "a"],
      validate=lambda p: need(0 < p["a"] < p["l"], "a must be between 0 and l (load in the first span)."),
      derived=lambda p: [("b = l - a", p["l"] - p["a"], "length")],
      model=lambda p: Model(2 * p["l"], _two_span(p), [Point(p["a"], p["P"])]),
      note="Two spans of length l; P in the first span at a from R₁.", x_name=None, **_TWO)
def _c31(p, x, x1):
    l, P, a = p["l"], p["P"], p["a"]
    b = l - a
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{Pb}{4l^3}\left(4l^2-a(l+a)\right)", P * b / (4 * l**3) * (4 * l**2 - a * (l + a)),
          FORCE, ("R", 0))
    r.add("R_2 = V_2 + V_3", "", r"\frac{Pa}{2l^3}\left(2l^2+b(l+a)\right)",
          P * a / (2 * l**3) * (2 * l**2 + b * (l + a)), FORCE, ("R", 1))
    r.add("R_3 = V_3", "", r"-\frac{Pab}{4l^3}(l+a)", -P * a * b / (4 * l**3) * (l + a), FORCE, ("R", 2))
    r.add("V_2", "", r"\frac{Pa}{4l^3}\left(4l^2+b(l+a)\right)", P * a / (4 * l**3) * (4 * l**2 + b * (l + a)),
          FORCE, ("Vneg", l, -1))
    r.add(r"M_{max}", "at point of load", r"\frac{Pab}{4l^3}\left(4l^2-a(l+a)\right)",
          P * a * b / (4 * l**3) * (4 * l**2 - a * (l + a)), MOMENT, ("M", a))
    r.add("M_1", "at support R₂", r"\frac{Pab}{4l^2}(l+a)", P * a * b / (4 * l**2) * (l + a), MOMENT, ("M", l))
    return r


# =====================================================================================
# 32-36  END MOMENTS AND APPLIED MOMENTS
# =====================================================================================
def _end_moments(p):
    return [Couple(0, -p["M1"]), Couple(p["l"], p["M2"])]


@case(num=32, title="Beam — uniformly distributed load and variable end moments", category="End and applied moments",
      params=["l", "w", "M1", "M2"],
      validate=lambda p: need(p["M1"] >= p["M2"], "The table assumes M₁ ≥ M₂ (put the larger end moment at the left)."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Dist(0, p["l"], p["w"], p["w"])] + _end_moments(p)),
      note="Simply supported span with hogging (negative) end moments M₁ ≥ M₂ entered as positive values.")
def _c32(p, x, x1):
    l, w, M1, M2, EI = p["l"], p["w"], p["M1"], p["M2"], p["EI"]
    d = (M1 - M2) / l
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{wl}{2}+\frac{M_1-M_2}{l}", w * l / 2 + d, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{wl}{2}-\frac{M_1-M_2}{l}", w * l / 2 - d, FORCE, ("R", 1))
    xm = l / 2 + (M1 - M2) / (w * l)
    r.add("M_3", "at x = l/2 + (M₁ − M₂)/(wl)",
          r"\frac{wl^2}{8}-\frac{M_1+M_2}{2}+\frac{(M_1-M_2)^2}{2wl^2}",
          w * l**2 / 8 - (M1 + M2) / 2 + (M1 - M2) ** 2 / (2 * w * l**2), MOMENT, ("M", xm))
    disc = l**2 / 4 - (M1 + M2) / w + ((M1 - M2) / (w * l)) ** 2
    if disc >= 0:
        r.add("b", "to locate inflection points (± b from x of M₃)",
              r"\sqrt{\frac{l^2}{4}-\frac{M_1+M_2}{w}+\left(\frac{M_1-M_2}{wl}\right)^2}", sqrt(disc), POS,
              ("Mzero", xm - sqrt(disc)))
    if x is not None:
        r.add("V_x", "", r"w\left(\frac{l}{2}-x\right)+\frac{M_1-M_2}{l}", w * (l / 2 - x) + d, FORCE, ("V", x), True)
        r.add("M_x", "", r"\frac{wx}{2}(l-x)+\left(\frac{M_1-M_2}{l}\right)x-M_1", w * x / 2 * (l - x) + d * x - M1,
              MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "",
              r"\frac{wx}{24EI}\left[x^3-\left(2l+\frac{4M_1}{wl}-\frac{4M_2}{wl}\right)x^2"
              r"+\frac{12M_1}{w}x+l^3-\frac{8M_1l}{w}-\frac{4M_2l}{w}\right]",
              w * x / (24 * EI) * (x**3 - (2 * l + 4 * M1 / (w * l) - 4 * M2 / (w * l)) * x**2 + 12 * M1 / w * x
                                   + l**3 - 8 * M1 * l / w - 4 * M2 * l / w), DEFL, ("Dsigned", x), True)
    return r


@case(num=33, title="Beam — concentrated load at center and variable end moments",
      category="End and applied moments", params=["l", "P", "M1", "M2"],
      defaults={"M1": (20.0, 30.0), "M2": (10.0, 15.0)},
      validate=lambda p: need(p["M1"] >= p["M2"], "The table assumes M₁ ≥ M₂ (put the larger end moment at the left)."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Point(p["l"] / 2, p["P"])] + _end_moments(p)),
      note="Simply supported span with hogging (negative) end moments M₁ ≥ M₂ entered as positive values.")
def _c33(p, x, x1):
    l, P, M1, M2, EI = p["l"], p["P"], p["M1"], p["M2"], p["EI"]
    d = (M1 - M2) / l
    r = Results()
    r.add("R_1 = V_1", "", r"\frac{P}{2}+\frac{M_1-M_2}{l}", P / 2 + d, FORCE, ("R", 0))
    r.add("R_2 = V_2", "", r"\frac{P}{2}-\frac{M_1-M_2}{l}", P / 2 - d, FORCE, ("R", 1))
    r.add("M_3", "at center", r"\frac{Pl}{4}-\frac{M_1+M_2}{2}", P * l / 4 - (M1 + M2) / 2, MOMENT, ("M", l / 2))
    if _lt(x, l / 2):
        r.add("M_x", "when x < l/2", r"\left(\frac{P}{2}+\frac{M_1-M_2}{l}\right)x-M_1", (P / 2 + d) * x - M1, MOMENT,
              ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2",
              r"\frac{Px}{48EI}\left(3l^2-4x^2-\frac{8(l-x)}{Pl}\left[M_1(2l-x)+M_2(l+x)\right]\right)",
              P * x / (48 * EI) * (3 * l**2 - 4 * x**2 - 8 * (l - x) / (P * l) * (M1 * (2 * l - x) + M2 * (l + x))),
              DEFL, ("Dsigned", x), True)
    if _gt(x, l / 2):
        r.add("M_x", "when x > l/2", r"\frac{P}{2}(l-x)+\frac{(M_1-M_2)x}{l}-M_1", P / 2 * (l - x) + d * x - M1, MOMENT,
              ("M", x), True)
    return r


@case(num=34, title="Simple beam — load increasing uniformly from center", category="End and applied moments",
      params=["l", "W"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")],
                            [Dist(0, p["l"] / 2, 2 * p["W"] / p["l"], 0),
                             Dist(p["l"] / 2, p["l"], 0, 2 * p["W"] / p["l"])]),
      note="Load zero at center, increasing linearly to both supports; W is the total load.")
def _c34(p, x, x1):
    l, W, EI = p["l"], p["W"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{2W}{3}", 2 * W / 3, FORCE)
    r.add("R = V", "", r"\frac{W}{2}", W / 2, FORCE, ("R", 0))
    r.add(r"M_{max}", "at center", r"\frac{Wl}{12}", W * l / 12, MOMENT, ("M", l / 2))
    r.add(r"\Delta_{max}", "at center", r"\frac{3Wl^3}{320EI}", 3 * W * l**3 / (320 * EI), DEFL, ("D", l / 2))
    if _lt(x, l / 2):
        r.add("V_x", "when x < l/2", r"\frac{W}{2}\left(\frac{l-2x}{l}\right)^2", W / 2 * ((l - 2 * x) / l) ** 2,
              FORCE, ("V", x), True)
        r.add("M_x", "when x < l/2", r"\frac{W}{2}\left(x-\frac{2x^2}{l}+\frac{4x^3}{3l^2}\right)",
              W / 2 * (x - 2 * x**2 / l + 4 * x**3 / (3 * l**2)), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < l/2",
              r"\frac{W}{12EI}\left(x^3-\frac{x^4}{l}+\frac{2x^5}{5l^2}-\frac{3l^2x}{8}\right)",
              W / (12 * EI) * (x**3 - x**4 / l + 2 * x**5 / (5 * l**2) - 3 * l**2 * x / 8), DEFL, ("D", x), True)
    return r


@case(num=35, title="Simple beam — concentrated moment at end", category="End and applied moments",
      params=["l", "M"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Couple(0, p["M"])]),
      note="Moment M applied at R₁.")
def _c35(p, x, x1):
    l, M, EI = p["l"], p["M"], p["EI"]
    r = Results()
    r.add(TEUL, "", r"\frac{8M}{l}", 8 * M / l, FORCE)
    r.add("R = V", "", r"\frac{M}{l}", M / l, FORCE, ("R", 0))
    r.add(r"M_{max}", "", "M", M, MOMENT, ("M", 0, 1))
    r.add(r"\Delta_{max}", "at x = 0.423 l", r"0.0642\,\frac{Ml^2}{EI}", M * l**2 / (9 * sqrt(3) * EI), DEFL,
          ("D", (1 - 1 / sqrt(3)) * l))
    if x is not None:
        r.add("M_x", "", r"M\left(1-\frac{x}{l}\right)", M * (1 - x / l), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "", r"\frac{M}{6EI}\left(3x^2-\frac{x^3}{l}-2lx\right)",
              M / (6 * EI) * (3 * x**2 - x**3 / l - 2 * l * x), DEFL, ("D", x), True)
    return r


@case(labels={'a': 'Moment position from R₁, a'}, num=36, title="Simple beam — concentrated moment at any point", category="End and applied moments",
      params=["l", "M", "a"],
      validate=lambda p: need(0 <= p["a"] <= p["l"], "a must be between 0 and l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Couple(p["a"], p["M"])]),
      note="Moment M applied at a from R₁.")
def _c36(p, x, x1):
    l, M, a, EI = p["l"], p["M"], p["a"], p["EI"]
    R = M / l
    r = Results()
    r.add(TEUL, "", r"\frac{8M}{l}", 8 * M / l, FORCE)
    r.add("R = V", "", r"\frac{M}{l}", R, FORCE, ("R", 0))
    r.add("M_1", "just left of load", "Ra", R * a, MOMENT, ("M", a, -1))
    r.add("M_2", "just right of load", "R(l-a)", R * (l - a), MOMENT, ("M", a, 1))
    if _lt(x, a):
        r.add("M_x", "when x < a", "Rx", R * x, MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x < a",
              r"\frac{M}{6EI}\left[\left(6a-\frac{3a^2}{l}-2l\right)x-\frac{x^3}{l}\right]",
              M / (6 * EI) * ((6 * a - 3 * a**2 / l - 2 * l) * x - x**3 / l), DEFL, ("D", x), True)
    if _gt(x, a):
        r.add("M_x", "when x > a", "R(l-x)", R * (l - x), MOMENT, ("M", x), True)
        r.add(r"\Delta_x", "when x > a",
              r"\frac{M}{6EI}\left[3\left(a^2+x^2\right)-\frac{x^3}{l}-\left(2l+\frac{3a^2}{l}\right)x\right]",
              M / (6 * EI) * (3 * (a**2 + x**2) - x**3 / l - (2 * l + 3 * a**2 / l) * x), DEFL, ("D", x), True)
    return r


# =====================================================================================
# 37-42  CONTINUOUS BEAMS, THREE AND FOUR EQUAL SPANS (coefficients)
# =====================================================================================
def _coef_case(num, title, n_spans, loaded, reactions, moments, dmax, note):
    """reactions: list of coefficients × wl; moments: list of (label, where, coef × wl², x/l from A);
    dmax: (coef × wl⁴/EI, x/l from A, where-text)."""
    def model(p):
        l, w = p["l"], p["w"]
        sup = [(i * l, "pin") for i in range(n_spans + 1)]
        loads = [Dist(i * l, (i + 1) * l, w, w) for i in loaded]
        return Model(n_spans * l, sup, loads)

    def results(p, x, x1):
        l, w, EI = p["l"], p["w"], p["EI"]
        r = Results()
        for i, cf in enumerate(reactions):
            s = "ABCDE"[i]
            # "%#.3g" keeps the Manual's trailing zeros (0.380, 1.20, 0.0330)
            r.add(f"R_{s}", "", "%#.3g" % cf + r"\,wl", cf * w * l, FORCE, ("R", i, 0.015))
        for name, where, cf, xl in moments:
            r.add(name, where, "%+#.3g" % cf + r"\,wl^2", cf * w * l**2, MOMENT, ("Msigned", xl * l, 0.005))
        cf, xl, where = dmax
        r.add(r"\Delta_{max}", where, "%#.2g" % cf + r"\,\frac{wl^4}{EI}", cf * w * l**4 / EI, DEFL,
              ("D", xl * l, 0.03))
        return r

    CASES[num] = Case(num=num, title=title, category="Continuous beams", params=["l", "w"], model=model,
                      results=results, x_name=None, note=note)


_coef_case(37, "Continuous beam — three equal spans — one end span unloaded", 3, [0, 1],
           [0.383, 1.20, 0.450, -0.0330],
           [("M_{AB}", "0.383 l from A", 0.0735, 0.383), ("M_B", "at support B", -0.117, 1.0),
            ("M_{BC}", "0.583 l from B", 0.0534, 1.583), ("M_C", "at support C", -0.0333, 2.0)],
           (0.0059, 0.430, "0.430 l from A"),
           "Spans AB and BC loaded with w; span CD unloaded. R_D acts downward.")
_coef_case(38, "Continuous beam — three equal spans — end spans loaded", 3, [0, 2],
           [0.450, 0.550, 0.550, 0.450],
           [("M_{AB}", "0.450 l from A", 0.101, 0.450), ("M_B = M_C", "at supports B and C", -0.0500, 1.0)],
           (0.0099, 0.479, "0.479 l from A or D"),
           "End spans loaded with w; center span unloaded.")
_coef_case(39, "Continuous beam — three equal spans — all spans loaded", 3, [0, 1, 2],
           [0.400, 1.10, 1.10, 0.400],
           [("M_{AB}", "0.400 l from A", 0.0800, 0.400), ("M_B = M_C", "at supports B and C", -0.100, 1.0),
            ("M_{BC}", "at center of BC", 0.0250, 1.5)],
           (0.0069, 0.446, "0.446 l from A or D"),
           "All three spans loaded with w.")
_coef_case(40, "Continuous beam — four equal spans — third span unloaded", 4, [0, 1, 3],
           [0.380, 1.22, 0.357, 0.598, 0.442],
           [("M_{AB}", "0.380 l from A", 0.0722, 0.380), ("M_B", "at support B", -0.121, 1.0),
            ("M_{BC}", "0.603 l from B", 0.0611, 1.603), ("M_C", "at support C", -0.0179, 2.0),
            ("M_D", "at support D", -0.0580, 3.0), ("M_{DE}", "0.442 l from E", 0.0977, 3.558)],
           (0.0094, 3.525, "0.475 l from E"),
           "Spans AB, BC and DE loaded with w; span CD unloaded.")
_coef_case(41, "Continuous beam — four equal spans — load first and third spans", 4, [0, 2],
           [0.446, 0.572, 0.464, 0.572, -0.0540],
           [("M_{AB}", "0.446 l from A", 0.0996, 0.446), ("M_B", "at support B", -0.0536, 1.0),
            ("M_C", "at support C", -0.0357, 2.0), ("M_{CD}", "0.518 l from D", 0.0805, 2.482),
            ("M_D", "at support D", -0.0536, 3.0)],
           (0.0097, 0.477, "0.477 l from A"),
           "Spans AB and CD loaded with w. R_E acts downward.")
_coef_case(42, "Continuous beam — four equal spans — all spans loaded", 4, [0, 1, 2, 3],
           [0.393, 1.14, 0.928, 1.14, 0.393],
           [("M_{AB}", "0.393 l from A", 0.0772, 0.393), ("M_B = M_D", "at supports B and D", -0.107, 1.0),
            ("M_{BC}", "0.536 l from B", 0.0364, 1.536), ("M_C", "at support C", -0.0714, 2.0)],
           (0.0065, 0.440, "0.440 l from A and E"),
           "All four spans loaded with w.")


# =====================================================================================
# 43-45  MOVING LOADS
# =====================================================================================
_MOV = dict(category="Moving loads", x_name=None, moving=True)


@case(num=43, title="Simple beam — one concentrated moving load", params=["l", "P"],
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], [Point(p["l"] / 2, p["P"])]),
      note="Diagrams show the load at mid-span (position of maximum moment).", **_MOV)
def _c43(p, x, x1):
    l, P = p["l"], p["P"]
    r = Results()
    r.add(r"R_{1\,max} = V_{1\,max}", "load at x = 0", "P", P, FORCE, ("Rmove", 0))
    r.add(r"M_{max}", "at point of load, when x = l/2", r"\frac{Pl}{4}", P * l / 4, MOMENT, ("Mmove",))
    return r


def _c44_pos(p):
    l, a = p["l"], p["a"]
    if a < (2 - sqrt(2)) * l:
        x = 0.5 * (l - a / 2)
        return x, [Point(x, p["P"]), Point(x + a, p["P"])]
    return l / 2, [Point(l / 2, p["P"])] + ([Point(l / 2 + a, p["P"])] if l / 2 + a <= l else [])


@case(labels={'a': 'Load spacing, a'}, num=44, title="Simple beam — two equal concentrated moving loads", params=["l", "P", "a"],
      defaults={"a": (4.0, 1.2)},
      validate=lambda p: need(0 < p["a"] < p["l"], "Load spacing a must be between 0 and l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], _c44_pos(p)[1]),
      note="Two loads P at fixed spacing a. Diagrams show the loads in the position of maximum moment.", **_MOV)
def _c44(p, x, x1):
    l, P, a = p["l"], p["P"], p["a"]
    r = Results()
    r.add(r"R_{1\,max} = V_{1\,max}", "load 1 at x = 0", r"P\left(2-\frac{a}{l}\right)", P * (2 - a / l), FORCE,
          ("Rmove", 0))
    if a < (2 - sqrt(2)) * l:
        r.add(r"M_{max}", "a < (2 − √2) l = 0.586 l: under load 1 at x = ½(l − a/2)",
              r"\frac{P}{2l}\left(l-\frac{a}{2}\right)^2", P / (2 * l) * (l - a / 2) ** 2, MOMENT, ("Mmove",))
    else:
        r.add(r"M_{max}", "a > 0.586 l: one load at center of span (Case 43)", r"\frac{Pl}{4}", P * l / 4, MOMENT,
              ("Mmove",))
    return r


def _c45_options(p):
    l, P1, P2, a = p["l"], p["P1"], p["P2"], p["a"]
    x = 0.5 * (l - P2 * a / (P1 + P2))
    m_two = (P1 + P2) * x**2 / l if x + a <= l else 0.0
    return x, m_two, P1 * l / 4


def _c45_loads(p):
    l = p["l"]
    x, m_two, m_one = _c45_options(p)
    if m_two >= m_one:
        return [Point(x, p["P1"]), Point(x + p["a"], p["P2"])]
    return [Point(l / 2, p["P1"])] + ([Point(l / 2 + p["a"], p["P2"])] if l / 2 + p["a"] <= l else [])


@case(labels={'a': 'Load spacing, a'}, num=45, title="Simple beam — two unequal concentrated moving loads", params=["l", "P1", "P2", "a"],
      defaults={"a": (4.0, 1.2)},
      validate=lambda p: need(p["P1"] >= p["P2"], "The table assumes P₁ ≥ P₂.") or
      need(0 < p["a"] < p["l"], "Load spacing a must be between 0 and l."),
      model=lambda p: Model(p["l"], [(0, "pin"), (p["l"], "pin")], _c45_loads(p)),
      note="Loads P₁ ≥ P₂ at fixed spacing a. Diagrams show the loads in the position of maximum moment.", **_MOV)
def _c45(p, x, x1):
    l, P1, P2, a = p["l"], p["P1"], p["P2"], p["a"]
    xp, m_two, m_one = _c45_options(p)
    r = Results()
    r.add(r"R_{1\,max} = V_{1\,max}", "P₁ at x = 0", r"P_1+P_2\frac{l-a}{l}", P1 + P2 * (l - a) / l, FORCE,
          ("Rmove", 0))
    gov_two = m_two >= m_one
    r.add(r"M_{max}", f"under P₁ at x = ½(l − P₂a/(P₁ + P₂)) = {xp / l:.3f} l" + (" — governs" if gov_two else ""),
          r"(P_1+P_2)\frac{x^2}{l}", m_two, MOMENT, ("Mmove",) if gov_two else None)
    r.add(r"M_{max}", "larger load at center, other off span (Case 43)" + ("" if gov_two else " — governs"),
          r"\frac{P_1l}{4}", m_one, MOMENT, None if gov_two else ("Mmove",))
    return r
