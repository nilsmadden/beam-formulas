"""Printable calculation sheet (HTML) for one Table 3-23 case.

Formulas are typeset with KaTeX; diagrams are inline SVG so they print crisply.
All user-entered text is HTML-escaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import floor, log10

import numpy as np

from beams import COEF, DEFL, FORCE, MOMENT, PARAMS, POS, Case
from solver import Couple, Dist, Point, Solution

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"


# ---------------------------------------------------------------- units
@dataclass
class Units:
    metric: bool

    # internal: kip & in (imperial) or N & mm (metric)
    def scale(self, kind: str) -> float:
        """Internal -> display multiplier."""
        if self.metric:
            return {"force": 1e-3, "moment": 1e-6, "defl": 1.0, "pos": 1e-3, "length": 1e-3,
                    "dist": 1.0, "coef": 1.0}[kind]
        return {"force": 1.0, "moment": 1 / 12, "defl": 1.0, "pos": 1 / 12, "length": 1 / 12,
                "dist": 12.0, "coef": 1.0}[kind]

    def unit(self, kind: str) -> str:
        if self.metric:
            return {"force": "kN", "moment": "kN·m", "defl": "mm", "pos": "m", "length": "m",
                    "dist": "kN/m", "coef": ""}[kind]
        return {"force": "kips", "moment": "kip·ft", "defl": "in.", "pos": "ft", "length": "ft",
                "dist": "kip/ft", "coef": ""}[kind]

    def show(self, value: float, kind: str) -> str:
        u = self.unit(kind)
        return f"{fmt(value * self.scale(kind))} {u}".strip()

    @property
    def E_unit(self):
        return "MPa" if self.metric else "ksi"

    @property
    def I_unit(self):
        return "×10⁶ mm⁴" if self.metric else "in.⁴"


def fmt(v: float, sig: int = 4) -> str:
    if v == 0 or not np.isfinite(v):
        return "0"
    d = max(0, sig - 1 - int(floor(log10(abs(v)))))
    s = f"{v:,.{d}f}"
    return s.replace("-", "−")


# ---------------------------------------------------------------- SVG helpers
W_SVG, X0, X1 = 700, 70, 650


def _xmap(L):
    return lambda x: X0 + (X1 - X0) * x / L


def sketch_svg(case: Case, model, units: Units, p: dict) -> str:
    L = model.length
    X = _xmap(L)
    y_beam = 120
    parts = []
    # load scale
    wmax = max([max(abs(ld.w1), abs(ld.w2)) for ld in model.loads if isinstance(ld, Dist)] or [1])
    for ld in model.loads:
        if isinstance(ld, Dist):
            h1, h2 = 38 * ld.w1 / wmax, 38 * ld.w2 / wmax
            xa, xb = X(ld.x1), X(ld.x2)
            top = y_beam - 6
            parts.append(f'<polygon points="{xa},{top} {xa},{top - h1} {xb},{top - h2} {xb},{top}" '
                         f'class="dist"/>')
            n = max(2, int((xb - xa) / 22))
            for k in range(n + 1):
                xx = xa + (xb - xa) * k / n
                hh = h1 + (h2 - h1) * k / n
                if hh > 4:
                    parts.append(f'<line x1="{xx}" y1="{top - hh}" x2="{xx}" y2="{top - 1}" class="arrow" '
                                 f'marker-end="url(#ah)"/>')
            wl = escape(units.show(max(ld.w1, ld.w2), "dist"))
            parts.append(f'<text x="{(xa + xb) / 2}" y="{top - max(h1, h2) - 6}" class="lbl" '
                         f'text-anchor="middle">{wl}</text>')
    for ld in model.loads:
        if isinstance(ld, Point):
            xx = X(ld.x)
            parts.append(f'<line x1="{xx}" y1="{y_beam - 70}" x2="{xx}" y2="{y_beam - 5}" class="parrow" '
                         f'marker-end="url(#ahb)"/>')
            parts.append(f'<text x="{xx + 4}" y="{y_beam - 74}" class="lbl">{escape(units.show(ld.P, "force"))}'
                         f'{" (moving)" if case.moving else ""}</text>')
        elif isinstance(ld, Couple):
            xx = X(ld.x)
            cw = ld.M > 0
            # arc drawn above the beam, arrow head shows the sense
            d = (f"M {xx - 18},{y_beam - 8} A 18 18 0 1 1 {xx + 18},{y_beam - 8}" if cw
                 else f"M {xx + 18},{y_beam - 8} A 18 18 0 1 0 {xx - 18},{y_beam - 8}")
            parts.append(f'<path d="{d}" class="parrow" fill="none" marker-end="url(#ahb)"/>')
            parts.append(f'<text x="{xx}" y="{y_beam - 34}" class="lbl" text-anchor="middle">'
                         f'{escape(units.show(abs(ld.M), "moment"))}</text>')
    # beam
    parts.append(f'<line x1="{X(0)}" y1="{y_beam}" x2="{X(L)}" y2="{y_beam}" class="beam"/>')
    # supports
    for i, (xs, kind) in enumerate(model.supports):
        xx = X(xs)
        if kind == "pin":
            parts.append(f'<polygon points="{xx},{y_beam + 3} {xx - 9},{y_beam + 18} {xx + 9},{y_beam + 18}" '
                         f'class="sup"/>')
            parts.append(f'<line x1="{xx - 13}" y1="{y_beam + 21}" x2="{xx + 13}" y2="{y_beam + 21}" class="gnd"/>')
        else:
            side = -1 if xs < L / 2 else 1
            wx = xx + side * 3
            parts.append(f'<line x1="{wx}" y1="{y_beam - 26}" x2="{wx}" y2="{y_beam + 26}" class="wall"/>')
            for k in range(7):
                yy = y_beam - 24 + k * 8
                parts.append(f'<line x1="{wx}" y1="{yy}" x2="{wx + side * 8}" y2="{yy + 7}" class="gnd"/>')
            if kind == "guided":
                for yy in (y_beam - 12, y_beam + 12):
                    parts.append(f'<circle cx="{wx - side * 5}" cy="{yy}" r="3.5" class="sup"/>')
    # dimension chain
    pts = sorted(set([0.0, L] + [s for s, _ in model.supports] +
                     [ld.x for ld in model.loads if isinstance(ld, (Point, Couple))] +
                     [v for ld in model.loads if isinstance(ld, Dist) for v in (ld.x1, ld.x2)]))
    pts = [q for i, q in enumerate(pts) if i == 0 or q - pts[i - 1] > 1e-6 * L]
    yd = y_beam + 46
    parts.append(f'<line x1="{X(0)}" y1="{yd}" x2="{X(L)}" y2="{yd}" class="dim"/>')
    for q in pts:
        parts.append(f'<line x1="{X(q)}" y1="{yd - 5}" x2="{X(q)}" y2="{yd + 5}" class="dim"/>')
    for a, b in zip(pts, pts[1:]):
        parts.append(f'<text x="{(X(a) + X(b)) / 2}" y="{yd - 4}" class="dimt" text-anchor="middle">'
                     f'{escape(units.show(b - a, "length"))}</text>')
    if len(pts) > 2:
        yt = yd + 22
        parts.append(f'<line x1="{X(0)}" y1="{yt}" x2="{X(L)}" y2="{yt}" class="dim"/>')
        for q in (0, L):
            parts.append(f'<line x1="{X(q)}" y1="{yt - 5}" x2="{X(q)}" y2="{yt + 5}" class="dim"/>')
        parts.append(f'<text x="{(X(0) + X(L)) / 2}" y="{yt - 4}" class="dimt" text-anchor="middle">'
                     f'{escape(units.show(L, "length"))} total</text>')
    h = yd + (40 if len(pts) > 2 else 16)
    defs = ('<defs><marker id="ah" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
            '<path d="M0,0 L6,3 L0,6 z" class="ahf"/></marker>'
            '<marker id="ahb" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            '<path d="M0,0 L8,4 L0,8 z" class="ahf"/></marker></defs>')
    return f'<svg viewBox="0 0 {W_SVG} {h}" class="sketch">{defs}{"".join(parts)}</svg>'


def thumb_svg(model, moving: bool = False) -> str:
    """Small label-free loading sketch for the case picker (inline styles, no CSS needed)."""
    L = model.length
    X = _xmap(L)
    yb = 92
    ink, blue, red = "#1b1f24", "#3b6a99", "#b03a2e"
    parts = []
    wmax = max([max(abs(ld.w1), abs(ld.w2)) for ld in model.loads if isinstance(ld, Dist)] or [1])
    for ld in model.loads:
        if isinstance(ld, Dist):
            h1, h2 = 44 * ld.w1 / wmax, 44 * ld.w2 / wmax
            xa, xb, top = X(ld.x1), X(ld.x2), yb - 6
            parts.append(f'<polygon points="{xa},{top} {xa},{top - h1} {xb},{top - h2} {xb},{top}" '
                         f'fill="#dbe6f2" stroke="{blue}" stroke-width="2"/>')
            n = max(2, int((xb - xa) / 40))
            for k in range(n + 1):
                xx, hh = xa + (xb - xa) * k / n, h1 + (h2 - h1) * k / n
                if hh > 8:
                    parts.append(f'<line x1="{xx}" y1="{top - hh}" x2="{xx}" y2="{top - 2}" stroke="{blue}" '
                                 f'stroke-width="2"/>')
        elif isinstance(ld, Point):
            xx = X(ld.x)
            parts.append(f'<line x1="{xx}" y1="{yb - 70}" x2="{xx}" y2="{yb - 16}" stroke="{red}" stroke-width="5"/>'
                         f'<polygon points="{xx - 9},{yb - 18} {xx + 9},{yb - 18} {xx},{yb - 4}" fill="{red}"/>')
            if moving:
                parts.append(f'<path d="M {xx - 34},{yb - 74} L {xx + 34},{yb - 74}" stroke="{red}" stroke-width="3" '
                             f'stroke-dasharray="6 5"/>')
        elif isinstance(ld, Couple):
            xx = X(ld.x)
            d = (f"M {xx - 24},{yb - 8} A 24 24 0 1 1 {xx + 24},{yb - 8}" if ld.M > 0
                 else f"M {xx + 24},{yb - 8} A 24 24 0 1 0 {xx - 24},{yb - 8}")
            tx = xx + 24 if ld.M > 0 else xx - 24      # arc ends here, heading downward
            parts.append(f'<path d="{d}" fill="none" stroke="{red}" stroke-width="5"/>'
                         f'<polygon points="{tx - 9},{yb - 14} {tx + 9},{yb - 14} {tx},{yb}" fill="{red}"/>')
    parts.append(f'<line x1="{X(0)}" y1="{yb}" x2="{X(L)}" y2="{yb}" stroke="{ink}" stroke-width="8"/>')
    for xs, kind in model.supports:
        xx = X(xs)
        if kind == "pin":
            parts.append(f'<polygon points="{xx},{yb + 5} {xx - 16},{yb + 32} {xx + 16},{yb + 32}" fill="#fff" '
                         f'stroke="{ink}" stroke-width="3"/>')
        else:
            side = -1 if xs < L / 2 else 1
            wx = xx + side * 5
            parts.append(f'<line x1="{wx}" y1="{yb - 40}" x2="{wx}" y2="{yb + 40}" stroke="{ink}" stroke-width="5"/>')
            for k in range(5):
                yy = yb - 36 + k * 16
                parts.append(f'<line x1="{wx}" y1="{yy}" x2="{wx + side * 14}" y2="{yy + 12}" stroke="{ink}" '
                             f'stroke-width="2.5"/>')
            if kind == "guided":
                for yy in (yb - 18, yb + 18):
                    parts.append(f'<circle cx="{wx - side * 8}" cy="{yy}" r="6" fill="#fff" stroke="{ink}" '
                                 f'stroke-width="2.5"/>')
    return (f'<svg viewBox="30 10 660 130" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block">{"".join(parts)}</svg>')


def diagram_svg(title: str, xs, ys, L, supports, units: Units, kind: str, invert=False, cls="v") -> str:
    X = _xmap(L)
    sc = units.scale(kind)
    ys = np.asarray(ys) * sc
    H, top, bot = 128, 20, 110
    ymax, ymin = max(ys.max(), 0), min(ys.min(), 0)
    span = (ymax - ymin) or 1.0
    if invert:   # deflection drawn downward when positive (downward)
        Y = lambda v: top + (bot - top) * (v - ymin) / span
    else:
        Y = lambda v: bot - (bot - top) * (v - ymin) / span
    y0 = Y(0)
    pts = " ".join(f"{X(x):.2f},{Y(v):.2f}" for x, v in zip(xs, ys))
    poly = f"{X(xs[0]):.2f},{y0:.2f} {pts} {X(xs[-1]):.2f},{y0:.2f}"
    parts = [f'<polygon points="{poly}" class="fill {cls}"/>',
             f'<polyline points="{pts}" class="curve {cls}"/>',
             f'<line x1="{X(0)}" y1="{y0}" x2="{X(L)}" y2="{y0}" class="axis"/>']
    for s, _ in supports:
        parts.append(f'<line x1="{X(s)}" y1="{top - 8}" x2="{X(s)}" y2="{bot + 8}" class="grid"/>')
    # extreme labels
    u = units.unit(kind)
    i_max, i_min = int(np.argmax(ys)), int(np.argmin(ys))
    done = []
    for i in (i_max, i_min):
        v = ys[i]
        if abs(v) < 1e-9 * max(abs(ymax), abs(ymin), 1e-30) or i in done:
            continue
        done.append(i)
        tx, ty = X(xs[i]), Y(v)
        above = (v > 0) != invert
        anchor = "start" if tx < (X0 + X1) / 2 else "end"
        dx = 5 if anchor == "start" else -5
        if above and ty < top + 14 and tx < X0 + 140:     # keep clear of the diagram title
            above = False
        parts.append(f'<circle cx="{tx}" cy="{ty}" r="2.5" class="dot"/>')
        parts.append(f'<text x="{tx + dx}" y="{ty + (-6 if above else 14)}" class="vlbl" text-anchor="{anchor}">'
                     f'{fmt(v)} {u}</text>')
    parts.append(f'<text x="4" y="14" class="dtitle">{escape(title)}</text>')
    return f'<svg viewBox="0 0 {W_SVG} {H}" class="diag">{"".join(parts)}</svg>'


# ---------------------------------------------------------------- report
CSS = """
:root{--ink:#1b1f24;--muted:#5b6470;--rule:#c9ced6;--head:#eef1f5;--accent:#1f4e79;}
*{box-sizing:border-box}
body{margin:0;background:#e9ecf0;color:var(--ink);font:11.5px/1.42 "Segoe UI",Arial,Helvetica,sans-serif}
.toolbar{width:8.5in;margin:12px auto 0;display:flex;gap:10px;justify-content:flex-end;align-items:center}
.toolbar span{color:var(--muted);font-size:12px}
.toolbar button{font:600 13px "Segoe UI",Arial,sans-serif;padding:7px 16px;border:1px solid var(--accent);
  background:var(--accent);color:#fff;border-radius:4px;cursor:pointer}
/* source content; laid out off-screen at sheet width, then flowed into letter-size sheets */
#flow{position:absolute;left:-10000px;top:0;width:7.4in}
#pages{padding:12px 0 24px}
.sheet{width:8.5in;height:11in;margin:0 auto 18px;background:#fff;padding:.45in .55in .4in;
  box-shadow:0 1px 5px rgba(0,0,0,.2);display:flex;flex-direction:column;overflow:hidden}
.pbody{flex:1 1 auto;min-height:0;overflow:hidden}
.pbody > section:first-child h2{margin-top:2px}
.runhead{display:none;justify-content:space-between;gap:12px;border-bottom:1.5px solid var(--accent);
  color:var(--muted);font-size:10.5px;padding-bottom:3px;margin-bottom:6px}
.runhead b{color:var(--accent)}
.sheet.cont .runhead{display:flex}
.pfoot{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--rule);padding-top:4px;
  margin-top:6px;color:var(--muted);font-size:9.5px}
.pfoot .pno{font-weight:600;color:var(--ink);white-space:nowrap}
table.hdr{width:100%;border-collapse:collapse;margin-bottom:8px}
table.hdr td{border:1px solid var(--ink);padding:3px 6px;font-size:11.5px;vertical-align:top}
table.hdr .k{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.03em;display:block}
h1{font-size:17px;margin:4px 0 0;color:var(--accent)}
.sub{font-size:13.5px;font-weight:600;margin:2px 0 10px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1.5px solid var(--accent);
  color:var(--accent);margin:12px 0 4px;padding-bottom:2px}
.note{color:var(--muted);font-size:10.5px;margin:2px 0 5px}
table.t{width:100%;border-collapse:collapse;font-size:11px}
table.t th{background:var(--head);text-align:left;font-weight:600;padding:3px 6px;border-bottom:1px solid var(--rule)}
table.t td{padding:2px 6px;border-bottom:1px solid var(--rule);vertical-align:middle}
table.t td.v{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-weight:600}
table.t td.w{color:var(--muted);font-size:11px}
table.t tr.grp td{background:#f7f8fa;font-weight:600;font-size:11.5px;color:var(--accent)}
table.t tr{break-inside:avoid}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
svg{width:auto;height:auto;max-width:100%;display:block;margin:0 auto}
.sketch{max-height:1.95in} .diag{max-height:1.12in}
svg .beam{stroke:var(--ink);stroke-width:4}
svg .sup{fill:#fff;stroke:var(--ink);stroke-width:1.3}
svg .gnd{stroke:var(--ink);stroke-width:1}
svg .wall{stroke:var(--ink);stroke-width:2.2}
svg .dist{fill:#dbe6f2;stroke:#3b6a99;stroke-width:1}
svg .arrow{stroke:#3b6a99;stroke-width:1}
svg .parrow{stroke:#b03a2e;stroke-width:2}
svg .ahf{fill:currentColor}
svg #ah path{fill:#3b6a99} svg #ahb path{fill:#b03a2e}
svg .lbl{font-size:12px;fill:var(--ink);font-weight:600}
svg .dim{stroke:var(--muted);stroke-width:.8}
svg .dimt{font-size:11px;fill:var(--muted)}
svg .axis{stroke:var(--ink);stroke-width:1}
svg .grid{stroke:var(--rule);stroke-dasharray:3 3}
svg .curve{fill:none;stroke-width:1.6}
svg .fill{stroke:none}
svg .curve.v{stroke:#1f4e79} svg .fill.v{fill:#d6e3f0}
svg .curve.m{stroke:#9c2f22} svg .fill.m{fill:#f3dcd8}
svg .curve.d{stroke:#2f6b3a} svg .fill.d{fill:#dcebdc}
svg .dot{fill:var(--ink)}
svg .vlbl{font-size:11px;font-weight:600;fill:var(--ink)}
svg .dtitle{font-size:11.5px;font-weight:700;fill:var(--muted)}
.ok{color:#1d6b2f;font-weight:700}.ng{color:#a12a1a;font-weight:700}
.foot{margin-top:14px;color:var(--muted);font-size:10px}
.katex{font-size:1.05em}
@page{size:letter;margin:0}
@media print{
  body{background:#fff}
  .toolbar{display:none}
  #pages{padding:0;zoom:1!important}
  .sheet{margin:0;box-shadow:none;break-after:page;height:10.98in}
  .sheet:last-child{break-after:auto}
}
"""

# Flows the off-screen content into letter-size sheets. Sections move whole to the next sheet
# when they do not fit; long tables split between rows with the header row repeated.
PAGINATE_JS = r"""
(function(){
  var done = false;
  function paginate(){
    if (done) return; done = true;
    var flow = document.getElementById('flow'), pages = document.getElementById('pages');
    var tpl = document.getElementById('sheet-tpl');
    var body = null;
    function newPage(){
      var s = tpl.content.firstElementChild.cloneNode(true);
      if (pages.children.length) s.classList.add('cont');
      pages.appendChild(s); body = s.querySelector('.pbody');
    }
    function over(){ return body.scrollHeight > body.clientHeight + 1; }
    function tbodyOf(t){ var b = document.createElement('tbody'); t.appendChild(b); return b; }
    function place(sec){
      body.appendChild(sec);
      if (!over()) return;
      body.removeChild(sec);
      var tbl = sec.querySelector(':scope > table.t');
      if (tbl && split(sec, tbl)) return;
      if (body.children.length){
        newPage(); body.appendChild(sec);
        if (!over()) return;
        body.removeChild(sec);
        if (tbl && split(sec, tbl)) return;
      }
      body.appendChild(sec);          // taller than a sheet and cannot split: clip
    }
    function split(sec, tbl){
      var rows = Array.prototype.slice.call(tbl.rows);
      var head = rows.length && rows[0].querySelector('th') ? rows[0] : null;
      var data = head ? rows.slice(1) : rows;
      var kids = Array.prototype.slice.call(sec.children), ti = kids.indexOf(tbl);
      var part = sec.cloneNode(false);
      kids.slice(0, ti).forEach(function(k){ part.appendChild(k.cloneNode(true)); });
      var t1 = tbl.cloneNode(false), b1 = tbodyOf(t1);
      if (head) b1.appendChild(head.cloneNode(true));
      part.appendChild(t1); body.appendChild(part);
      var n = 0;
      for (var i = 0; i < data.length; i++){
        b1.appendChild(data[i].cloneNode(true));
        if (over()){ b1.removeChild(b1.lastChild); break; }
        n++;
      }
      while (n > 0 && b1.lastChild.classList.contains('grp')){ b1.removeChild(b1.lastChild); n--; }
      if (n < Math.min(2, data.length)){ body.removeChild(part); return false; }
      var rest = sec.cloneNode(false);
      if (n < data.length){
        if (kids[0] && kids[0].tagName === 'H2'){
          var h = kids[0].cloneNode(true); h.appendChild(document.createTextNode(' (continued)'));
          rest.appendChild(h);
        }
        var t2 = tbl.cloneNode(false), b2 = tbodyOf(t2);
        if (head) b2.appendChild(head.cloneNode(true));
        data.slice(n).forEach(function(r){ b2.appendChild(r.cloneNode(true)); });
        rest.appendChild(t2);
      }
      kids.slice(ti + 1).forEach(function(k){ rest.appendChild(k.cloneNode(true)); });
      if (rest.children.length){ newPage(); place(rest); }
      return true;
    }
    newPage();
    Array.prototype.slice.call(flow.children).forEach(function(el){ place(el); });
    flow.parentNode.removeChild(flow);
    var all = pages.querySelectorAll('.pno');
    for (var i = 0; i < all.length; i++) all[i].textContent = 'Page ' + (i + 1) + ' of ' + all.length;
    var cnt = document.getElementById('pcount');
    if (cnt) cnt.textContent = all.length + (all.length === 1 ? ' page' : ' pages') + ', letter size';
    fit();
  }
  // shrink the sheets to fit a narrow window (screen only)
  function fit(){
    var pages = document.getElementById('pages');
    var need = 8.5 * 96 + 24, w = document.documentElement.clientWidth;
    var k = Math.min(1, w / need);
    pages.style.zoom = k < 1 ? k : '';
  }
  window.addEventListener('resize', function(){ if (done) fit(); });
  window.addEventListener('beforeprint', paginate);
  window.addEventListener('load', function(){
    void document.getElementById('flow').offsetHeight;   // lay out so KaTeX fonts start loading
    var go = function(){ setTimeout(paginate, 30); };
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(go); else go();
    setTimeout(paginate, 4000);      // never leave the sheet unpaginated
  });
})();
"""


def _m(tex: str) -> str:
    return f"\\({tex}\\)"


def build_report(case: Case, p: dict, x, x1, units: Units, header: dict, E_disp: float, I_disp: float,
                 limit: int | None, inputs_disp: dict) -> str:
    """p: parameters in internal units (incl. 'EI'); inputs_disp: values as the user entered them."""
    model = case.model(p)
    sol = Solution(model, p["EI"])
    results = case.results(p, x, x1)
    dg = sol.diagrams()
    ext = sol.extremes()
    L = model.length

    # ---------- header
    h = {k: escape(v or "") for k, v in header.items()}
    hdr = f"""
<table class="hdr">
<tr><td colspan="2"><span class="k">Project</span>{h['project']}</td>
    <td><span class="k">Job no.</span>{h['job']}</td><td><span class="k">Sheet</span>{h['sheet']}</td></tr>
<tr><td colspan="2"><span class="k">Subject / member</span>{h['member']}</td>
    <td><span class="k">By</span>{h['by']}</td><td><span class="k">Date</span>{h['date']}</td></tr>
<tr><td colspan="2"><span class="k">Section</span>{h['section']}</td>
    <td><span class="k">Checked</span>{h['checked']}</td><td><span class="k">Date</span>{h['checked_date']}</td></tr>
</table>"""

    # ---------- inputs
    rows = []
    for k in case.params:
        prm = PARAMS[k]
        rows.append(f"<tr><td>{_m(prm.symbol)}</td><td>{escape(case.param_label(k))}</td>"
                    f"<td class='v'>{fmt(inputs_disp[k])} {units.unit(prm.kind)}</td></tr>")
    for tex, val, kind in case.derived(p):
        sym, rhs = tex.split("=", 1)
        rows.append(f"<tr><td>{_m(sym.strip())}</td><td>{_m(rhs.strip())}</td>"
                    f"<td class='v'>{units.show(val, kind)}</td></tr>")
    rows.append(f"<tr><td>{_m('E')}</td><td>Modulus of elasticity</td><td class='v'>{fmt(E_disp)} {units.E_unit}</td></tr>")
    rows.append(f"<tr><td>{_m('I')}</td><td>Moment of inertia</td><td class='v'>{fmt(I_disp)} {units.I_unit}</td></tr>")
    ei_unit = "kN·m²" if units.metric else "kip·in.²"
    ei_val = p["EI"] * (1e-9 if units.metric else 1.0)
    rows.append(f"<tr><td>{_m('EI')}</td><td>Flexural stiffness</td><td class='v'>{fmt(ei_val)} {ei_unit}</td></tr>")
    if x is not None:
        rows.append(f"<tr><td>{_m(case.x_name)}</td><td>Evaluation point, {escape(case.x_note)}</td>"
                    f"<td class='v'>{units.show(x, 'pos')}</td></tr>")
    if x1 is not None:
        rows.append(f"<tr><td>{_m('x_1')}</td><td>Evaluation point on overhang, measured from R₂</td>"
                    f"<td class='v'>{units.show(x1, 'pos')}</td></tr>")
    inputs_html = "<table class='t'><tr><th style='width:14%'>Symbol</th><th>Description</th>" \
                  "<th style='text-align:right'>Value</th></tr>" + "".join(rows) + "</table>"

    # ---------- results
    def res_row(r):
        val = r.value
        if r.kind == DEFL:
            pos = r.chk[1] if r.chk else None
            direction = ""
            if pos is not None:
                d = sol.deflection(pos)
                if abs(d) > 1e-12 * max(abs(ext["D"][0][0]), abs(ext["D"][1][0]), 1e-30):
                    direction = " ↓" if d > 0 else " ↑"
            shown = f"{fmt(abs(val) * units.scale(DEFL))} {units.unit(DEFL)}{direction}"
        else:
            shown = units.show(val, r.kind)
        where = escape(r.where)
        return (f"<tr><td>{_m(r.name)}</td><td class='w'>{where}</td><td>{_m(chr(92) + 'displaystyle ' + r.formula)}</td>"
                f"<td class='v'>{shown}</td></tr>")

    general = [r for r in results if not r.at_x]
    at_x = [r for r in results if r.at_x]
    rrows = [res_row(r) for r in general]
    if at_x:
        lab = []
        if x is not None:
            lab.append(f"x = {units.show(x, 'pos')}")
        if x1 is not None:
            lab.append(f"x₁ = {units.show(x1, 'pos')}")
        rrows.append(f"<tr class='grp'><td colspan='4'>At {escape(', '.join(lab))}</td></tr>")
        rrows += [res_row(r) for r in at_x]
    results_html = ("<table class='t'><tr><th style='width:17%'>Quantity</th><th style='width:24%'>Location / "
                    "condition</th><th>Formula (Table 3-23)</th><th style='text-align:right'>Result</th></tr>"
                    + "".join(rrows) + "</table>")

    # ---------- diagrams
    sups = model.supports
    diagrams = (diagram_svg("SHEAR  V", dg["xv"], dg["V"], L, sups, units, FORCE, cls="v")
                + diagram_svg("MOMENT  M", dg["xv"], dg["M"], L, sups, units, MOMENT, cls="m")
                + diagram_svg("DEFLECTION  Δ  (downward +)", dg["xd"], dg["D"], L, sups, units, DEFL,
                              invert=True, cls="d"))

    # ---------- independent analysis summary
    def at(xv):
        return units.show(xv, "pos")

    names = ["R_A", "R_B", "R_C", "R_D", "R_E"] if len(sups) > 3 else ["R_1", "R_2", "R_3"]
    a_rows = []
    for i, rc in enumerate(sups):
        rc = sol.reactions[i]
        txt = units.show(rc.F, FORCE)
        if rc.kind in ("fixed", "guided"):
            txt += f"; moment {units.show(abs(rc.M), MOMENT)}"
        if rc.kind == "guided":
            txt = f"moment {units.show(abs(rc.M), MOMENT)} (guided end)"
        nm = names[i] if len(sups) > 1 else "R"
        if len(sups) == 2 and sups[0][1] == "guided":
            nm = ["M_1", "R"][i]
        a_rows.append(f"<tr><td>{_m(nm)}</td><td class='w'>at {at(rc.x)}</td>"
                      f"<td class='v'>{txt}</td></tr>")
    (vmax, xvmax), (vmin, xvmin) = ext["V"]
    (mmax, xmmax), (mmin, xmmin) = ext["M"]
    (dmax, xdmax), (dmin, xdmin) = ext["D"]
    vabs, xva = (vmax, xvmax) if abs(vmax) >= abs(vmin) else (vmin, xvmin)
    a_rows.append(f"<tr><td>|V|max</td><td class='w'>at {at(xva)}</td><td class='v'>{units.show(abs(vabs), FORCE)}</td></tr>")
    if mmax > 1e-9 * max(abs(mmin), 1):
        a_rows.append(f"<tr><td>+M max (sagging)</td><td class='w'>at {at(xmmax)}</td>"
                      f"<td class='v'>{units.show(mmax, MOMENT)}</td></tr>")
    if mmin < -1e-9 * max(abs(mmax), 1):
        a_rows.append(f"<tr><td>−M max (hogging)</td><td class='w'>at {at(xmmin)}</td>"
                      f"<td class='v'>{units.show(mmin, MOMENT)}</td></tr>")
    if dmax > 0:
        a_rows.append(f"<tr><td>Δmax downward</td><td class='w'>at {at(xdmax)}</td>"
                      f"<td class='v'>{units.show(dmax, DEFL)}</td></tr>")
    if dmin < -1e-9 * max(dmax, 1e-30):
        a_rows.append(f"<tr><td>Δmax upward</td><td class='w'>at {at(xdmin)}</td>"
                      f"<td class='v'>{units.show(-dmin, DEFL)}</td></tr>")
    analysis_html = ("<table class='t'><tr><th style='width:30%'>Quantity</th><th>Location</th>"
                     "<th style='text-align:right'>Value</th></tr>" + "".join(a_rows) + "</table>")

    # ---------- deflection check
    check_html = ""
    if limit and not case.moving:
        span = p["l"]
        dabs = max(abs(dmax), abs(dmin))
        allow = span / limit
        ratio = span / dabs if dabs > 0 else float("inf")
        ok = dabs <= allow
        span_txt = "cantilever length l" if case.category == "Cantilevers" else "span l"
        check_html = (
            f"<h2>Deflection check</h2><table class='t'>"
            f"<tr><td>Allowable, {escape(span_txt)} / {limit}</td><td class='v'>{units.show(allow, DEFL)}</td></tr>"
            f"<tr><td>Maximum deflection (analysis)</td><td class='v'>{units.show(dabs, DEFL)}"
            f" = l / {fmt(ratio, 3) if np.isfinite(ratio) else '∞'}</td></tr>"
            f"<tr><td>Result</td><td class='v {'ok' if ok else 'ng'}'>"
            f"{'OK — Δ ≤ l/' + str(limit) if ok else 'NOT OK — Δ > l/' + str(limit)}</td></tr></table>"
            f"<p class='note'>Deflection limits are a design choice (e.g. IBC Table 1604.3). "
            f"For cantilevers many engineers check 2l/limit instead.</p>")

    # running header (sheets 2+) and footer
    run_right = " · ".join(v for v in (h["project"], h["member"], h["job"] and "Job " + h["job"]) if v)
    foot_left = "AISC Table 3-23, Case " + str(case.num)
    if h["by"]:
        foot_left += f" · By {h['by']}" + (f" {h['date']}" if h["date"] else "")
    if h["checked"]:
        foot_left += f" · Checked {h['checked']}" + (f" {h['checked_date']}" if h["checked_date"] else "")

    note = f"<p class='note'>{escape(case.note)}</p>" if case.note else ""
    moving = ("<p class='note'>Moving-load case: diagrams show one load position (the one giving "
              "maximum moment), not an envelope.</p>" if case.moving else "")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Table 3-23 Case {case.num}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\(',right:'\\\\)',display:false}}]}});"></script>
<style>{CSS}</style></head>
<body>
<div class="toolbar"><span id="pcount"></span><button onclick="window.print()">Print / Save as PDF</button></div>
<template id="sheet-tpl"><div class="sheet">
<div class="runhead"><span><b>Case {case.num}</b> — {escape(case.title)}</span><span>{run_right}</span></div>
<div class="pbody"></div>
<div class="pfoot"><span>{foot_left}</span><span class="pno"></span></div>
</div></template>
<div id="flow">
<section>
{hdr}
<h1>Shears, Moments and Deflections — AISC Manual Table 3-23</h1>
<div class="sub">Case {case.num}: {escape(case.title)}</div>
</section>

<section><h2>Loading and geometry</h2>
{sketch_svg(case, model, units, p)}
{note}</section>

<section><h2>Input</h2>
{inputs_html}</section>

<section><h2>Results — Table 3-23 formulas</h2>
{results_html}
<p class="note">Deflection results are shown as magnitudes; ↓ downward, ↑ upward (from the analysis below).</p></section>

<section><h2>Shear, moment and deflection diagrams</h2>
{moving}
{diagrams}</section>

<section><h2>Independent check — stiffness analysis</h2>
<p class="note">Computed separately from the Table 3-23 formulas (Euler–Bernoulli beam elements), as a check
and for values the table does not give. Sagging moment positive.</p>
{analysis_html}</section>
{f"<section>{check_html}</section>" if check_html else ""}

<section class="foot">Formulas from AISC <i>Steel Construction Manual</i>, Table 3-23 “Shears, Moments and
Deflections”. Linear elastic, prismatic member, small deflections, shear deformation neglected.
Verify the case and its assumptions apply to the member being designed.</section>
</div>
<div id="pages"></div>
<script>{PAGINATE_JS}</script>
</body></html>"""
