"""Beam formulas (AISC Manual Table 3-23) — the web page.

Run locally with:  streamlit run app.py

beams.py holds the formulas, solver.py the independent stiffness analysis,
report.py the printable calculation sheet. This file is the input form.
Internal units: kip & inch (imperial) or N & mm (metric).
"""

import datetime as dt

import streamlit as st

from beams import CASES, PARAMS
from report import Units, build_report, thumb_svg

st.set_page_config(page_title="Beam Formulas — AISC Table 3-23", page_icon="📐", layout="wide")

# display unit -> internal unit
TO_INTERNAL = {
    False: {"length": 12.0, "force": 1.0, "dist": 1 / 12.0, "moment": 12.0},     # ft, kip, kip/ft, kip-ft -> kip, in
    True: {"length": 1000.0, "force": 1000.0, "dist": 1.0, "moment": 1e6},       # m, kN, kN/m, kN-m -> N, mm
}
CATEGORIES = list(dict.fromkeys(c.category for c in CASES.values()))


@st.cache_data
def thumbnails() -> dict[int, str]:
    """Label-free loading sketch for every case, drawn with the case's default inputs."""
    out = {}
    for n, c in CASES.items():
        p = {k: c.defaults.get(k, (PARAMS[k].default_imp, PARAMS[k].default_si))[0] for k in c.params}
        p["EI"] = 1.0
        out[n] = thumb_svg(c.model(p), c.moving)
    return out


def _pick(num: int):
    c = CASES[num]
    st.session_state["category"] = c.category
    st.session_state[f"case_in_{c.category}"] = num


@st.dialog("Choose a beam case — AISC Table 3-23", width="large")
def browse_cases():
    thumbs = thumbnails()
    current = st.session_state.get("current_case")
    tabs = st.tabs(CATEGORIES)
    for tab, cat in zip(tabs, CATEGORIES):
        with tab:
            nums = [c.num for c in CASES.values() if c.category == cat]
            for row in range(0, len(nums), 3):
                cols = st.columns(3)
                for col, n in zip(cols, nums[row:row + 3]):
                    with col.container(border=True):
                        st.markdown(thumbs[n], unsafe_allow_html=True)
                        st.markdown(f"**{n}.** {CASES[n].title}")
                        if st.button("Selected" if n == current else "Select", key=f"pick_{n}",
                                     type="primary" if n == current else "secondary",
                                     use_container_width=True, on_click=_pick, args=(n,)):
                            st.rerun()

# ---------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("Beam case")
    units_choice = st.radio("Units", ["Imperial", "Metric"], horizontal=True)
    metric = units_choice == "Metric"
    units = Units(metric)

    if st.button("🖼 Browse cases with pictures", use_container_width=True):
        browse_cases()
    category = st.selectbox("Category", CATEGORIES, key="category")
    in_cat = [c.num for c in CASES.values() if c.category == category]
    case = CASES[st.selectbox("Case (Table 3-23)", in_cat, format_func=lambda n: CASES[n].label,
                              key=f"case_in_{category}")]
    st.session_state["current_case"] = case.num
    st.markdown(thumbnails()[case.num], unsafe_allow_html=True)
    if case.note:
        st.caption(case.note)

    st.header("Loads and dimensions")
    p, shown = {}, {}
    for k in case.params:
        prm = PARAMS[k]
        imp, si = case.defaults.get(k, (prm.default_imp, prm.default_si))
        u = units.unit(prm.kind)
        val = st.number_input(f"{case.param_label(k)} [{u}]", min_value=0.0, value=float(si if metric else imp),
                              step=0.5 if prm.kind == "length" else 1.0, format="%.4g",
                              key=f"{case.num}_{k}_{units_choice}")
        shown[k] = val
        p[k] = val * TO_INTERNAL[metric][prm.kind]

    st.header("Section")
    section = st.text_input("Section designation (for the report)", "", placeholder="e.g. W16×26")
    E = st.number_input(f"E [{units.E_unit}]", min_value=1.0, value=200000.0 if metric else 29000.0,
                        step=1000.0, format="%.6g", key=f"E_{units_choice}")
    I = st.number_input(f"I [{units.I_unit}]", min_value=0.0001, value=212.0 if metric else 510.0,
                        step=10.0, format="%.5g", key=f"I_{units_choice}",
                        help="Moment of inertia about the bending axis. "
                             "Default: W18×35, Ix = 510 in.⁴ (W460×52, 212×10⁶ mm⁴).")
    p["EI"] = E * (I * 1e6 if metric else I)          # N·mm² or kip·in²

    limit_opt = st.selectbox("Deflection limit", ["l/360", "l/240", "l/180", "l/480", "l/600", "None"],
                             help="Used for the OK / NOT OK deflection check in the report.")
    limit = None if limit_opt == "None" else int(limit_opt.split("/")[1])

    # ---- evaluation point
    x = x1 = None
    if case.x_name or case.has_x1:
        st.header("Evaluate at a point")
        span = shown["l"]
        if case.x_name and st.checkbox(f"Formulas at x ({case.x_note})", value=True,
                                       key=f"usex_{case.num}"):
            xv = st.number_input(f"x [{units.unit('pos')}]", min_value=0.0, max_value=float(span),
                                 value=round(span / 4, 3), format="%.4g", key=f"x_{case.num}_{units_choice}_{span}")
            x = xv * TO_INTERNAL[metric]["length"]
        if case.has_x1:
            oh_key = "loh" if "loh" in case.params else "a"
            oh = shown[oh_key]
            if st.checkbox("Formulas on the overhang at x₁ (from R₂)", value=True, key=f"usex1_{case.num}"):
                x1v = st.number_input(f"x₁ [{units.unit('pos')}]", min_value=0.0, max_value=float(oh),
                                      value=round(oh / 2, 3), format="%.4g",
                                      key=f"x1_{case.num}_{units_choice}_{oh}")
                x1 = x1v * TO_INTERNAL[metric]["length"]

# ---------------------------------------------------------------- main: header + report
st.title("Beam diagrams and formulas")
st.caption("AISC *Steel Construction Manual*, Table 3-23 — Shears, Moments and Deflections. "
           "Enter the inputs on the left; print or download the calculation sheet below.")

with st.expander("Report header (project, member, engineer)", expanded=False):
    c1, c2, c3 = st.columns([2, 1, 1])
    header = {
        "project": c1.text_input("Project"),
        "job": c2.text_input("Job no."),
        "sheet": c3.text_input("Sheet", help="Your calc-package sheet number. Pages are numbered automatically."),
        "member": c1.text_input("Subject / member", f"Case {case.num}"),
        "by": c2.text_input("By"),
        "date": c3.text_input("Date", dt.date.today().isoformat()),
        "checked": c2.text_input("Checked"),
        "checked_date": c3.text_input("Checked date"),
    }
    header["section"] = section

# validation
errors = []
if p["l"] <= 0:
    errors.append("The span length l must be greater than zero.")
for k in case.params:
    if PARAMS[k].kind in ("force", "dist", "moment") and p[k] == 0 and k not in ("M2", "w2", "P2"):
        errors.append(f"{PARAMS[k].label} is zero.")
if not errors:
    msg = case.validate(p)
    if msg:
        errors.append(msg)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

try:
    html = build_report(case, p, x, x1, units, header, E, I, limit, shown)
except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
    st.error(f"These inputs cannot be evaluated: {exc}")
    st.stop()

left, right = st.columns([1, 1])
left.download_button("⬇ Download report (HTML)", html, file_name=f"beam-case-{case.num}.html",
                     mime="text/html", use_container_width=True,
                     help="Open in a browser and print, or keep with the calculations.")
right.caption("Use **Print / Save as PDF** on the sheet below for a paper or PDF copy.")

st.iframe(html, height="content")   # sizes itself to the paginated sheets
