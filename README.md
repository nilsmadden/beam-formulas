# Beam diagrams and formulas (AISC Table 3-23)

A web calculator for the 45 beam cases in the AISC *Steel Construction Manual*, Table 3-23
"Shears, Moments and Deflections", built with [Streamlit](https://streamlit.io).
Pick a case, enter the loads and dimensions, and get a printable calculation sheet.

**Live app: https://beam-formulas.streamlit.app/**

## Using it

1. **Sidebar:** choose units (Imperial or Metric), then pick the case — either from the category and
   case lists, or with **Browse cases with pictures**, which shows a loading sketch for every case.
   Enter the loads,
   dimensions, E and I. Optionally set a point *x* (and *x₁* on an overhang) where the
   Table 3-23 "x" formulas are evaluated, and a deflection limit (l/360 etc.).
2. **Report header:** project, job no., member, by/checked, dates, and section name.
   The section name is a label only; the analysis uses the E and I you enter.
3. **Print:** the sheet is laid out as separate 8.5 × 11 in. pages, exactly as they print, with a
   running header and "Page n of N" on each. Sections are not split across pages; long tables
   break between rows with the header row repeated. Use **Print / Save as PDF** (usually 2 pages), or
   **Download report (HTML)** to keep a copy that can be printed later.

## What the calculation sheet shows

- Header block, case title, loading sketch with dimensions
- Inputs (with derived values such as b = l − a, and EI)
- **Table 3-23 formulas**: each quantity, its location/condition, the formula, and the result.
  Deflections are shown as magnitudes with ↓/↑ for direction.
- Shear, moment and deflection diagrams with peak values
- **Independent check**: reactions, peak shear, peak ± moments and peak deflections from a separate
  stiffness (finite element) analysis. This also gives values the table does not list
  (e.g. deflections for cases 4, 6, 10, 11, 29–31).
- Deflection check against the chosen limit (OK / NOT OK)

## Units

| | Imperial | Metric |
|---|---|---|
| Length | ft | m |
| Load | kips, kip/ft | kN, kN/m |
| Moment | kip·ft | kN·m |
| E | ksi | MPa |
| I | in.⁴ | ×10⁶ mm⁴ |
| Deflection | in. | mm |

Internally the formulas run in kip & inch, or N & mm.

## Notes on the table

- Cases 32 and 33 take the end moments M₁ ≥ M₂ as positive numbers for hogging (negative) moments.
- Cases 37–42 use the Manual's tabulated coefficients. Two of them are rounded in the Manual:
  case 37 R_D = −0.0330wl (exact −1/30) and case 41 R_E = −0.0540wl (exact −3/56).
  The independent check shows the exact values.
- Cases 43–45 (moving loads): the diagrams show the load position that gives maximum moment.
  For case 45 the larger of the two M_max expressions is marked "governs".

## Files

- `beams.py` – the 45 cases: inputs, formulas (LaTeX + values), and the beam model for each.
- `solver.py` – independent beam analysis (Euler–Bernoulli elements) for diagrams and checks.
- `report.py` – builds the printable HTML sheet (KaTeX for formulas, inline SVG for diagrams).
- `app.py` – the Streamlit page.
- `test_beams.py` – checks every formula in every case against `solver.py`.

## Run on this PC

```
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501. Restart the server after editing `beams.py`, `solver.py` or `report.py`.

## Test

```
.venv\Scripts\python.exe -m pytest -q
```

## Put it online (free)

Same as the flame-temperature app: push this folder (without `.venv`) to GitHub, then deploy
`app.py` at https://share.streamlit.io.

Formulas are typeset by KaTeX loaded from a CDN, so the sheet needs an internet connection
to show formulas when it is opened.
