"""Assemble targets.json and targets.md from the paper-stated numbers (typed below with
page references) and the figure extractions produced by
    extract_vector_figs.py, extract_fig11_raster.py, extract_fft_peaks.py,
    check_fig11_vs_fig9.py, check_fig6b_poles.py, check_fig6a_peaks.py, context_minima.py.

Page convention: journal page J is PDF page J-245 (PDF page 1 is the repository cover
sheet), i.e. pymupdf index J-246.
Context strings are short paraphrases written for this project, not quotations.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: dict(journal_page=p, pdf_page=p - 245)


def load_csv(name):
    return list(csv.DictReader(open(os.path.join(HERE, name))))


def load_json(name):
    return json.load(open(os.path.join(HERE, name)))


T = []


def add(id_, group, quantity, value, unit, page, where, source, context, compare="", tol="", notes="",
        status="stated"):
    e = dict(id=id_, group=group, quantity=quantity, value=value, unit=unit, **J(page), where=where,
             source=source, context=context)
    if compare:
        e["compare_with"] = compare
    if tol:
        e["suggested_tolerance"] = tol
    if notes:
        e["notes"] = notes
    e["status"] = status
    T.append(e)


# ======================================================================= Table 1 (p.251)
add("T1.struct.fn", "Table 1", "preloaded structure natural frequency", 129.3, "Hz", 251, "Table 1", "table",
    "dominant workpiece mode identified by impact test (also stated in Sec. 2.2 text)", "params.FN", "exact")
add("T1.struct.zeta", "Table 1", "preloaded structure damping ratio", 1.45, "%", 251, "Table 1", "table",
    "identified damping of the dominant mode (also in Sec. 2.2 text)", "params.ZETA_TABLE1", "exact",
    notes="Figs. 6a, 6b and the uncontrolled SLD imply 1.34 % (see ambiguity A1); Fig. 5 identified curve "
          "is consistent with 1.45 %.", status="ambiguous")
add("T1.struct.k", "Table 1", "preloaded structure stiffness", 1.34e7, "N/m", 251, "Table 1", "table",
    "modal stiffness of the dominant mode", "params.K", "exact")
add("T1.beam.fn", "Table 1", "beam natural frequencies (actuator mounted)", [23, 47], "Hz", 251, "Table 1",
    "table", "beam carrying the actuator, identified before preload", notes="not used by the paper's model "
    "(beam-workpiece interface modelled as rigid, Sec. 2.2)")
add("T1.beam.zeta", "Table 1", "beam damping ratios", [4.3, 2.9], "%", 251, "Table 1", "table",
    "beam modes")
add("T1.beam.k", "Table 1", "beam stiffnesses", [7.9e5, 2.0e6], "N/m", 251, "Table 1", "table", "beam modes")
add("T1.act.fn", "Table 1", "actuator natural frequency", 8.4, "Hz", 251, "Table 1", "table",
    "proof-mass actuator resonance; sqrt(2785.6)/(2 pi) = 8.40 Hz", "params.WP/(2*pi)", "0.05 Hz")
add("T1.act.zeta", "Table 1", "actuator damping ratio", 15, "%", 251, "Table 1", "table",
    "15.834/(2 sqrt(2785.6)) = 0.150", "params.ZETA_P", "0.1 %")
add("T1.act.k", "Table 1", "actuator 'stiffness'", 2785.6, "N/m (as printed)", 251, "Table 1", "table",
    "same number as the constant term of Eq. (3)", notes="dimensionally this is omega_p^2 in rad^2/s^2, "
    "not a stiffness (ambiguity A11)", status="ambiguous")

# ======================================================================= Table 2 (p.252)
for k, (q, v, u) in enumerate([("tool diameter", 16, "mm"), ("number of teeth", 4, "-"),
                               ("helix angle", 45, "deg"), ("workpiece material", "Al-7075-T6", "-"),
                               ("cutting coefficient Krc", 180e6, "N/m^2"),
                               ("cutting coefficient Ktc", 660e6, "N/m^2"), ("milling type", "down milling", "-"),
                               ("radial depth of cut", "half immersion", "-"), ("feed per tooth", 0.05, "mm"),
                               ("time-domain steps per revolution", 512, "-"),
                               ("time-domain number of revolutions", 50, "-"),
                               ("time-domain axial layers", 30, "-")]):
    add("T2.%02d" % (k + 1), "Table 2", q, v, u, 252, "Table 2", "table",
        "cutting and simulation conditions (repeated in Sec. 3 text, Krc=180 MPa, Ktc=660 MPa)", "params", "exact")

# ======================================================================= Table 3 (p.255)
TAB3 = [(500, 15.2, 14.9), (560, 8.7, 8.0), (600, 12.2, 11.8), (800, 11.2, 10.8), (1300, 12.1, 12.0),
        (2300, 4.5, 4.9), (2500, 4.7, 5.3), (2800, 5.8, 6.6), (3000, 6.2, 7.1)]
fig9 = load_json("fig9_features.json")["features"]
c9 = {r["rpm"]: r for r in fig9["at_speeds"]}
for rpm, clip, prop in TAB3:
    note = ""
    if rpm in c9:
        d = prop - c9[rpm]["controlled_mm"]
        note = "Fig. 9 mapped-gain curve at this speed: %.2f mm (difference %+.2f mm)" % (c9[rpm]["controlled_mm"], d)
    add("T3.%d" % rpm, "Table 3", "a_lim at %d rpm: clipped DVF / proposed saturation-aware" % rpm,
        dict(clipped_DVF=clip, proposed=prop), "mm", 255, "Table 3", "table",
        "clipped-DVF baseline (gain tuned ignoring limits, force clipped at 27 N in simulation) vs proposed method",
        "a_lim of clipped-DVF controller and of the SADE gain at this speed", "0.1 mm (table resolution)", note)

# ======================================================================= equations / method numbers
add("E3", "Equations", "actuator transfer function f_act/V_in (Eq. 3)",
    "3 s^2 / (s^2 + 15.834 s + 2785.6)", "N/V", 250, "Eq. (3)", "equation",
    "force per volt of the proof-mass actuator; wn = sqrt(2785.6) rad/s (about 8.4 Hz), zeta about 0.15",
    "structure.Ga", "exact")
add("E1", "Equations", "DVF control law (Eq. 1)", "f_act = -g_dvf * xdot_m", "-", 249, "Eq. (1)", "equation",
    "actuator force proportional to measured structural velocity", "structure.Gcl")
add("E2", "Equations", "fitness (Eq. 2)", "a_lim if max|F_d| <= F_max else 0", "mm", 249, "Eq. (2)", "equation",
    "feasibility rule of the gain search", "objective.fitness")
add("E4", "Equations", "closed-loop characteristic equation (Eq. 4)",
    "1 + g_dvf s * g_a s^2/(s^2 + 2 zeta wp s + wp^2) * G(s) = 0", "-", 251, "Eq. (4)", "equation",
    "loop with actuator dynamics", "structure.Gcl")
add("S2.Fmax", "Method", "actuator force capacity F_max", 27, "N", 249, "Sec. 2.1 Step 2; Sec. 2.2", "text",
    "peak force of the Micromega ADD-45; candidates whose peak demanded force exceeds it are rejected",
    "objective.F_MAX", "exact")
add("S2.Fd", "Method", "demanded force definition", "F_d(t) = g_dvf * xdot_m(t)", "N", 249, "Sec. 2.1 Step 2",
    "text", "peak of this quantity over the time-domain run is compared with F_max",
    notes="the paper's F_d has no actuator dynamics: kind='volt' in objective.py is g|xdot| (V), kind='force' "
          "3 g|xdot| (N). The stated units of g (V s/m) make g*xdot a voltage; which one the paper compared with "
          "27 N is not stated explicitly (ambiguity A12).", status="ambiguous")
add("S2.grid", "Method", "spindle speed grid of the optimisation", dict(start=500, stop=3000, step=10), "rpm",
    249, "Sec. 2.1 and Sec. 3", "text", "per-speed gain search on a 10 rpm grid", "params.RPM_GRID", "exact")
add("S2.act.bw", "Method", "actuator usable bandwidth", 2000, "Hz (approx.)", 249, "Sec. 2.2", "text",
    "ADD-45 data; not used in the model")
add("S2.act.kf", "Method", "actuator force constant", 3, "N/V", 249, "Sec. 2.2", "text",
    "scaling of Eq. (3)", "params.GA_GAIN", "exact")
add("S2.act.fn", "Method", "actuator resonance", 8.4, "Hz", 249, "Sec. 2.2", "text", "ADD-45 resonance")
add("S2.preload", "Method", "static preload beam -> workpiece during identification", 350, "N", 251, "Sec. 2.2",
    "text", "contact preload; not a model input")
add("S2.guard", "Method", "suggested guard band below F_max", [10, 20], "% of F_max", 255, "Sec. 4", "text",
    "optional safety margin in the feasibility check (not applied in the paper's results)")

# ======================================================================= worked example (p.249)
add("X1.rpm", "Worked example", "spindle speed of worked example", 1300, "rpm", 249, "Sec. 2.1 Step 3", "text",
    "illustration of the feasibility rule", "objective at 1300 rpm", "exact")
add("X1.g", "Worked example", "candidate gain", 1240, "V s/m", 249, "Sec. 2.1 Step 3", "text",
    "a randomly drawn candidate from the search set (bounds of the set are not given)")
add("X1.alim", "Worked example", "ZOA a_lim at 1300 rpm, g=1240", 11.8, "mm", 249, "Sec. 2.1 Step 3", "text",
    "frequency-domain limit for this candidate", "zoa.a_lim(1300, 1240)", "0.1 mm",
    notes="Table 3 gives 12.0 mm at 1300 rpm for the optimised (smaller, ~1160) gain and Fig. 9 shows 11.50 mm; "
          "see ambiguity A10", status="ambiguous")
add("X1.Fd", "Worked example", "peak demanded force at a_lim (time domain)", 29, "N", 249, "Sec. 2.1 Step 3",
    "text", "exceeds 27 N, so the candidate gets zero fitness",
    "tdsim.simulate(1300, 11.8e-3, 1240) max demand", "~1 N (integer in text)")

# ======================================================================= results text (p.252-255)
add("R.grange", "Gain map", "range of optimised gains over 500-3000 rpm", dict(min=237, max=1487), "V s/m", 252,
    "Sec. 3, para. on Fig. 8", "text", "spread of the per-speed SADE gains",
    "min/max of the reproduced gain map", "~1 %",
    notes="Fig. 8 (vector) max 1487.0 at 740 rpm agrees; Fig. 8 min is 256.9 at 1940 rpm, not 237 "
          "(237 is the single-gain value) - ambiguity A2", status="ambiguous")
add("R.lowgain", "Gain map", "region of smallest gains", 1950, "rpm (approx.)", 252, "Sec. 3", "text",
    "gains are lowest around the dominant (first) lobe, about 1950 rpm",
    "location of gain-map minimum", "+-50 rpm", notes="Fig. 8 minimum at 1940 rpm; uncontrolled lobe peak at 1952 rpm (Fig. 9)")
add("R.base", "Experiments", "speeds where uncontrolled stability was measured", [1200, 2800], "rpm", 253,
    "Sec. 3", "text", "uncontrolled baseline tests",
    notes="Fig. 9 also shows an uncontrolled chatter point at 700 rpm, 4 mm (ambiguity A15)")
add("R.13x", "Headline", "max improvement of a_lim, mapped gains", 13, "x", 253, "Sec. 3; Sec. 4; Sec. 5; abstract",
    "text", "near 730 rpm, controlled / uncontrolled limiting depth",
    "ratio of mapped-gain SLD to uncontrolled SLD near 730 rpm", "rounded",
    notes="Fig. 9 curves: ratio 12.15 at 730 rpm, maximum 12.65 at 710 rpm (15.39/1.217 mm)")
add("R.13x.rpm", "Headline", "speed of the maximum improvement", 730, "rpm (approx.)", 253, "Sec. 3", "text",
    "location of the 13x improvement", "argmax of ratio", "+-30 rpm")
add("R.worst", "Headline", "worst case a_lim uncontrolled -> controlled", dict(uncontrolled=1.2, controlled=3.6),
    "mm", 253, "Sec. 3; Sec. 4 (p.253, p.255); Sec. 5", "text",
    "the least favourable speed still triples the limit",
    "global minima of uncontrolled and mapped-gain SLDs", "0.1 mm",
    notes="uncontrolled minima in Fig. 9 are 1.212 mm (matches); no 3.6 mm minimum is visible on the Fig. 9 "
          "controlled curve - ambiguity A6", status="ambiguous")
add("R.range3to13", "Headline", "range of depth increases", [3, 13], "x", 255, "Sec. 4", "text",
    "productivity statement (MRR proportional to axial depth)")
add("R.single", "Single gain", "single speed-independent SADE gain", 237, "V s/m", 253, "Sec. 4", "text",
    "one gain optimised over the whole range under the same 27 N constraint",
    "SADE run with a single gain over all speeds", "~2 %")
add("R.1300", "Single gain", "1300 rpm: mapped vs single-gain stable depth", dict(mapped=10, single_gain=3),
    "mm", 254, "Sec. 4; Sec. 5", "text", "more than threefold difference claimed",
    notes="Fig. 9/10 classify 10 mm at 1300 rpm as chatter (cut B) and 8 mm as stable; Fig. 11 single-gain "
          "curve at 1300 rpm is ~5.1 mm and the single-gain tests are at 1200 rpm (3 mm stable, 4 mm chatter) - "
          "ambiguity A8", status="ambiguous")
add("R.T3.low", "Table 3 discussion", "speed band where saturation is rarely active", [500, 800], "rpm", 254,
    "Sec. 4", "text", "clipped and proposed controllers give similar limits there")
add("R.T3.500", "Table 3 discussion", "a_lim at 500 rpm clipped vs proposed", dict(clipped_DVF=15.2, proposed=14.9),
    "mm", 254, "Sec. 4", "text", "clipped slightly higher because of its slightly larger gain", "Table 3 row 500")
add("R.T3.1300", "Table 3 discussion", "a_lim at 1300 rpm clipped vs proposed",
    dict(clipped_DVF=12.1, proposed=12.0), "mm", 254, "Sec. 4", "text", "nearly identical", "Table 3 row 1300")
add("R.T3.2500", "Table 3 discussion", "a_lim at 2500 rpm clipped -> proposed", dict(clipped_DVF=4.7, proposed=5.3),
    "mm", 254, "Sec. 4", "text", "saturation-dominated region; +12.8 %", "Table 3 row 2500")
add("R.T3.3000", "Table 3 discussion", "a_lim at 3000 rpm clipped -> proposed", dict(clipped_DVF=6.2, proposed=7.1),
    "mm", 254, "Sec. 4", "text", "saturation-dominated region; +14.5 %", "Table 3 row 3000")
add("R.T3.pct", "Table 3 discussion", "improvement of proposed over clipped at high speed", [10, 15], "%", 254,
    "Sec. 4", "text", "approximate gain in the saturation-dominated region",
    notes="Table 3 gives +8.9 % (2300), +12.8 % (2500), +13.8 % (2800), +14.5 % (3000)")

# ======================================================================= Fig. 6 legend (p.252)
FIG6 = [(500, 1487), (1300, 1162), (2500, 585), (3000, 797)]
f8 = load_json("fig8_features.json")
pk6a = {r["case"]: r for r in load_json("fig6a_peak_check.json")}
for rpm, g in FIG6:
    case = "%drpm_g%d" % (rpm, g)
    add("F6.g%d" % rpm, "Fig. 6 legend", "gain shown for %d rpm" % rpm, g, "V s/m", 252, "Fig. 6(a) legend and 6(b) labels",
        "legend", "closed-loop FRF and pole of the optimised gain at this speed",
        "gain map at %d rpm" % rpm, "~1 %",
        notes="Fig. 8 at %d rpm: %.1f; gain implied by the plotted Fig. 6a peak (1-DOF model, zeta 1.34 %%): %.0f"
              % (rpm, f8["at_speeds"][str(rpm)], pk6a[case]["gain_matching_digitised_peak"]),
        status="stated" if rpm == 1300 else "ambiguous")
add("F6.fn", "Fig. 6 legend", "marked natural frequency", 129.3, "Hz", 252, "Fig. 6(a) legend", "legend",
    "dashed vertical line at the open-loop mode")

# ======================================================================= figure-derived (vector, exact)
frf = load_json("frf_features.json")
add("F5.peak.id", "Fig. 5 (vector)", "identified FRF peak", round(frf["fig5"]["identified"]["peak_value"], 13), "m/N",
    251, "Fig. 5, red curve", "figure_vector", "peak of the identified 1-DOF FRF; equals 1/(2 zeta k) with zeta=1.45 %",
    "|G| peak with Table 1 values", "0.5 %", status="derived")
add("F5.peak.meas", "Fig. 5 (vector)", "measured FRF peak", round(frf["fig5"]["measured"]["peak_value"], 13), "m/N",
    251, "Fig. 5, blue dashed", "figure_vector", "measured receptance peak at %.1f Hz" %
    frf["fig5"]["measured"]["peak_freq_Hz"], status="derived")
for case, e in frf["fig6a"].items():
    add("F6a.%s" % case, "Fig. 6a (vector)", "closed-loop FRF peak: %s" % case,
        dict(peak_dB=e["peak_value"], peak_Hz=e["peak_freq_Hz"]), "dB re m/N, Hz", 252, "Fig. 6(a)",
        "figure_vector", "maximum of the plotted curve (equivalent 1-DOF damping %.2f %%)" % e["zeta_equiv_percent"],
        "20 log10 |Gcl| peak for the legend gain", "0.1 dB, 0.5 Hz", status="derived")
for r in load_csv("fig6b_poles.csv"):
    if float(r["imag_rad_s"]) < 0:
        continue
    add("F6b.%s.%s" % (r["case"], "struct" if float(r["imag_rad_s"]) > 500 else "act"), "Fig. 6b (vector)",
        "closed-loop pole (upper half plane): %s, %s mode" % (r["case"], "structural" if float(r["imag_rad_s"]) > 500 else "actuator"),
        dict(real=float(r["real_rad_s"]), imag=float(r["imag_rad_s"]), zeta_percent=float(r["zeta_percent"])),
        "rad/s", 252, "Fig. 6(b)", "figure_vector", "marker centre read from the vector plot",
        "roots of the closed-loop characteristic polynomial", "0.3 rad/s", status="derived")
add("F8.max", "Fig. 8 (vector)", "largest optimised gain", f8["max"], "rpm, V s/m", 253, "Fig. 8", "figure_vector",
    "maximum of the plotted gain map", "gain map", "1 %", status="derived")
add("F8.min", "Fig. 8 (vector)", "smallest optimised gain", f8["min"], "rpm, V s/m", 253, "Fig. 8", "figure_vector",
    "minimum of the plotted gain map", "gain map", "1 %", status="derived")
add("F8.at", "Fig. 8 (vector)", "gain map at selected speeds", f8["at_speeds"], "V s/m", 253, "Fig. 8",
    "figure_vector", "linear interpolation of the plotted polyline (10 rpm vertices)", "gain map", "2 %",
    status="derived")
add("F8.straight", "Fig. 8 (vector)", "straight-line stretches of the gain map", f8["straight_stretches"], "-",
    253, "Fig. 8", "figure_vector", "constant-slope runs >= 40 rpm; probably interpolated, not optimised",
    notes="ambiguity A5", status="ambiguous")
add("F9.unc.min", "Fig. 9 (vector)", "uncontrolled SLD minima", fig9["uncontrolled_minima"], "rpm, mm", 253,
    "Fig. 9 dashed", "figure_vector", "lobe minima of the uncontrolled ZOA SLD",
    "zoa.sld(rpm, 0)", "0.01 mm", status="derived")
add("F9.unc.peak", "Fig. 9 (vector)", "uncontrolled SLD lobe peaks", fig9["uncontrolled_peaks"], "rpm, mm", 253,
    "Fig. 9 dashed", "figure_vector", "lobe peaks of the uncontrolled ZOA SLD", "zoa.sld(rpm, 0)",
    "0.1 mm, 2 rpm", status="derived")
add("F9.ctrl.at", "Fig. 9 (vector)", "mapped-gain SLD and uncontrolled SLD at selected speeds", fig9["at_speeds"],
    "mm", 253, "Fig. 9 solid/dashed", "figure_vector", "curve values at the speeds of Table 3 / tests",
    "zoa.sld with the reproduced gain map", "0.1 mm", status="derived")
add("F9.ctrl.max", "Fig. 9 (vector)", "mapped-gain SLD maximum", fig9["controlled_max"], "rpm, mm", 253, "Fig. 9",
    "figure_vector", "highest point of the controlled curve", status="derived")
add("F9.ratio.max", "Fig. 9 (vector)", "maximum ratio mapped/uncontrolled", fig9["max_ratio"], "-", 253, "Fig. 9",
    "figure_vector", "basis of the 13x statement", status="derived")
add("F9.ctrl.hidden", "Fig. 9 (vector)", "controlled-curve portions covered by white rectangles",
    dict(rpm_ranges=fig9["controlled_hidden_rpm_ranges"], min_in_hidden=fig9["controlled_hidden_min"],
         min_visible=fig9["controlled_min_visible"]), "rpm, mm", 253, "Fig. 9", "figure_vector",
    "the vector path continues under opaque white boxes; there the controlled curve lies below the uncontrolled one",
    notes="ambiguity A4", status="ambiguous")
f11 = load_json("fig11_features.json")["features"]
add("F11.ctrl.min", "Fig. 11 (raster)", "single-gain SLD minima", f11["controlled_minima"], "rpm, mm", 254,
    "Fig. 11 solid", "figure_raster", "lobe minima of the g=237 controlled SLD",
    "zoa.sld(rpm, 237)", "0.1 mm", status="derived")
add("F11.ctrl.peak", "Fig. 11 (raster)", "single-gain SLD lobe peaks", f11["controlled_peaks"], "rpm, mm", 254,
    "Fig. 11 solid", "figure_raster", "cusp heights (top pixel + half line width)", "zoa.sld(rpm, 237)",
    "0.15 mm, 5 rpm", status="derived")
add("F11.ctrl.at", "Fig. 11 (raster)", "single-gain and uncontrolled SLD at selected speeds", f11["at_speeds"], "mm",
    254, "Fig. 11", "figure_raster", "column read-out (interpolated where a marker hides the line)",
    "zoa.sld(rpm, 237)", "0.1 mm", status="derived")
cal = load_json("fig11_calibration_check.json")
add("F11.cal", "Fig. 11 (raster)", "raster calibration check vs Fig. 9 dashed curve", cal, "mm", 254, "Fig. 11",
    "derived", "same uncontrolled SLD drawn in both figures; residual of the raster read-out",
    status="derived")

# ======================================================================= experimental points
exp = []
R9 = load_csv("fig9_curves.csv")


def curve(rows, name, col="depth_mm"):
    x = np.array([float(r["rpm"]) for r in rows if r["curve"] == name])
    y = np.array([float(r[col]) for r in rows if r["curve"] == name])
    o = np.argsort(x)
    return x[o], y[o]


xc9, yc9 = curve(R9, "controlled")
xu9, yu9 = curve(R9, "uncontrolled")
R11 = load_csv("fig11_curves.csv")
xc11, yc11 = curve(R11, "controlled", "depth_mm_centre")
CAP = {("Fig9", "A"): "Fig. 10 caption: cut A, stable, 1300 rpm, 8 mm",
       ("Fig9", "B"): "Fig. 10 caption: cut B, chatter, 1300 rpm, 10 mm",
       ("Fig11", "A"): "Fig. 12 caption: cut A, stable, 2800 rpm, 0.5 mm",
       ("Fig11", "B"): "Fig. 12 caption: cut B, chatter, 2800 rpm, 1.5 mm",
       ("Fig11", "C"): "Fig. 12 caption: cut C, stable, 2700 rpm, 3 mm",
       ("Fig11", "D"): "Fig. 12 caption: cut D, chatter, 2700 rpm, 4 mm"}
for fig, rows, page in (("Fig9", load_csv("fig9_markers.csv"), 253), ("Fig11", load_csv("fig11_markers.csv"), 254)):
    for r in rows:
        kind = r["kind"]
        ctrl = "none" if kind.startswith("uncontrolled") else ("mapped gain" if fig == "Fig9" else "single gain 237")
        state = "stable" if kind.endswith("stable") else "chatter"
        rpm = float(r["rpm"])
        a = float(r["depth_mm"])
        rpm_n = round(rpm / 10) * 10 if fig == "Fig9" else int(r["nominal_rpm"])
        a_n = round(a * 2) / 2
        if ctrl == "none":
            pred = float(np.interp(rpm_n, xu9, yu9))
            src = "Fig. 9 dashed (vector)"
        elif fig == "Fig9":
            pred = float(np.interp(rpm_n, xc9, yc9))
            src = "Fig. 9 solid (vector)"
        else:
            m = np.abs(xc11 - rpm_n) > 15          # skip the columns hidden by the marker itself
            pred = float(np.interp(rpm_n, xc11[m], yc11[m]))
            src = "Fig. 11 solid (raster)"
        consistent = (state == "stable" and a_n < pred) or (state == "chatter" and a_n > pred)
        lab = r["label"]
        exp.append(dict(figure=fig, journal_page=page, pdf_page=page - 245, label=lab or None,
                        control=ctrl, state=state, rpm=rpm_n, depth_mm=a_n,
                        rpm_read=round(rpm, 2), depth_read_mm=round(a, 3),
                        uncertainty=("vector: +-0.05 rpm, +-0.002 mm" if fig == "Fig9" else
                                     "raster: +-%s rpm, +-%s mm" % (r["u_rpm"], r["u_mm"])),
                        glyph=r["glyph"], caption_check=CAP.get((fig, lab), None),
                        predicted_boundary_mm=round(pred, 2), boundary_source=src,
                        consistent_with_published_curve=bool(consistent)))
with open(os.path.join(HERE, "experimental_points.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    keys = list(exp[0].keys())
    w.writerow(keys)
    for e in exp:
        w.writerow([e[k] for k in keys])

# ======================================================================= FFT lines
fft = load_csv("fft_peaks.csv")
CHAT = [("Fig10", "B", 1300, 10.0, "~149 Hz line marked with a diamond in the figure; separate from spindle harmonic 7 "
                                     "(151.7 Hz), which appears as its own line ~3 Hz higher"),
        ("Fig12", "B", 2800, 1.5, "dominant non-synchronous line"),
        ("Fig12", "D", 2700, 4.0, "non-synchronous line next to spindle harmonic 3 (135 Hz)")]
chat_freq = []
for fig, lab, rpm, depth, note in CHAT:
    rows = [r for r in fft if r["figure"] == fig and r["cut"] == lab and float(r["freq_Hz_raw"]) > 120
            and float(r["freq_Hz_raw"]) < 160]
    if fig == "Fig10":
        best = max(rows, key=lambda r: float(r["height_frac_of_panel"]))
    else:
        best = max([r for r in rows if r["class"].startswith("non")], key=lambda r: float(r["height_frac_of_panel"]))
    chat_freq.append(dict(figure=fig, cut=lab, rpm=rpm, depth_mm=depth, chatter_freq_Hz=float(best["freq_Hz_raw"]),
                          uncertainty_Hz=1.9, note=note))
    add("FFT.%s.%s" % (fig, lab), "Chatter frequency (raster)", "chatter frequency, %s cut %s (%d rpm, %.1f mm)" %
        (fig, lab, rpm, depth), float(best["freq_Hz_raw"]), "Hz", 253 if fig == "Fig10" else 254,
        "%s FFT panel" % fig.replace("Fig", "Fig. "), "figure_raster", note,
        "ZOA chatter frequency at the boundary / time-domain spectrum", "+-2 Hz (digitisation)", status="derived")

# ======================================================================= ambiguities
ctx = load_json("context_minima.json")
AMB = [
    ("A1", "Damping ratio", "Table 1 and Sec. 2.2 give 1.45 %%; the Fig. 5 identified curve peaks at 2.574e-6 m/N "
     "(= 1/(2*0.0145*1.34e7)). But the Fig. 6a uncontrolled peak (-111.10 dB -> 1.339 %%), the Fig. 6b open-loop pole "
     "(-10.89 +- 812.2j rad/s -> 1.341 %%) and the uncontrolled SLD minima (1.212 mm; model with 1.34 %%: %.3f mm) all "
     "imply 1.34 %%. The computed curves of the paper use 1.34 %%." % ctx["0.0"]["min_mm"]),
    ("A2", "Gain range", "Text: gains between 237 and 1487 V s/m. Fig. 8 maximum 1487 (740 rpm) agrees, but the Fig. 8 "
     "minimum is 256.9 V s/m at 1940 rpm. 237 is the single-gain result of Sec. 4."),
    ("A3", "Fig. 6 legend gains", "Legend: 500 rpm g=1487, 1300 rpm g=1162, 2500 rpm g=585, 3000 rpm g=797. Fig. 8 gives "
     "1479 / 1158 / 543 / 794.5 at these speeds (1487 is the Fig. 8 maximum at 740 rpm, not the 500 rpm value). The "
     "plotted Fig. 6a peaks and Fig. 6b poles are mutually consistent and, with the 1-DOF + Eq. (3) loop at zeta=1.34 %, "
     "correspond to g ~ 1543 / 1162 / 685 / 830. Only the 1300 rpm case is consistent across legend, Fig. 8 and curves."),
    ("A4", "Hidden part of the Fig. 9 controlled SLD", "The controlled curve is one vector path sampled every 10 rpm from "
     "500 to 3000 rpm. Opaque white rectangles drawn after it cover 1500-2050 rpm and 960-970 rpm - exactly the speeds "
     "where the controlled curve lies below the uncontrolled one. In the hidden band it falls to 3.23 mm at 1940 rpm "
     "while the uncontrolled SLD there is ~29 mm. In print the solid line seems to stop at 1500 and restart at 2050 rpm."),
    ("A5", "Interpolated parts of the gain map", "Fig. 8 has constant-slope stretches 500-550, 560-620, 620-740 (a single "
     "segment) and 1410-1860 rpm (25 collinear vertices). These look like linear interpolation rather than per-speed "
     "optima; 1410-1860 overlaps the hidden band of Fig. 9."),
    ("A6", "Worst case 1.2 -> 3.6 mm", "1.2 mm matches the uncontrolled minima (1.212 mm). No 3.6 mm minimum exists on the "
     "Fig. 9 controlled curve: the visible minimum is 3.82 mm at 2070 rpm and the hidden one 3.23 mm at 1940 rpm. The "
     "Fig. 11 single-gain minima are 3.52 mm. Model context (zeta 1.34 %%): global minimum of the ZOA SLD is %.2f mm for "
     "g=237 and %.2f mm for g=257 (smallest Fig. 8 gain)." % (ctx["237.0"]["min_mm"], ctx["257.0"]["min_mm"])),
    ("A7", "13x near 730 rpm", "From the Fig. 9 curves the ratio is 12.15 at 730 rpm and peaks at 12.65 (710 rpm). '13' is "
     "a rounded value; experimentally only 8 mm (stable) was cut at 700 and 760 rpm, so the 13x is a prediction."),
    ("A8", "1300 rpm: 10 mm vs 3 mm", "Sec. 4/5 say the mapped gain allows stable cutting up to 10 mm at 1300 rpm; Fig. 9 and "
     "Fig. 10 classify 1300 rpm / 10 mm as chatter (cut B), with 8 mm stable (cut A). The single-gain curve of Fig. 11 is "
     "~5.1 mm at 1300 rpm; the single-gain tests are at 1200 rpm (3 mm stable, 4 mm chatter). The '3 mm' therefore seems "
     "to be the 1200 rpm experiment and '10 mm' the boundary bracket 8-10 mm."),
    ("A9", "Table 3 vs Fig. 9", "Table 3 'proposed' a_lim differs from the Fig. 9 controlled curve by +0.5 mm at 1300 "
     "(12.0 vs 11.50), +0.24 at 2300, +0.26 at 2800 and +0.24 at 3000 rpm; the other rows agree within 0.1 mm."),
    ("A10", "Worked example vs optimum at 1300 rpm", "g=1240 gives a_lim=11.8 mm (text), the optimum g~1160 gives 12.0 mm "
     "(Table 3) and Fig. 9 shows 11.50 mm. A larger gain giving a smaller a_lim is possible (actuator-mode interaction) "
     "but the three numbers cannot all be on the same curve unless the a_lim(g) relation is non-monotonic near 1160-1240."),
    ("A11", "Actuator stiffness in Table 1", "2785.6 'N/m' equals the Eq. (3) constant wp^2 (rad^2/s^2); it is not a "
     "stiffness. The beam modes of Table 1 (23, 47 Hz) are not part of the model (rigid interface)."),
    ("A12", "Units of the demanded-force check", "g is in V s/m, so g*xdot is a voltage; Eq. (3) gives 3 N/V. The text "
     "writes F_d = g xdot and compares it with 27 N. Whether the check used g|xdot| (V), 3 g|xdot| (N) or the actuator "
     "output force is not stated; the worked example (29 N at g=1240, a=11.8 mm, 1300 rpm) can discriminate."),
    ("A13", "Experimental points vs predicted boundary", "Two mapped-gain chatter points fall below the published "
     "controlled curve: 1300 rpm/10 mm (curve 11.50 mm) and 2800 rpm/5.5 mm (curve 6.34 mm). All other points are on the "
     "predicted side. The 1200/4 and 3000/4 mm single-gain chatter points lie only ~0.1 mm above the Fig. 11 curve."),
    ("A14", "Fig. 12 control state", "The Fig. 12 caption gives no control state. From the Fig. 11 glyphs: A (+) and B "
     "(diamond) are uncontrolled cuts at 2800 rpm - the same points as the uncontrolled 2800 rpm points of Fig. 9 - while "
     "C (circle) and D (x) are single-gain controlled cuts at 2700 rpm."),
    ("A15", "Uncontrolled test speeds", "Sec. 3 says uncontrolled stability was characterised at 1200 and 2800 rpm, but "
     "Fig. 9 also has an uncontrolled chatter marker at 700 rpm, 4 mm."),
    ("A16", "Unstated optimiser settings", "SADE population size, F/CR initialisation, learning period, generation cap, "
     "stagnation tolerance and the bounds of the gain search set are not given (the set is only named)."),
    ("A17", "Fig. 10 chatter line", "The diamond-marked line of Fig. 10 cut B is at ~149.5 Hz (raw read-out), only ~2-3 Hz "
     "below spindle harmonic 7 (151.7 Hz) which shows as a separate smaller line; digitisation resolution is ~1.24 Hz/px."),
    ("A18", "Fig. 9 legend", "The legend's marker swatch is painted over by a white box; the legend shows only the two "
     "curve styles (cosmetic)."),
]

out = dict(
    paper=dict(citation="Ozsoy M, Sims ND, Ozturk E (2026) Robust active control of milling chatter under actuator "
                        "constraints: a spindle speed mapped framework with experimental validation. J Manuf Process "
                        "171:247-257. doi:10.1016/j.jmapro.2026.05.009",
               page_convention="journal page J = PDF page J-245 (PDF page 1 is the repository cover) = pymupdf index J-246",
               license_note="CC BY-NC-ND: this file holds numbers and short paraphrased context only"),
    sources=dict(vector_figures="extract_vector_figs.py (Figs. 5, 6a, 6b, 8, 9)",
                 raster_fig11="extract_fig11_raster.py", fft_panels="extract_fft_peaks.py (Figs. 10, 12)",
                 checks="check_fig11_vs_fig9.py, check_fig6b_poles.py, check_fig6a_peaks.py, context_minima.py"),
    targets=T,
    experimental_points=exp,
    chatter_frequencies=chat_freq,
    ambiguities=[dict(id=a, topic=t, description=d) for a, t, d in AMB],
)
with open(os.path.join(HERE, "targets.json"), "w") as fh:
    json.dump(out, fh, indent=1)

# ======================================================================= markdown
def fmt(v):
    if isinstance(v, float):
        return ("%.4g" % v) if abs(v) < 1e4 else ("%.3g" % v)
    if isinstance(v, dict):
        if len(json.dumps(v)) > 140:
            return "see targets.json"
        return ", ".join("%s=%s" % (k, fmt(x)) for k, x in v.items())
    if isinstance(v, list):
        if len(json.dumps(v)) > 140:
            return "see targets.json"
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


L = ["# Numerical targets - Ozsoy, Sims & Ozturk (2026), JMP 171:247-257", "",
     "Generated by `build_targets.py` from the paper (numbers only, paraphrased context) and from the figure "
     "extractions in this folder. Journal page J is PDF page J-245. Status: *stated* = number printed in the paper; "
     "*derived* = read from a figure; *ambiguous* = see the ambiguity list.", ""]
groups = []
for t in T:
    if t["group"] not in groups:
        groups.append(t["group"])
for g in groups:
    L += ["## " + g, "", "| id | quantity | value | unit | p. (PDF) | status | compare with / notes |",
          "|---|---|---|---|---|---|---|"]
    for t in [t for t in T if t["group"] == g]:
        note = t.get("compare_with", "")
        if t.get("notes"):
            note = (note + "; " if note else "") + t["notes"]
        esc = lambda x: str(x).replace("|", "\\|")
        L.append("| %s | %s | %s | %s | %d (%d) | %s | %s |" % (t["id"], esc(t["quantity"]), esc(fmt(t["value"])),
                                                               t["unit"], t["journal_page"], t["pdf_page"], t["status"],
                                                               esc(note)))
    L.append("")
L += ["## Experimental test points (Figs. 9 and 11)", "",
      "Positions read from the vector paths of Fig. 9 (exact) and by colour thresholding of the Fig. 11 JPEG "
      "(+-4 rpm, +-0.07 mm). `pred` is the published boundary at that speed; `ok` = outcome on the predicted side.", "",
      "| fig | label | control | state | rpm | depth mm | read rpm / mm | pred mm | ok | caption |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for e in sorted(exp, key=lambda e: (e["figure"], e["control"], e["rpm"], e["depth_mm"])):
    L.append("| %s | %s | %s | %s | %d | %.1f | %.1f / %.3f | %.2f | %s | %s |" % (
        e["figure"], e["label"] or "", e["control"], e["state"], e["rpm"], e["depth_mm"], e["rpm_read"],
        e["depth_read_mm"], e["predicted_boundary_mm"], "yes" if e["consistent_with_published_curve"] else "NO",
        e["caption_check"] or ""))
L += ["", "## Chatter frequencies (Figs. 10, 12 FFT panels)", "",
      "| fig | cut | rpm | depth mm | chatter f (Hz) | +- | note |", "|---|---|---|---|---|---|---|"]
for c in chat_freq:
    L.append("| %s | %s | %d | %.1f | %.1f | %.1f | %s |" % (c["figure"], c["cut"], c["rpm"], c["depth_mm"],
                                                           c["chatter_freq_Hz"], c["uncertainty_Hz"], c["note"]))
L += ["", "All spectral lines with their harmonic classification: `fft_peaks.csv`.", "",
      "## Ambiguities and inconsistencies", ""]
for a, t, d in AMB:
    L.append("- **%s %s.** %s" % (a, t, d))
L.append("")
open(os.path.join(HERE, "targets.md"), "w").write("\n".join(L))
print("targets:", len(T), "experimental points:", len(exp), "ambiguities:", len(AMB))
