"""(a) Which damping ratio do the paper's plotted curves correspond to?

Least-squares fits of zeta on four independent plotted objects:
  1. Fig. 5  red 'Identified FRF' (linear magnitude, m/N, 50-210 Hz)
  2. Fig. 6a black 'Uncontrolled' FRF (dB, 0-250 Hz)
  3. Fig. 6b black open-loop pole marker (sigma = -zeta*wn)
  4. Fig. 9  dashed 'Uncontrolled SLD' (zero-order a_lim at g = 0)
For 1, 2 and 4 two fits are made: zeta only (fn = 129.3 Hz, k = 1.34e7 N/m
fixed, as in Table 1), and a free fit (fn, zeta, k).  The fixed-parameter
residual at zeta = 1.45 % (Table 1) and 1.34 % is also reported.

Outputs: a_damping.json, a_damping.csv
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from common import curve, dedup, fig9_uncontrolled, save_json, save_csv
import params as P
import zoa
from b_fig6 import fig6b_markers

FN, K = P.FN, P.K
Z_TAB, Z_ALT = 0.0145, 0.0134


def sdof_mag(f, fn, z, k):
    r = f / fn
    return 1.0 / (k * np.sqrt((1 - r**2) ** 2 + (2 * z * r) ** 2))


def fit_frf(f, y, to_db, label):
    """y is magnitude (linear or dB). Returns dict of fits."""
    def model(p):
        m = sdof_mag(f, *p)
        return 20 * np.log10(m) if to_db else m

    scale = 1.0 if to_db else 1e-7          # residual scale for least_squares conditioning

    def res_z(z):
        return (model((FN, z[0], K)) - y) / scale

    def res_free(p):
        return (model((p[0], p[1], p[2] * 1e7)) - y) / scale

    out = dict(curve=label, n_points=len(f), f_range=[f.min(), f.max()])
    r1 = least_squares(res_z, [0.014], bounds=([0.005], [0.03]), xtol=1e-14, ftol=1e-14)
    out["zeta_fit_fixed_fn_k"] = r1.x[0]
    out["rms_fit_fixed"] = np.sqrt(np.mean((r1.fun * scale) ** 2))
    r2 = least_squares(res_free, [FN, 0.014, K / 1e7], xtol=1e-14, ftol=1e-14)
    out["free_fit"] = dict(fn=r2.x[0], zeta=r2.x[1], k=r2.x[2] * 1e7,
                           rms=np.sqrt(np.mean((r2.fun * scale) ** 2)))
    for z in (Z_TAB, Z_ALT):
        out[f"rms_at_zeta_{z*100:.2f}%"] = np.sqrt(np.mean((res_z([z]) * scale) ** 2))
    # peak
    i = np.argmax(y)
    out["plotted_peak"] = dict(f=f[i], value=y[i])
    for z in (Z_TAB, Z_ALT):
        ff = np.linspace(FN - 2, FN + 2, 40001)
        m = sdof_mag(ff, FN, z, K).max()
        out[f"model_peak_at_zeta_{z*100:.2f}%"] = 20 * np.log10(m) if to_db else m
    out["units"] = "dB re 1 m/N" if to_db else "m/N"
    return out


def fig9_fit():
    rpm, a = fig9_uncontrolled()
    a = a * 1e-3
    keep = (rpm >= 400) & (rpm <= 3040) & (a > 0)
    rpm, a = rpm[keep], a[keep]
    # drop the near-vertical flanks at lobe intersections: the residual there is
    # dominated by tiny rpm misalignment.  Slope is taken from the plotted curve.
    la = np.log(a)
    slope = np.abs(np.gradient(la, rpm))
    flat = slope < 0.01                     # |d ln a / d rpm| < 1 % per rpm
    out = dict(n_points=int(keep.sum()), n_points_flat=int(flat.sum()))

    def model(z, r):
        P.set_damping(z)
        return zoa.sld(r, 0.0)

    def cost(z, mask):
        return np.sqrt(np.mean((np.log(model(z, rpm[mask])) - la[mask]) ** 2))

    for name, mask in (("all_points", np.ones_like(flat)), ("flat_points", flat)):
        r = minimize_scalar(lambda z: cost(z, mask), bounds=(0.011, 0.017), method="bounded",
                            options=dict(xatol=1e-6))
        out[name] = dict(zeta_fit=r.x, rms_log_residual=r.fun,
                         rms_log_at_1p45=cost(Z_TAB, mask), rms_log_at_1p34=cost(Z_ALT, mask))
    # the absolute minimum of the lobe envelope is proportional to ~zeta(1+zeta)
    i0 = (rpm > 1000) & (rpm < 1300)
    i1 = (rpm > 2300) & (rpm < 2900)
    out["plotted_min_1000_1300rpm_mm"] = a[i0].min() * 1e3
    out["plotted_min_2300_2900rpm_mm"] = a[i1].min() * 1e3
    rr = np.arange(400, 3041, 1.0)
    for z in (Z_TAB, Z_ALT):
        m = model(z, rr)
        out[f"model_min_mm_zeta_{z*100:.2f}%"] = m.min() * 1e3
    # one-parameter scaling: a_min(zeta) ~ linear; zeta that reproduces the plotted minimum
    zs = np.linspace(0.0125, 0.0155, 7)
    mins = [model(z, rr).min() * 1e3 for z in zs]
    target = min(out["plotted_min_1000_1300rpm_mm"], out["plotted_min_2300_2900rpm_mm"])
    out["zeta_from_envelope_minimum"] = float(np.interp(target, mins, zs))
    # peak of the first lobe (near 1950 rpm)
    j = (rpm > 1800) & (rpm < 2050)
    out["plotted_lobe_peak"] = dict(rpm=rpm[j][np.argmax(a[j])], mm=a[j].max() * 1e3)
    rr2 = np.arange(1900, 1990, 0.25)
    for z in (Z_TAB, Z_ALT):
        m = model(z, rr2)
        out[f"model_lobe_peak_zeta_{z*100:.2f}%"] = dict(rpm=rr2[np.argmax(m)], mm=m.max() * 1e3)
    P.set_damping(Z_ALT)
    return out


def main():
    rep = {}
    f, y = dedup(*curve("fig5_frf.json", "identified"))
    rep["fig5_identified"] = fit_frf(f, y, False, "Fig 5 identified (red)")
    f, y = dedup(*curve("fig5_frf.json", "measured_dashed"))
    rep["fig5_measured"] = fit_frf(f, y, False, "Fig 5 measured (dashed blue)")
    f, y = dedup(*curve("fig6a_closed_loop_frf.json", "uncontrolled"))
    rep["fig6a_uncontrolled"] = fit_frf(f, y, True, "Fig 6a uncontrolled (black)")

    mk = fig6b_markers()
    ol = mk["open-loop"]["structural"]
    wn = np.hypot(ol[0], ol[1])
    rep["fig6b_open_loop_pole"] = dict(sigma=ol[0], omega_d=ol[1], wn=wn, fn_Hz=wn / 2 / np.pi,
                                       zeta=-ol[0] / wn,
                                       model_sigma_1p45=-Z_TAB * 2 * np.pi * FN,
                                       model_sigma_1p34=-Z_ALT * 2 * np.pi * FN)
    rep["fig9_uncontrolled"] = fig9_fit()
    save_json("a_damping.json", rep)

    rows = []
    for key in ("fig5_identified", "fig5_measured", "fig6a_uncontrolled"):
        r = rep[key]
        rows.append((key, r["zeta_fit_fixed_fn_k"] * 100, r["free_fit"]["zeta"] * 100,
                     r["free_fit"]["fn"], r["free_fit"]["k"], r["rms_at_zeta_1.45%"],
                     r["rms_at_zeta_1.34%"], r["units"]))
    o = rep["fig6b_open_loop_pole"]
    rows.append(("fig6b_open_loop_pole", o["zeta"] * 100, o["zeta"] * 100, o["fn_Hz"], np.nan,
                 abs(o["sigma"] - o["model_sigma_1p45"]), abs(o["sigma"] - o["model_sigma_1p34"]),
                 "rad/s (|sigma - model|)"))
    r9 = rep["fig9_uncontrolled"]
    rows.append(("fig9_uncontrolled_flat", r9["flat_points"]["zeta_fit"] * 100, np.nan, np.nan, np.nan,
                 r9["flat_points"]["rms_log_at_1p45"], r9["flat_points"]["rms_log_at_1p34"],
                 "rms of ln(a_model/a_plot)"))
    rows.append(("fig9_uncontrolled_all", r9["all_points"]["zeta_fit"] * 100, np.nan, np.nan, np.nan,
                 r9["all_points"]["rms_log_at_1p45"], r9["all_points"]["rms_log_at_1p34"],
                 "rms of ln(a_model/a_plot)"))
    save_csv("a_damping.csv", ["object", "zeta_fit_pct_fixed_fn_k", "zeta_fit_pct_free", "fn_free_Hz",
                               "k_free_N_per_m", "rms_resid_at_1.45pct", "rms_resid_at_1.34pct",
                               "resid_units"], rows)
    import json
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
