"""Check 4: the voltage limit vmax (clipped DVF) in src/tdsim.c.

(a) applied command never exceeds vmax and equals clip(-g xdot, +-vmax) sample-wise;
(b) summary max_v equals max |vcmd|, max_demand is the unclipped |g xdot|;
(c) vmax above the peak demand reproduces the unlimited run bit-for-bit;
(d) x, xdot and the actuator force agree with an independent reference that
    integrates the clipped loop with DOP853 (rtol 1e-11) and realises Ga(s)
    with scipy's tf2ss (ref_sim.ref_clipped);
(e) |f_act| bound: for a clipped input the actuator force is Ga * V, so its
    peak can exceed 3*vmax (Ga has a resonant peak of 3/(2 zeta_p) ~ 10 N/V);
(f) effect on stability: time-domain boundary vs vmax at 1300 rpm, g = 1200.
Writes check4_vmax.json and check4_vmax.png.
"""
import json
import warnings
import common
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
import ref_sim as R
import tdsim
import zoa
import detector as Dt

warnings.filterwarnings("ignore", module="scipy.signal")
res = {}

rpm, g = 1300, 1200.0
a_z = zoa.a_lim(rpm, g)
a = 0.9 * a_z
nrev = 60
free = tdsim.simulate(rpm, a, g, store=True, nrev=nrev)
peak = free["max_demand"]
res["case"] = dict(rpm=rpm, g=g, a_mm=a * 1e3, a_zoa_mm=a_z * 1e3,
                   unclipped_peak_demand_V=peak, unclipped_peak_fact_N=free["max_fact"])

checks = []
for frac in (2.0, 1.0001, 0.75, 0.5, 0.25, 0.1, 0.03):
    vmax = frac * peak
    r = tdsim.simulate(rpm, a, g, vmax=vmax, store=True, nrev=nrev)
    n = len(r["x"]) - 1 if not r["diverged"] else None
    vc, v = r["vcmd"][:n], r["v"][:n]
    expect = np.clip(-g * v, -vmax, vmax)
    c = dict(vmax_over_peak=frac, vmax=vmax,
             max_abs_vcmd=float(np.max(np.abs(vc))),
             vcmd_le_vmax=bool(np.max(np.abs(vc)) <= vmax * (1 + 1e-12)),
             vcmd_equals_clip=float(np.max(np.abs(vc - expect))),
             summary_max_v_equals_hist=bool(np.isclose(r["max_v"], np.max(np.abs(vc)), rtol=1e-12)),
             summary_max_demand=r["max_demand"],
             max_demand_equals_unclipped_hist=bool(np.isclose(r["max_demand"],
                                                              np.max(np.abs(g * v)), rtol=1e-12)),
             max_fact=r["max_fact"], max_fact_over_3vmax=r["max_fact"] / (3 * vmax),
             fraction_of_steps_saturated=float(np.mean(np.abs(-g * v) > vmax)),
             max_x_um=r["max_x"] * 1e6, diverged=r["diverged"],
             stable=Dt.classify(r["x"], P.STEPS_PER_REV, r["diverged"])["stable"])
    if frac > 1:
        # same path up to round-off (the two branches of sat() compile differently)
        c["rel_diff_to_unlimited_x"] = float(np.max(np.abs(r["x"] - free["x"]))
                                             / np.max(np.abs(free["x"])))
        c["rel_diff_to_unlimited_fact"] = float(np.max(np.abs(r["fact"] - free["fact"]))
                                                / np.max(np.abs(free["fact"])))
    checks.append(c)
    print(c)
res["clip_checks"] = checks

# (d) independent clipped reference.  Case A: bounded, heavily clipped cut
# (a = 5 mm, vmax = 0.3 x its own peak demand); case B: the growing (unstable)
# run of (a) with vmax = 0.5 x peak, compared over its first 3 revolutions.
cmp = []
rel = lambda u, w: float(np.max(np.abs(u - w)) / np.max(np.abs(w)))
for label, a_c, frac, nr in (("A_bounded", 5e-3, 0.3, 6), ("B_growing", a, 0.5, 3)):
    pk = tdsim.simulate(rpm, a_c, g, nrev=nr)["max_demand"]
    vmax = frac * pk
    ref = R.ref_clipped(rpm, a_c, g, vmax, nrev=nr)
    for nsub in (2, 8):
        c = tdsim.simulate(rpm, a_c, g, vmax=vmax, store=True, nrev=nr, nsub=nsub)
        d = dict(case=label, a_mm=a_c * 1e3, vmax=vmax, vmax_over_peak=frac, nsub=nsub,
                 nrev=nr, diverged=c["diverged"],
                 fraction_saturated=float(np.mean(np.abs(g * ref["v"]) > vmax)),
                 rel_err_x=rel(c["x"], ref["w"]), rel_err_v=rel(c["v"], ref["v"]),
                 rel_err_fact=rel(c["fact"], ref["fact"]),
                 err_vcmd_over_vmax=float(np.max(np.abs(c["vcmd"] - ref["vcmd"])) / vmax),
                 ref_max_vcmd=float(np.max(np.abs(ref["vcmd"]))))
        cmp.append(d)
        print(d)
    if label == "A_bounded":
        long = tdsim.simulate(rpm, a_c, g, vmax=vmax, store=True, nrev=300)
        res["bounded_clipped_case_stable_300rev"] = Dt.classify(
            long["x"], P.STEPS_PER_REV, long["diverged"])["stable"]
res["vs_independent_clipped_reference"] = cmp

# (f) TD boundary vs vmax
bnd = []
for vmax in (0.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5):
    lo, hi = Dt.boundary(rpm, g, nrev=300, vmax=vmax, a_guess=0.9 * a_z, rtol=2e-3)
    r = tdsim.simulate(rpm, lo, g, vmax=vmax, nrev=300)
    bnd.append(dict(vmax=vmax, a_td_mm=0.5 * (lo + hi) * 1e3,
                    peak_demand_at_boundary=r["max_demand"]))
    print(bnd[-1])
a0 = zoa.a_lim(rpm, 0.0)
res["boundary_vs_vmax"] = dict(rpm=rpm, g=g, a_zoa_g_mm=a_z * 1e3, a_zoa_g0_mm=a0 * 1e3,
                               rows=bnd)

with open(common.out("check4_vmax.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)

# figure: the bounded, heavily clipped run of case A (a = 5 mm, vmax = 0.3 x peak)
a = 5e-3
vmax = 0.3 * tdsim.simulate(rpm, a, g, nrev=6)["max_demand"]
r = tdsim.simulate(rpm, a, g, vmax=vmax, store=True, nrev=nrev)
sl = slice(len(r["t"]) - 3 * P.STEPS_PER_REV, len(r["t"]))
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(r["t"][sl], -g * r["v"][sl], "0.6", lw=1, label="demand -g xdot")
ax[0].plot(r["t"][sl], r["vcmd"][sl], "C0", lw=1.2, label="applied V (clipped)")
ax[0].axhline(vmax, color="C3", ls=":", lw=1); ax[0].axhline(-vmax, color="C3", ls=":", lw=1)
ax[0].set_ylabel("V"); ax[0].legend(fontsize=7)
ax[1].plot(r["t"][sl], r["fact"][sl], "C1", lw=1.2, label="actuator force")
ax[1].axhline(3 * vmax, color="C3", ls=":", lw=1); ax[1].axhline(-3 * vmax, color="C3", ls=":", lw=1)
ax[1].set_ylabel("N"); ax[1].set_xlabel("t (s)"); ax[1].legend(fontsize=7)
ax[0].set_title(f"{rpm} rpm, g = {g:.0f}, a = {a*1e3:.2f} mm, vmax = {vmax:.2f} V "
                "(last 3 revolutions)", fontsize=9)
fig.tight_layout()
fig.savefig(common.out("check4_vmax.png"), dpi=130)
