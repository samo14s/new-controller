"""Analyses requested by the internal review of the manuscript (paper/notes/
review_triage.json), stored so that every number added to the paper has a file.

A. The two published cuts that the linear model misclassifies (Ozsoy et al.
   2026, controlled, chattered below the predicted limit): 1300 rpm / 10 mm at
   g = 1157.98 V s/m and 2800 rpm / 5.5 mm at g = 679.55 V s/m.  Closed-form
   forced command, clipped step and face entries, and the 16-phase critical
   kick with the 9 V clip and the full surface memory (C1 protocol).
B. Certified kick beta against the collocation order N (C2 and DVF + D).
C. Certified kick against the validity bound delta (C2).
D. Eccentric runout at which the forced command of the g = 1500 designs
   reaches 9 V at some depth up to 0.9 a_lim (closed form, 8 runout phases).
E. From stored results: free voltage at a_lim with an 8.1 V clip; depths
   beyond 2D = 32 mm for DVF and DVF + D (g2 = 2000); displacement
   equivalents of the worst-phase critical kicks at 0.98 a_lim.

    OMP_NUM_THREADS=1 python3 verification/paper/revision_runs.py
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RES = os.path.join(HERE, "..", "..", "results")
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
CUTS = [(1300.0, 10e-3, 1157.98), (2800.0, 5.5e-3, 679.55)]


def job(args):
    import params as P
    P.set_damping(ZETA)
    kind = args[0]
    if kind == "kick":
        _, rpm, a, g, ph = args
        import c1_unsafe_zone as C
        return args, float(C._critical_kick((rpm, a, g, C.VMAX, ph, ZETA)))
    if kind == "entry":
        _, rpm, a, g, entry = args
        from tdboundary import classify
        import c1_unsafe_zone as C
        opts = dict(step={}, face=dict(edge=True))[entry]
        c = classify(rpm, a, g, C.VMAX, nreg=0, **opts)
        return args, dict(chatter=bool(c["chatter"]), max_x_mm=float(c["run"]["max_x"] * 1e3))
    import c2_certificate as c2
    from zoa import a_lim
    if kind == "nconv":
        _, rpm, frac, N = args
        a = frac * a_lim(rpm, 1500.0)
        beta, note = c2.certify_kick(rpm, a, 1500.0, N=N)
        return args, dict(beta_mmps=beta * 1e3, note=note)
    if kind == "nconv_comb":
        _, rpm, a, g1, g2, N = args
        import c4_comb as c4
        A, B, K = c4.model(rpm, a, g1, g2, N=N)
        _, _, bmax = c2.orbit(rpm, g1, a, N)
        e = np.zeros(len(B)); e[1] = 1.0
        beta, note = c2.certify(A, B, K, c2.VMAX - bmax, e, rpm, c2.TIGHT)
        return args, dict(beta_mmps=beta * 1e3, note=note)
    if kind == "delta":
        _, rpm, frac, d = args
        a = frac * a_lim(rpm, 1500.0)
        beta, note = c2.certify_kick(rpm, a, 1500.0, delta=d)
        return args, dict(beta_mmps=beta * 1e3, note=note)
    raise ValueError(kind)


def closed_form_parts():
    import params as P
    P.set_damping(ZETA)
    from zoa import a_lim
    from feasibility import forced_velocity
    import c4_robust as R
    out = {}
    # A: forced command and impulse scale at the two published cuts
    cuts = []
    for rpm, a, g in CUTS:
        al = a_lim(rpm, g)
        b = -g * forced_velocity(rpm, g, [a])[0]
        Fy = P.N_TEETH * a * P.FZ * (np.pi / 4 * P.K_TC + 0.5 * P.K_RC) / (2 * np.pi)
        T = 60.0 / (rpm * P.N_TEETH)
        cuts.append(dict(rpm=rpm, a_mm=a * 1e3, g=g, a_lim_mm=al * 1e3, frac=a / al,
                         bmax_V=float(np.abs(b).max()), tooth_impulse_Ns=Fy * T, mass=P.M))
    out["published_cuts"] = cuts
    # D: runout at which the worst forced command reaches 9 V (g = 1500, up to 0.9 a_lim)
    rr = {}
    for rpm in (1090.0, 1420.0, 2500.0, 2870.0):
        al = a_lim(rpm, 1500.0)
        depths = np.linspace(0.9 * al / 40, 0.9 * al, 40)
        f = lambda r: R.worst_orbit_command(rpm, depths, 1500.0, runout=r)
        c0 = f(0.0)
        hi = 10e-6
        if f(hi) < 9.0:
            rr[str(int(rpm))] = dict(cmd0_V=c0, runout_um=None, note="below 9 V up to 10 um")
            continue
        lo = 0.0
        for _ in range(30):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if f(m) < 9.0 else (lo, m)
        rr[str(int(rpm))] = dict(cmd0_V=c0, runout_um=hi * 1e6)
    out["runout_to_clip_g1500"] = rr
    return out


def stored_parts():
    out = {}
    df = np.load(os.path.join(RES, "c1_df.npz"))
    bmax_lim = 9.0 - df["A_dual"][:, -1]
    out["free_voltage_at_alim"] = {str(int(n)): dict(bmax_V=float(b), free_9V=float(9.0 - b), free_8p1V=float(8.1 - b))
                                   for n, b in zip(df["rpm"], bmax_lim)}
    L = np.load(os.path.join(RES, "c4_limits.npz"))
    names = [str(s) for s in L["feas_names"]]
    al = L["feas_a_lim"] * 1e3
    i_dvf, i_d2 = names.index("dvf"), names.index("d2000")
    cap = 32.0
    out["depth_2D"] = dict(names=names, n_speeds=int(al.shape[0]),
                           dvf_above_32mm=int(np.sum(al[:, i_dvf] > cap)),
                           d2000_above_32mm=int(np.sum(al[:, i_d2] > cap)),
                           median_ratio=float(np.median(al[:, i_d2] / al[:, i_dvf])),
                           median_ratio_capped_32mm=float(np.median(np.minimum(al[:, i_d2], cap) / np.minimum(al[:, i_dvf], cap))),
                           max_dvf_mm=float(al[:, i_dvf].max()), max_d2000_mm=float(al[:, i_d2].max()))
    b = np.load(os.path.join(RES, "c1_basin.npz"))
    import params as P
    wn = 2 * np.pi * 129.3
    rows = []
    for n in (1090.0, 1420.0, 2500.0, 2870.0):
        m = (b["rpm"] == n) & (b["vmax"] > 0) & np.isclose(b["frac"], 0.98)
        lo = float(np.min(b["dv"][m]))
        rows.append(dict(rpm=n, dv_worst_mmps=lo * 1e3, displacement_um=lo / wn * 1e6))
    out["kick_displacement_098"] = rows
    return out


def main():
    t0 = time.time()
    jobs = [("kick", n, a, g, ph) for n, a, g in CUTS for ph in range(16)]
    jobs += [("entry", n, a, g, e) for n, a, g in CUTS for e in ("step", "face")]
    jobs += [("nconv", 2500.0, 0.9, N) for N in (20, 30, 40)]
    jobs += [("nconv", 1090.0, 0.9, N) for N in (24, 36, 48)]
    jobs += [("nconv", 2870.0, 0.3, N) for N in (20, 30, 40)]
    jobs += [("nconv_comb", 2500.0, 22e-3, 500.0, 4000.0, N) for N in (20, 30, 40)]
    jobs += [("delta", n, f, d) for n, f in ((1090.0, 0.15), (2870.0, 0.3), (2500.0, 0.9)) for d in (0.25, 0.5, 1.0, None)]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    out = dict(note=__doc__, closed_form=closed_form_parts(), stored=stored_parts())
    kicks = {}
    for (args, v) in res:
        if args[0] == "kick":
            kicks.setdefault(f"{args[1]:.0f}", []).append(v)
    out["published_cuts_kicks_mps"] = kicks
    out["published_cuts_entries"] = [dict(rpm=a[1], a_mm=a[2] * 1e3, g=a[3], entry=a[4], **v) for a, v in res if a[0] == "entry"]
    out["beta_vs_N"] = [dict(kind=a[0], args=[x for x in a[1:]], **v) for a, v in res if a[0] in ("nconv", "nconv_comb")]
    out["beta_vs_delta"] = [dict(rpm=a[1], frac=a[2], delta=a[3], **v) for a, v in res if a[0] == "delta"]
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "revision_runs.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
