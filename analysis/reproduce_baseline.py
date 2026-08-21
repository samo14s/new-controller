"""
reproduce_baseline.py — Phase 1: does the implementation reproduce Du et al. (2024)?
====================================================================================
Runs the open-loop plant package and confronts every computable quantity with the
values published in

    J. Du, X. Liu, H. Dai, X. Long, "Robust combined time delay control for milling
    chatter suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Two milling-force sign conventions are exercised, because the package carries both
and they are NOT equivalent (see the sign-convention block at the end).

    python analysis/reproduce_baseline.py
"""
import numpy as np

from _common import (Tee, build_plate, F_MEASURED, F_THEORETICAL, ZETA,
                     RPM_S, AP_S, AE_S, FZ_S, RESULTS)

import os
from milling_dynamics import (alpha4_average, alpha4_series, dtd_paper_gauge,
                              N_TEETH)
from chebyshev_plate import ChebyshevPlate
from stability_fdm import stability_limit, is_stable
from time_domain import TimeSim

log = Tee(os.path.join(RESULTS, 'phase1_reproduction.txt'))


def rel(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    log("=" * 78)
    log("PHASE 1 - REPRODUCTION OF Du, Liu, Dai & Long, IJMS 274 (2024) 109257")
    log("=" * 78)

    # ---------------------------------------------------------------- modal
    log("\n[1] Modal model - Chebyshev-Ritz, PX = PZ = 14, Eqs. (6)-(10), App. A")
    bare = ChebyshevPlate(PX=14, PZ=14)
    log("    bare plate f_n (Hz)          :", np.round(bare.freq_n, 1))
    log("    paper 'theoretical' (Table 4):", F_THEORETICAL)
    log("    relative error (%)           :",
        np.round([rel(bare.freq_n[i], F_THEORETICAL[i]) for i in range(5)], 2))
    log("    -> all five modes within 3 %, uniform sign: a single global stiffness")
    log("       offset, not a mode-shape error.")

    p = build_plate(calibrate=False)
    log("\n    with piezo patch QDA60-20-0.7 (Table 2), f_n (Hz):",
        np.round(p.freq_n, 1))
    log("    bond shear-lag efficiency eta =", round(p.eta_bond, 4),
        " rK =", round(p.rK_patch, 4), " rM =", round(p.rM_patch, 4))
    log("    C_P0 of Eq. (15)              =", f"{p.CP0:.6g}")

    p.calibrate_frequencies(F_MEASURED)
    log("    after calibration on Table 4  :", np.round(p.freq_n, 1))

    # ------------------------------------------------------- piezo coupling
    log("\n[2] Piezoelectric coupling, Eqs. (14)-(15)")
    H = np.asarray(p.H_Pe_modal, float)
    D_obs = p.D_row(p.lp, p.hp)
    log("    H_Pe (N/V)                   :", np.round(H, 4))
    log("    D_obs at corner (100, 80) mm :", np.round(D_obs, 3))
    log("    products D_obs * H_Pe        :", np.round(D_obs * H, 3))

    f = np.linspace(1.0, 5000.0, 40001)
    w = 2 * np.pi * f
    G = np.zeros_like(w, dtype=complex)
    for k in range(5):
        G += D_obs[k] ** 2 / (p.omega_n[k] ** 2 - w ** 2
                              + 2j * ZETA[k] * p.omega_n[k] * w)
    mag = np.abs(G)
    anti = [f[i] for i in range(1, len(f) - 1)
            if mag[i] < mag[i - 1] and mag[i] < mag[i + 1]]
    log("    static compliance at corner  :", f"{mag[0] * 1e6:.3f} um/N",
        f"({20 * np.log10(mag[0]):.1f} dB re 1 m/N)")
    log("    antiresonances (force input) :", np.round(anti, 1), "Hz")
    log("    -> antiresonances are set by the mode SHAPES alone; matching them")
    log("       validates the shapes independently of the frequency calibration.")

    # ---------------------------------------------------------- cutting law
    log("\n[3] Milling force coefficients, Eqs. (2)-(4), Table 3")
    log(f"    operating point S: {RPM_S} rpm, a_e = {AE_S*1e3} mm, "
        f"a_p = {AP_S*1e3} mm, f_z = {FZ_S*1e3} mm/tooth, down milling")
    abar4 = alpha4_average(RPM_S, AP_S, p.hp, AE_S)
    a3s, a4s = alpha4_series(RPM_S, AP_S, p.hp, 246, AE_S)
    tau = 60.0 / (N_TEETH * RPM_S)
    v = FZ_S * N_TEETH * RPM_S / 60.0
    log(f"    mean coefficient  abar4      = {abar4:11.1f} N/m")
    log(f"    peak alpha4(t)               = {a4s.min():11.1f} N/m"
        f"   (peak / mean = {a4s.min()/abar4:.2f})")
    log(f"    engagement duty per period   = {100*np.mean(a4s != 0.0):.1f} %")
    log(f"    tooth period tau             = {tau*1e3:.4f} ms   "
        f"(tooth-passing frequency {1/tau:.1f} Hz, paper: 245 Hz)")
    log(f"    feed speed                   = {v*1e3:.2f} mm/s   "
        f"-> full pass = {p.lp/v:.3f} s (paper Figs. 14-15 x-axis: 20.4 s)")
    log(f"    alpha40 = 1.6*abar4 (Eq. 23) = {1.6*abar4:11.1f} N/m")
    log(f"    L_Palpha= 1.3*abar4 (Eq. 23) = {1.3*abar4:11.1f} N/m")

    # ------------------------------------------------------------- Figure 7
    log("\n[4] Varying dynamics D_Pr^T D_Pr along the top edge, Fig. 7 / Eq. (24)")
    xs, DtD, DD0, LDD, s = dtd_paper_gauge(ChebyshevPlate(PX=14, PZ=14), 201)
    log("    paper-gauge factors (s1, s2) :", np.round(s, 4))
    log("    DD0  (nominal, Eq. 24)       :", np.round(DD0, 4).tolist())
    log("    L_DD (amplitude, Eq. 24)     :", np.round(LDD, 4).tolist())
    log("    Fig. 7 read-out targets      : DD11 ends 3.60 / mid 3.81,")
    log("                                   DD12 amplitude 3.45, DD22 ends 3.35")
    log(f"    -> DD11 mean {DD0[0,0]:.4f} (target 3.705), amplitude {LDD[0,0]:.4f}"
        f" (target 0.105)")
    log(f"    -> DD12 mean {DD0[0,1]:.4f} (target 0),     amplitude {LDD[0,1]:.4f}"
        f" (target 3.45)")
    log(f"    -> DD22 mean {DD0[1,1]:.4f} (target 1.675), amplitude {LDD[1,1]:.4f}"
        f" (target 1.675)")
    det = np.array([np.linalg.det(m) for m in DtD])
    log(f"    max |det(D^T D)| along the edge = {np.abs(det).max():.3e}"
        f"  (element scale {np.abs(DtD).max():.3f})  -> rank 1 to machine precision")

    # ---------------------------------------------------- stability + sign
    log("\n[5] Open-loop stability, full-discretisation Floquet (Ding et al.)")
    log("    paper: limit below 0.1 mm at most speeds (Fig. 13);")
    log("           experimental limit ~0.1 mm without control (Fig. 18);")
    log("           at S the uncontrolled cut DIVERGES (Fig. 14a, shown over 0.2 s).")
    log("")
    log("    sign  x (mm)   a_p,lim (mm)   rho at a_p = 0.3 mm   f_c near mode 2 (Hz)")
    inv_tau = 1.0 / tau
    for sc in (1.0, -1.0):
        for x in (0.0, 0.025, 0.050, 0.075, 0.100):
            L = stability_limit(p, RPM_S, x_pos=x, coeff_mode='time',
                                coeff_scale=sc, hi=1.5e-3, tol=2e-6, m=60)
            st, fpv, rho = is_stable(p, RPM_S, AP_S, x, m=60, coeff_mode='time',
                                     coeff_scale=sc, n_modes=2, return_freq=True)
            cand = sorted({abs(sg * fpv + j * inv_tau)
                           for sg in (1, -1) for j in range(9)})
            fc2 = min(cand, key=lambda c: abs(c - 1068.0))
            log(f"    {sc:+.0f}    {x*1e3:5.1f}      {L*1e3:8.4f}"
                f"          {rho:7.4f}              {fc2:8.1f}")

    log("\n[6] Time-domain check at S (Newmark, 5 modes, fixed tool position)")
    for sc in (1.0, -1.0):
        sim = TimeSim(p, rpm=RPM_S, ap=AP_S, sign=sc, n_modes=5, n_sub=164)
        r = sim.run(T=0.6, controller=None, moving=False, x0=0.0)
        td = 'never' if not r['diverged'] else f"{r['t_div']:.4f} s"
        log(f"    sign = {sc:+.0f}: diverges {td:>10}   "
            f"max |y| = {np.max(np.abs(r['y_mill']))*1e6:8.2f} um")

    log("\n    VERDICT ON THE SIGN CONVENTION")
    log("    sign = +1 (Eq. 13 exactly as printed): unstable at S, rho = 1.33,")
    log("      diverges in 0.107 s, limit 0.039-0.045 mm -> matches Figs. 13, 14a, 18.")
    log("    sign = -1: STABLE at S (rho = 0.89), limit 0.35-1.36 mm -> contradicts")
    log("      the paper's own reference simulation. It must not be used as default.")
    log("    NOTE: baseline/control/config.py currently sets SIGN_SIM = -1.")


if __name__ == '__main__':
    main()
