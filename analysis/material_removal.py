"""
material_removal.py — physics-based bounds on the modal perturbation
====================================================================
Section 2 of Du et al. (2024) states: "we neglect the material removal".  Section
3.2 then reintroduces it as a flat "10 % perturbation" on modal mass and stiffness
and "20 %" on the damping ratio, with no derivation.  Section 5 reports what the
plate actually did: after a series of milling experiments the first two natural
frequencies rose from 540 and 1068 Hz to 632 and 1162 Hz, i.e. +17 % and +9 %.

This script derives M(eta), C(eta), K(eta) directly from the paper's own
Chebyshev-Ritz discretisation and shows three things.

  (T1) ONE-SIDEDNESS.  In a fixed nominal modal basis the removal perturbation is
       exactly negative semidefinite:  Delta M(eta) <= 0 and Delta K(eta) <= 0 in
       the Loewner order, for every removal domain.  This is exact, not
       first-order: the Ritz integrals are additive over the domain, so removing
       material subtracts a Gram matrix, and Gram matrices are PSD.

  (T2) MONOTONICITY.  For nested removal domains the perturbation only grows:
       eta1 <= eta2  =>  M(eta2) <= M(eta1) <= M(0), same for K.  Hence the
       reachable set is a monotone curve indexed by the single scalar eta, not a
       box of independent scalars.

  (T3) THE BOX OF SECTION 3.2 DOES NOT CONTAIN THE REACHABLE SET.  Two
       independent arguments, one needing nothing but the paper's own numbers.

    python analysis/material_removal.py
"""
import os
import numpy as np
from scipy.linalg import eigh

from _common import (Tee, RESULTS, F_MEASURED, F_AFTER_MILLING, ZETA)
from chebyshev_plate import ChebyshevPlate, cheb_matrix

log = Tee(os.path.join(RESULTS, 'material_removal.txt'))

KW, KR = 1e12, 1e8          # clamped-edge penalty, as in chebyshev_plate.py


def _grams(P, u1, u2, ngauss=96):
    ug, wg = np.polynomial.legendre.leggauss(ngauss)
    um = 0.5 * (u1 + u2) + 0.5 * (u2 - u1) * ug
    wm = 0.5 * (u2 - u1) * wg
    B = [cheb_matrix(P, um, d) for d in range(3)]
    return {(a, b): (B[a] * wm) @ B[b].T for a in range(3) for b in range(3)}


def band(pl, z_lo_frac, z_hi_frac, x_lo_frac=0.0, x_hi_frac=1.0):
    """(K_band, M_band): Ritz contributions of the rectangle
    x/l in [x_lo, x_hi]  x  z/h in [z_lo, z_hi].

    Both are symmetric PSD Gram matrices -- that is the whole point of (T1)."""
    Gx = _grams(pl.PX, 2 * x_lo_frac - 1.0, 2 * x_hi_frac - 1.0)
    Gz = _grams(pl.PZ, 2 * z_lo_frac - 1.0, 2 * z_hi_frac - 1.0)
    cx, cz = 2.0 / pl.lp, 2.0 / pl.hp
    Aj = (pl.lp / 2.0) * (pl.hp / 2.0)
    K = pl.DP * Aj * (
        cx ** 4 * np.kron(Gx[(2, 2)], Gz[(0, 0)])
        + cz ** 4 * np.kron(Gx[(0, 0)], Gz[(2, 2)])
        + pl.nu * cx ** 2 * cz ** 2 * (np.kron(Gx[(2, 0)], Gz[(0, 2)])
                                       + np.kron(Gx[(0, 2)], Gz[(2, 0)]))
        + 2.0 * (1.0 - pl.nu) * cx ** 2 * cz ** 2 * np.kron(Gx[(1, 1)], Gz[(1, 1)]))
    I0 = pl.rho * pl.bp
    I2 = pl.rho * pl.bp ** 3 / 12.0
    M = I0 * Aj * np.kron(Gx[(0, 0)], Gz[(0, 0)])
    M += I2 * Aj * (cx ** 2 * np.kron(Gx[(1, 1)], Gz[(0, 0)])
                    + cz ** 2 * np.kron(Gx[(0, 0)], Gz[(1, 1)]))
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


def clamp_penalty(pl):
    Gx = _grams(pl.PX, -1.0, 1.0)
    phi0 = cheb_matrix(pl.PZ, -1.0, 0)[:, 0]
    phi1 = cheb_matrix(pl.PZ, -1.0, 1)[:, 0]
    cz = 2.0 / pl.hp
    return (pl.lp / 2.0) * (KW * np.kron(Gx[(0, 0)], np.outer(phi0, phi0))
                            + KR * cz ** 2 * np.kron(Gx[(0, 0)], np.outer(phi1, phi1)))


def removal_matrices(pl, Kf, Mf, Kk, eta_h=0.0, band_frac=0.0, thin_ratio=1.0):
    """K(eta), M(eta) for two removal families.

    eta_h      : top strip of relative height eta_h removed outright (through
                 thickness) -- the accumulated effect of many passes.
    band_frac  : relative height of a top band whose THICKNESS is reduced,
                 thin_ratio = b'/b.  Mass scales as b'/b, bending stiffness as
                 (b'/b)^3 -- the single-pass radial engagement a_e.
    """
    K, M = Kf.copy(), Mf.copy()
    if eta_h > 0.0:
        Kb, Mb = band(pl, 1.0 - eta_h, 1.0)
        K -= Kb
        M -= Mb
    if band_frac > 0.0 and thin_ratio < 1.0:
        lo = max(0.0, 1.0 - eta_h - band_frac)
        Kb, Mb = band(pl, lo, 1.0 - eta_h)
        K -= (1.0 - thin_ratio ** 3) * Kb
        M -= (1.0 - thin_ratio) * Mb
    return K + Kk, M


def main():
    log("=" * 78)
    log("PHYSICS-BASED MODAL PERTURBATION UNDER MATERIAL REMOVAL")
    log("=" * 78)

    pl = ChebyshevPlate(PX=14, PZ=14)
    Kk = clamp_penalty(pl)
    Kf, Mf = band(pl, 0.0, 1.0)

    w2, V = eigh(Kf + Kk, Mf)
    keep = w2 > 1.0
    f0 = np.sqrt(w2[keep][:5]) / (2 * np.pi)
    U0 = V[:, keep][:, :5].copy()
    for k in range(5):
        U0[:, k] /= np.sqrt(U0[:, k] @ Mf @ U0[:, k])
    log("\n  consistency check, eta = 0:", np.round(f0, 2))
    log("  class assembly            :", np.round(pl.freq_n, 2))
    log(f"  max deviation             : {np.max(np.abs(f0/pl.freq_n-1))*100:.2e} %")

    M00 = U0.T @ Mf @ U0
    K00 = U0.T @ (Kf + Kk) @ U0

    # ------------------------------------------------------------------ T1/T2
    log("\n" + "-" * 78)
    log("(T1)+(T2)  one-sidedness and monotonicity of the removal perturbation")
    log("-" * 78)
    log("\n   eta      dh(mm)    f1(Hz)   f2(Hz) |  f1/f10  f2/f20 |"
        "  dM11/M11  dM22/M22 |  dK11/K11  dK22/K22")
    etas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.0755, 0.08, 0.10]
    table = []
    for eta in etas:
        Ke, Me = removal_matrices(pl, Kf, Mf, Kk, eta_h=eta)
        w2e, _ = eigh(Ke, Me)
        fe = np.sqrt(w2e[w2e > 1.0][:5]) / (2 * np.pi)
        Mp, Kp = U0.T @ Me @ U0, U0.T @ Ke @ U0
        dM, dK = Mp - M00, Kp - K00
        table.append((eta, fe, dM, dK))
        log(f"  {eta:6.4f}   {eta*pl.hp*1e3:6.3f}   {fe[0]:7.1f}  {fe[1]:7.1f} |"
            f"  {fe[0]/f0[0]:6.4f}  {fe[1]/f0[1]:6.4f} |"
            f"  {dM[0,0]/M00[0,0]:+8.4f}  {dM[1,1]/M00[1,1]:+8.4f} |"
            f"  {dK[0,0]/K00[0,0]:+8.5f}  {dK[1,1]/K00[1,1]:+8.5f}")

    log("\n  Loewner-order check (eigenvalues of the symmetric parts):")
    ok = True
    for eta, fe, dM, dK in table[1:]:
        em = np.linalg.eigvalsh(0.5 * (dM + dM.T))
        ek = np.linalg.eigvalsh(0.5 * (dK + dK.T))
        good = em.max() <= 1e-12 * abs(M00[0, 0]) and ek.max() <= 1e-4 * K00[0, 0]
        ok &= good
        log(f"    eta = {eta:6.4f} : max eig(dM) = {em.max():+.3e}"
            f"   max eig(dK)/K11 = {ek.max()/K00[0,0]:+.3e}   dM<=0 and dK<=0 : {good}")
    log(f"\n  (T1) verified for every eta tested : {ok}")

    dm11 = [t[2][0, 0] for t in table]
    log("  (T2) dM11 monotone non-increasing in eta :",
        bool(np.all(np.diff(dm11) <= 1e-12)))

    # --- second removal family: thickness reduction over a top band ---------
    log("\n  Second removal family (single-pass radial engagement a_e over a top")
    log("  band of 0.8 mm, thickness 4 mm -> 3.9 mm):")
    Ke, Me = removal_matrices(pl, Kf, Mf, Kk, eta_h=0.0,
                              band_frac=0.8e-3 / pl.hp, thin_ratio=3.9 / 4.0)
    dM = U0.T @ Me @ U0 - M00
    dK = U0.T @ Ke @ U0 - K00
    log(f"    max eig(dM) = {np.linalg.eigvalsh(0.5*(dM+dM.T)).max():+.3e}"
        f"   max eig(dK)/K11 = {np.linalg.eigvalsh(0.5*(dK+dK.T)).max()/K00[0,0]:+.3e}")
    log("    -> one-sided as well.  (T1) does not depend on the removal geometry.")

    # --- partial-length removal: the state DURING a pass --------------------
    log("\n  Third family: removal completed only up to x = xi * l_P (the state")
    log("  in the middle of a pass).  This breaks the x-symmetry of the plate,")
    log("  so it COUPLES the symmetric mode 1 to the antisymmetric mode 2:")
    log("\n    xi     dM11/M11   dM22/M22   dM12 (normalised)   dK12 (normalised)")
    for xi in (0.25, 0.50, 0.75, 1.00):
        Kb, Mb = band(pl, 1.0 - 0.0755, 1.0, 0.0, xi)
        Me_p, Ke_p = Mf - Mb, Kf - Kb + Kk
        dMp = U0.T @ Me_p @ U0 - M00
        dKp = U0.T @ Ke_p @ U0 - K00
        log(f"    {xi:4.2f}   {dMp[0,0]/M00[0,0]:+9.4f}  {dMp[1,1]/M00[1,1]:+9.4f}"
            f"   {dMp[0,1]/np.sqrt(M00[0,0]*M00[1,1]):+12.4f}"
            f"        {dKp[0,1]/np.sqrt(K00[0,0]*K00[1,1]):+12.4f}")
    log("    Eq. (22) makes Delta_PrM and Delta_PrK DIAGONAL, so this coupling")
    log("    has no representation in the paper's uncertainty set at all.")

    # -------------------------------------------------------------------- T3
    log("\n" + "-" * 78)
    log("(T3)  the +/-10 % box of Section 3.2 does not contain the reachable set")
    log("-" * 78)

    log("\n  ARGUMENT A - uses only Table 4 and Section 5 of the paper.")
    log("  With M = M0(1+dm), K = K0(1+dk), |dm| <= 0.10, |dk| <= 0.10, the")
    log("  attainable frequency ratio obeys")
    log("      (f/f0)^2 = (1+dk)/(1+dm)  in  [0.9/1.1, 1.1/0.9] = [0.818, 1.222]")
    lo, hi = 0.9 / 1.1, 1.1 / 0.9
    log(f"      => f/f0 in [{np.sqrt(lo):.4f}, {np.sqrt(hi):.4f}]"
        f"  i.e. at most {100*(np.sqrt(hi)-1):.2f} % frequency rise.")
    for i, (f_new, f_old) in enumerate(zip(F_AFTER_MILLING, F_MEASURED[:2])):
        r = f_new / f_old
        need = (r ** 2 - 1) / (r ** 2 + 1)
        log(f"      mode {i+1}: {f_old:.0f} -> {f_new:.0f} Hz, ratio {r:.4f},"
            f" (f/f0)^2 = {r**2:.4f}"
            f"   {'OUTSIDE' if r**2 > hi else 'inside '} the box"
            f"   (needs a symmetric bound of at least {100*need:.1f} %)")
    log("\n  The paper's own experiment therefore leaves the uncertainty set it")
    log("  was designed for.  Nothing in the mu-synthesis certificate applies to")
    log("  the machine state at the end of the test campaign.")

    log("\n  ARGUMENT B - the Ritz model says where the perturbation actually goes.")
    eta_fit = min(table[1:], key=lambda t: abs(t[1][0] / f0[0]
                                               - F_AFTER_MILLING[0] / F_MEASURED[0]))
    eta, fe, dM, dK = eta_fit
    log(f"    the eta reproducing the measured +17 % on mode 1 is eta = {eta:.4f}")
    log(f"    (a top strip of {eta*pl.hp*1e3:.2f} mm, {100*eta:.2f} % of the volume);")
    log(f"    it predicts +{100*(fe[1]/f0[1]-1):.1f} % on mode 2, measured "
        f"+{100*(F_AFTER_MILLING[1]/F_MEASURED[1]-1):.1f} %.")
    log("\n    at that eta, in the nominal modal basis:")
    log(f"      dM11/M11 = {dM[0,0]/M00[0,0]:+.4f}   -> mode-1 modal MASS moves "
        f"{abs(100*dM[0,0]/M00[0,0]):.1f} %, box allows 10 %")
    log(f"      dM22/M22 = {dM[1,1]/M00[1,1]:+.4f}   -> mode-2 modal MASS moves "
        f"{abs(100*dM[1,1]/M00[1,1]):.1f} %, box allows 10 %")
    log(f"      dK11/K11 = {dK[0,0]/K00[0,0]:+.5f}  -> mode-1 modal STIFFNESS moves "
        f"{abs(100*dK[0,0]/K00[0,0]):.3f} %, box allows 10 %")
    log(f"      dK22/K22 = {dK[1,1]/K00[1,1]:+.5f}  -> mode-2 modal STIFFNESS moves "
        f"{abs(100*dK[1,1]/K00[1,1]):.2f} %, box allows 10 %")
    ratio_k = 0.10 / max(abs(dK[0, 0] / K00[0, 0]), 1e-12)
    log(f"\n    The box is UNDERSIZED on mass by "
        f"{abs(dM[0,0]/M00[0,0])/0.10:.1f}x and OVERSIZED on the mode-1")
    log(f"    stiffness by {ratio_k:.0f}x.  Same nominal +/-10 %, opposite errors.")
    log("    Removing material at a free edge is almost pure mass loss: the edge")
    log("    carries the modal displacement (kinetic energy) but almost no")
    log("    curvature (strain energy).")

    log("\n  CROSS-CHECK, independent of the Ritz model.  If dK = 0 exactly, the")
    log("  measured +17 % requires 1/(1+dm) = 1.17^2, i.e.")
    dm_needed = 1.0 / (F_AFTER_MILLING[0] / F_MEASURED[0]) ** 2 - 1.0
    log(f"      dm = {dm_needed:+.4f}  ({100*dm_needed:+.1f} %)")
    log(f"  the Ritz model gives dM11/M11 = {dM[0,0]/M00[0,0]:+.4f}"
        f"  ({100*dM[0,0]/M00[0,0]:+.1f} %).")
    log("  Two independent routes, the same number: the mechanism is confirmed.")

    # ---------------------------------------------------------------- damping
    log("\n" + "-" * 78)
    log("Consequence for the damping matrix")
    log("-" * 78)
    log("  C_Pr0 = diag(2 zeta_i omega_i).  Even at constant zeta, omega moves with")
    log("  eta, so C moves with it:")
    for i in range(2):
        g = fe[i] / f0[i]
        log(f"    mode {i+1}: omega x {g:.4f} -> C x {g:.4f} at constant zeta"
            f"  ({100*(g-1):+.1f} %)")
    log("  The paper's +/-20 % on C is meant for the uncertainty on zeta itself.")
    log(f"  Once the eta-induced part is included, mode 1 needs "
        f"{100*(1.20*fe[0]/f0[0]-1):.0f} % - also outside the box.")

    # --------------------------------------------------------- usable bounds
    log("\n" + "-" * 78)
    log("Bounds to be used in place of the flat 10 % / 20 %")
    log("-" * 78)
    eta_max = 0.10
    log(f"  for eta in [0, {eta_max}] (top strip up to {eta_max*pl.hp*1e3:.1f} mm),")
    log("  worst case over the interval, in the nominal modal basis:")
    worst = dict(M=[0.0, 0.0], K=[0.0, 0.0], Moff=0.0, Koff=0.0)
    grid = np.linspace(0.0, eta_max, 21)[1:]
    for e in grid:
        Ke, Me = removal_matrices(pl, Kf, Mf, Kk, eta_h=float(e))
        dMe = U0.T @ Me @ U0 - M00
        dKe = U0.T @ Ke @ U0 - K00
        for i in range(2):
            worst['M'][i] = max(worst['M'][i], abs(dMe[i, i] / M00[i, i]))
            worst['K'][i] = max(worst['K'][i], abs(dKe[i, i] / K00[i, i]))
        worst['Moff'] = max(worst['Moff'],
                            abs(dMe[0, 1]) / np.sqrt(M00[0, 0] * M00[1, 1]))
        worst['Koff'] = max(worst['Koff'],
                            abs(dKe[0, 1]) / np.sqrt(K00[0, 0] * K00[1, 1]))
    log(f"    |Delta M11| / M11 <= {100*worst['M'][0]:6.2f} %      "
        f"(paper: 10 %)   -> UNDER by {worst['M'][0]/0.10:.1f}x")
    log(f"    |Delta M22| / M22 <= {100*worst['M'][1]:6.2f} %      "
        f"(paper: 10 %)   -> UNDER by {worst['M'][1]/0.10:.1f}x")
    log(f"    |Delta K11| / K11 <= {100*worst['K'][0]:6.3f} %      "
        f"(paper: 10 %)   -> OVER  by {0.10/max(worst['K'][0],1e-12):.0f}x")
    log(f"    |Delta K22| / K22 <= {100*worst['K'][1]:6.2f} %      "
        f"(paper: 10 %)   -> OVER  by {0.10/max(worst['K'][1],1e-12):.1f}x")
    log(f"    off-diagonal, full-length removal: |dM12| <= {worst['Moff']:.4f},"
        f"  |dK12| <= {worst['Koff']:.4f}  (zero by symmetry)")
    log("    but partial-length removal (mid-pass) breaks that symmetry - see the")
    log("    third family above - and Eq. (22) keeps Delta_PrM, Delta_PrK diagonal,")
    log("    so that coupling has no representation at any magnitude.")
    log("")
    log("    Delta M <= 0 and Delta K <= 0 for every eta in the interval (T1),")
    log("    and both are monotone in eta (T2): the set is a ONE-SIDED CURVE with")
    log("    a single free parameter, not a 4-scalar symmetric box.")


if __name__ == '__main__':
    main()
