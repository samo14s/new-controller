"""
uncertainty_audit.py — audit of the uncertainty description of Du et al. (2024)
===============================================================================
Two results, both exact and independent of any experiment.

RESULT 1 (algebraic).  Eq. (25) bounds the perturbation of the product
alpha4(t) * D_Pr^T D_Pr by  L_Palpha * L_DD.  The exact envelope of a product of
two intervals is

    a in [a0 - La, a0 + La],  d in [d0 - Ld, d0 + Ld]
    =>  a d  in  a0 d0  +/-  ( |a0| Ld + |d0| La + La Ld ).

Eq. (25) keeps only the third, smallest, term.  The two cross terms it drops are
the dominant ones, so Eq. (25) UNDER-bounds the true perturbation.  A mu-synthesis
design carried out on that box is not robust to the uncertainty it claims to cover.

RESULT 2 (structural).  Eq. (24) gives the four entries of Delta_DD four
independent scalars delta_DD1..4, |delta| <= 1.  The reachable set is
{ alpha4 * D(x)^T D(x) }, which is symmetric, rank 1 and sign-definite.  The box
is a 4-dimensional object around a 2-dimensional set, and it contains matrices
(non-symmetric, full-rank, indefinite) that the physics can never produce.

    python analysis/uncertainty_audit.py
"""
import os
import numpy as np

from _common import Tee, RESULTS, RPM_S, AP_S, AE_S
from chebyshev_plate import ChebyshevPlate
from milling_dynamics import alpha4_average, dtd_paper_gauge

log = Tee(os.path.join(RESULTS, 'uncertainty_audit.txt'))


def main():
    log("=" * 78)
    log("AUDIT OF Eqs. (22)-(25) - UNCERTAINTY DESCRIPTION")
    log("=" * 78)

    bare = ChebyshevPlate(PX=14, PZ=14)
    xs, DtD, DD0, LDD, s = dtd_paper_gauge(bare, 201)
    abar4 = alpha4_average(RPM_S, AP_S, bare.hp, AE_S)
    a40, LPa = 1.6 * abar4, 1.3 * abar4          # Eq. (23)
    lo, hi = 0.3 * abar4, 2.9 * abar4

    log(f"\noperating point S: {RPM_S} rpm, a_p = {AP_S*1e3} mm, a_e = {AE_S*1e3} mm")
    log(f"abar4 = {abar4:.1f} N/m,  alpha40 = {a40:.1f},  L_Palpha = {LPa:.1f}")
    log("DD0  =", np.round(DD0, 4).tolist())
    log("L_DD =", np.round(LDD, 4).tolist())

    # ------------------------------------------------------------- RESULT 1
    log("\n" + "-" * 78)
    log("RESULT 1 - Eq. (25) drops the cross terms of an interval product")
    log("-" * 78)

    L_paper = np.abs(LPa) * LDD                                     # Eq. (25)
    L_formula = (np.abs(a40) * LDD + np.abs(DD0) * abs(LPa)
                 + abs(LPa) * LDD)                                  # exact
    prod = np.concatenate([lo * DtD, hi * DtD], axis=0)
    L_sampled = np.max(np.abs(prod - a40 * DD0), axis=0)            # sampled

    log("\n  nominal alpha40 * DD0 :", np.round(a40 * DD0, 1).tolist())
    log("  Eq. (25)  L_PD        :", np.round(L_paper, 1).tolist())
    log("  exact formula         :", np.round(L_formula, 1).tolist())
    log("  sampled over the edge :", np.round(L_sampled, 1).tolist())
    log("  formula vs sampled, max relative difference : "
        f"{np.max(np.abs(L_formula/np.maximum(L_sampled,1e-30)-1))*100:.4f} %")
    ratio = L_formula / np.maximum(L_paper, 1e-30)
    log("\n  UNDER-BOUNDING FACTOR (exact / Eq. 25) :")
    log("      element (1,1) :", f"{ratio[0,0]:8.2f} x")
    log("      element (1,2) :", f"{ratio[0,1]:8.2f} x")
    log("      element (2,1) :", f"{ratio[1,0]:8.2f} x")
    log("      element (2,2) :", f"{ratio[1,1]:8.2f} x")
    log("\n  Term-by-term for element (1,1), the worst one:")
    log(f"      |alpha40| * L_DD1        = {abs(a40)*LDD[0,0]:12.1f}   (dropped)")
    log(f"      |DD0_11|  * L_Palpha     = {DD0[0,0]*abs(LPa):12.1f}   (dropped)")
    log(f"      L_Palpha  * L_DD1        = {abs(LPa)*LDD[0,0]:12.1f}   (kept - Eq. 25)")
    log(f"      exact envelope           = {L_formula[0,0]:12.1f}")
    log("\n  The (1,1) entry is the one the paper singles out as nearly constant")
    log("  ('Except for the first element ... other elements all vary largely').")
    log("  Its perturbation is small because L_DD1 is small - but alpha4(t) still")
    log("  multiplies the LARGE nominal DD0_11, and that term is missing from")
    log("  Eq. (25).  The regenerative stiffness of mode 1 is therefore left")
    log(f"  {ratio[0,0]:.0f} times under-covered.")

    # ------------------------------------------------------------- RESULT 2
    log("\n" + "-" * 78)
    log("RESULT 2 - the box is 4-dimensional, the reachable set is 2-dimensional")
    log("-" * 78)
    det = np.array([np.linalg.det(m) for m in DtD])
    log(f"\n  max |det(D^T D)| over 201 positions = {np.abs(det).max():.3e}"
        f"   (entry scale {np.abs(DtD).max():.3f})")
    log("  -> D^T D is an outer product: rank 1, symmetric, positive semidefinite,")
    log("     for every milling position.  One free parameter (x), plus one for")
    log("     alpha4.  The reachable set of alpha4 * D^T D is a 2-D surface.")
    log("\n  Eq. (24) instead uses four independent reals delta_DD1..delta_DD4:")
    log("     * symmetry broken  : delta_DD2 and delta_DD3 are independent, so")
    log("       the box contains non-symmetric 'stiffness' matrices;")
    log("     * rank broken      : det is generically non-zero in the box;")
    log("     * sign broken      : see the corner scan below.")

    eigs = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                for s4 in (-1, 1):
                    Mx = a40 * DD0 + np.array(
                        [[s1 * L_paper[0, 0], s2 * L_paper[0, 1]],
                         [s3 * L_paper[1, 0], s4 * L_paper[1, 1]]])
                    eigs.append(np.linalg.eigvals(0.5 * (Mx + Mx.T)).real)
    eigs = np.array(eigs)
    indef = np.any(eigs.min(axis=1) * eigs.max(axis=1) < 0)
    log(f"\n  16 corners of the Eq.(25) box: eigenvalues span "
        f"[{eigs.min():.1f}, {eigs.max():.1f}]")
    log(f"  indefinite corners present : {bool(indef)}")
    tr = np.array([np.linalg.eigvals(a40 * m).real for m in DtD])
    log(f"  reachable set eigenvalues  : one is exactly 0, the other lies in "
        f"[{tr.min():.1f}, {tr.max():.1f}]")
    log("  -> every reachable matrix is semidefinite of ONE sign; the box spends")
    log("     part of its volume on matrices of the opposite sign, which no")
    log("     milling position can produce.  That volume is pure conservatism,")
    log("     and it is paid for in mu.")

    log("\n" + "-" * 78)
    log("CONSEQUENCE")
    log("-" * 78)
    log("  The set used for synthesis is simultaneously")
    log("     TOO SMALL  in the direction that matters (Result 1: the")
    log(f"                regenerative stiffness of mode 1 is {ratio[0,0]:.0f}x under-covered),")
    log("     TOO LARGE  in directions the physics cannot reach (Result 2).")
    log("  Enlarging the box uniformly fixes the first and worsens the second.")
    log("  The fix is structural: parameterise the set by the physical variables")
    log("  (position x, coefficient alpha4) instead of by four free scalars.")


if __name__ == '__main__':
    main()
