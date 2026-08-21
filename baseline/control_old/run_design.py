"""
run_design.py — Etape 1 : synthese des trois correcteurs du papier
==================================================================
Produit results/controllers.npz :
  * K_Pmu(s)  : correcteur robuste par iteration D-K (Eqs. 26-29)
  * (K_Pp, K_Pd) : commande a retard actif (Eq. 30)
  * les diagnostics : courbe de mu, historique D-K, modes de boucle fermee.

    python run_design.py
"""
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'plant'), HERE]

import config as C
from robust_design import (build_plate, design_robust, closed_loop_modes,
                           mu_of_controller, freq_grid)
from uncertain_plant import MillingUncertainSystem
from delay_control import (pd_gains_cancellation, cancellation_ratio,
                           parasitic_coupling, suitable_delay_gain)
from closed_loop import limit

OUT = os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 74)
    print(" ETAPE 1 — SYNTHESE (Section 3 du papier)")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE)
    D_obs = plate.D_row(plate.lp, plate.hp)
    H = np.asarray(plate.H_Pe_modal, float)
    print(f"  pastille : coin {C.PATCH_SIDE};  produits modaux D_obs*H =",
          np.round(D_obs * H, 4))
    print(f"  somme = {float(D_obs @ H):+.3f}"
          "  (meme signe = paire actionneur/capteur duale)")

    print("\n  --- iteration D-K, Eq. (29) ---")
    print(f"  convention de couplage : SIGN_SIM = {C.SIGN_SIM:+.0f} "
          "(celle des figures de la Section 4 — voir config.py)")
    res = design_robust(plate, C.RPM_S, C.AP_S, wpf=C.W_PF, wpu=C.W_PU,
                        wpn=C.W_PN, n_iter=3, orders=(0, 1, 1, 1),
                        reduce_to=C.CTRL_ORDER, verbose=True, sign=C.SIGN_SIM)
    K = res['K']
    print(f"  mu = {res['mu']:.3f} ; correcteur reduit a {K.nstates} etats")

    f, z, mre = closed_loop_modes(plate, res['ss'], C.N_MODES_SIM)
    print("  boucle fermee nominale :")
    for i in range(min(4, len(f))):
        print(f"     {f[i]:7.1f} Hz   zeta = {100 * z[i]:6.2f} %")
    print(f"     (amortissements en boucle ouverte : "
          f"{np.round(100 * np.array(C.ZETA[:2]), 2)} %)")

    U = MillingUncertainSystem(plate, C.RPM_S, C.AP_S, sign=C.SIGN_SIM)
    # Eq. (31) : K_Pp est FIXE par l'annulation de la regeneration modale
    kpp_eq31, _ = pd_gains_cancellation(U, target='diagonal')
    kpp_frob, _ = pd_gains_cancellation(U, target='frobenius')
    off, reg_off = parasitic_coupling(U)
    kpp, worst, _ = suitable_delay_gain(plate, C.RPM_S, C.AP_S, sign=C.SIGN_SIM,
                                        m=C.M_FLOQUET,
                                        positions=C.POSITIONS)
    print(f"\n  --- commande a retard, Eqs. (30)-(31) ---")
    print(f"  Eq. (31), annulation complete : K_Pp = {kpp_eq31:.4g} V/m"
          f"  ({100 * cancellation_ratio(U, kpp_eq31):.1f} % de la"
          f" regeneration modale)")
    print(f"  (lecture Frobenius sur la matrice entiere : {kpp_frob:.4g} V/m,"
          f" soit {100 * cancellation_ratio(U, kpp_frob):.1f} % — a eviter)")
    print(f"  couplage retarde PARASITE hors diagonale, par volt :"
          f" {off[0]:+.3f} / {off[1]:+.3f}")
    print(f"     a K_Pp = {kpp_eq31:.3g} : {kpp_eq31 * off[0]:+.0f} et"
          f" {kpp_eq31 * off[1]:+.0f}  contre une regeneration croisee de"
          f" seulement {reg_off:+.0f}")
    print(f"  -> K_Pp REALISABLE ('designed suitably') = {kpp:.4g} V/m"
          f"  ({100 * cancellation_ratio(U, kpp):.1f} % d'annulation ;"
          f" pire taux du retard seul {worst:+.1f} s^-1)")
    print(f"  K_Pd = {C.K_PD:.4g} V/(m/s)  (libre : l'Eq. 31 ne fixe que K_Pp)")

    kw = dict(n_modes=C.N_MODES_SIM, m=C.M_FLOQUET, n_period=20, hi=3e-3,
              tol=2e-5, coeff_scale=C.SIGN_SIM)
    lims = {}
    for name, ctrl, pd in [('robuste', res['ss'], None),
                           ('combine', res['ss'], (kpp, C.K_PD)),
                           ('retard seul', None, (kpp, C.K_PD))]:
        L = np.array([limit(plate, C.RPM_S, p * plate.lp, ctrl=ctrl, pd=pd,
                            **kw) for p in C.POSITIONS])
        lims[name] = L
        print(f"  limite {name:12s} : {np.round(L * 1e3, 3)} mm"
              f"   min = {L.min() * 1e3:.3f} mm")

    mus = res['mus']
    np.savez(os.path.join(OUT, 'controllers.npz'),
             A=res['ss'][0], B=res['ss'][1], C=res['ss'][2], D=res['ss'][3],
             A_full=np.array(res['K_full'].A), B_full=np.array(res['K_full'].B),
             C_full=np.array(res['K_full'].C), D_full=np.array(res['K_full'].D),
             K_Pp=kpp, K_Pd=C.K_PD, kpp_frobenius=kpp_frob, kpp_eq31=kpp_eq31,
             cancel=cancellation_ratio(U, kpp),
             mu=res['mu'], mus=mus, f_mu=res['f'], gamma=res['gamma'],
             dk_gamma=[h['gamma'] for h in res['history']],
             dk_mu=[h['mu'] for h in res['history']],
             cl_freq=f, cl_zeta=z,
             lim_robust=lims['robuste'], lim_combined=lims['combine'],
             lim_delay=lims['retard seul'],
             positions=np.array(C.POSITIONS))
    print(f"\n  -> results/controllers.npz   ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
