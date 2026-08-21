"""
run_stability.py — Etape 3 : cartes de stabilite (Figs. 6, 13, 18)
===================================================================
  Fig. 6  : lobes en boucle ouverte avec alpha4(t) exact et 0.3, 1, 2.9 fois
            le coefficient moyen, aux positions depart / 1/4 / 1/2 (Eq. 23).
            Convention SIGN_EXP (+1) : c'est elle qui reproduit la forme des
            Figs. 6/13(b)/18 (creux a 4900/6700, pic a 5500).
  Fig. 13 : (a) surface de stabilite boucle ouverte position x vitesse,
            (b) minimum sur toutes les positions.
  Fig. 18 : limite de coupe experimentale reconstituee : sans commande, avec
            commande robuste, avec commande robuste combinee, aux cinq
            vitesses testees par le papier.
  + robustesse aux derives modales constatees Section 5 (+17 %, +9 %).

    python run_stability.py [--quick]
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
from robust_design import build_plate
from stability_fdm import stability_limit
from closed_loop import limit, chatter_frequency

OUT = os.path.join(HERE, '..', 'results')


def main(quick=False):
    t00 = time.time()
    print("=" * 74)
    print(" ETAPE 3 — CARTES DE STABILITE (Figs. 6, 13, 18)")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE)
    d = np.load(os.path.join(OUT, 'controllers.npz'))
    ss = (d['A'], d['B'], d['C'], d['D'])
    pd = (float(d['K_Pp']), float(d['K_Pd']))
    store = {}

    # ---------------- Fig. 6 : lobes selon le coefficient de coupe --------
    speeds6 = np.arange(3000, 7001, 100 if not quick else 250)
    for pos, tag in [(0.0, 'start'), (0.25, 'quarter'), (0.5, 'half')]:
        cols = {}
        for label, mode, sc in [('a4(t)', 'time', C.SIGN_EXP),
                                ('0.3', 'avg', 0.3 * C.SIGN_EXP),
                                ('1.0', 'avg', 1.0 * C.SIGN_EXP),
                                ('2.9', 'avg', 2.9 * C.SIGN_EXP)]:
            cols[label] = np.array([
                stability_limit(plate, r, pos * plate.lp, hi=1.5e-3, tol=5e-6,
                                coeff_mode=mode, coeff_scale=sc)
                for r in speeds6])
        store[f'fig6_{tag}'] = dict(rpm=speeds6, **cols)
        print(f"  Fig.6 {tag:8s} termine ({time.time() - t00:.0f} s)", flush=True)

    # ---------------- Fig. 13 : surface boucle ouverte --------------------
    speeds = np.arange(3000, 7001, 100 if not quick else 250)
    xs = np.linspace(0.0, plate.lp, 11 if not quick else 5)
    S = np.zeros((len(xs), len(speeds)))
    for i, x in enumerate(xs):
        S[i] = [stability_limit(plate, r, x, hi=2e-3, tol=5e-6,
                                coeff_mode='time', coeff_scale=C.SIGN_EXP)
                for r in speeds]
        print(f"  Fig.13 position {x * 1e3:5.1f} mm ({time.time() - t00:.0f} s)",
              flush=True)
    store['fig13'] = dict(rpm=speeds, x=xs, S=S, lowest=S.min(axis=0))

    # ---------------- Fig. 18 : limites en boucle fermee ------------------
    # Fig. 18 sous la convention du correcteur (SIGN_SIM) ; la courbe 'sans'
    # est aussi donnee sous SIGN_EXP (celle qui reproduit l'experience).
    kw = dict(n_modes=C.N_MODES_SIM, m=C.M_FLOQUET, n_period=20, hi=2.5e-3,
              tol=1e-5, coeff_scale=C.SIGN_SIM)
    pd_alone = pd          # Eq. (30) avec le gain de l'Eq. (31)
    res = {}
    for name, ctrl, p_, sgn in [('sans', None, None, C.SIGN_SIM),
                                ('sans_exp', None, None, C.SIGN_EXP),
                                ('robuste', ss, None, C.SIGN_SIM),
                                ('combine', ss, pd, C.SIGN_SIM),
                                ('retard', None, pd_alone, C.SIGN_SIM)]:
        vals = []
        for rpm in C.RPM_EXP:
            if ctrl is None and p_ is None:
                L = [stability_limit(plate, rpm, f * plate.lp, hi=1.5e-3,
                                     tol=5e-6, coeff_mode='time',
                                     coeff_scale=sgn)
                     for f in C.POSITIONS]
            else:
                L = [limit(plate, rpm, f * plate.lp, ctrl=ctrl, pd=p_, **kw)
                     for f in C.POSITIONS]
            vals.append(min(L))
            print(f"  Fig.18 {name:8s} {rpm} tr/min : {min(L) * 1e3:.3f} mm"
                  f" ({time.time() - t00:.0f} s)", flush=True)
        res[name] = np.array(vals)
    store['fig18'] = dict(rpm=np.array(C.RPM_EXP), **res)
    print("  moyennes : " + "  ".join(
        f"{k} = {v.mean() * 1e3:.2f} mm" for k, v in res.items()))

    # ---------------- frequences de broutement ----------------------------
    fc = {}
    for name, ctrl, p_, ap in [('sans', None, None, 0.35e-3),
                               ('retard', None, pd_alone, 0.70e-3),
                               ('robuste', ss, None, 1.00e-3),
                               ('combine', ss, pd, 1.50e-3)]:
        f0, _ = chatter_frequency(plate, C.RPM_S, ap, 0.0, ctrl=ctrl, pd=p_,
                                  n_modes=C.N_MODES_SIM, m=C.M_FLOQUET,
                                  coeff_scale=C.SIGN_SIM)
        fc[name] = np.array(f0[:2])
        print(f"  broutement {name:8s} (ap={ap * 1e3:.2f} mm) : "
              f"{np.round(f0[:2], 0)} Hz")
    store['chatter'] = fc

    # -------- robustesse a la derive modale constatee Section 5 -----------
    drift = {}
    for tag, f1, f2 in [('origine', 540.0, 1068.0), ('apres essais', 632.0,
                                                     1162.0)]:
        pl = build_plate(C.PATCH_SIDE, calibrate=False)
        pl.calibrate_frequencies([f1, f2] + C.F_MEASURED[2:])
        L = [limit(pl, C.RPM_S, f * pl.lp, ctrl=ss, pd=pd, **kw)
             for f in C.POSITIONS]  # kw porte deja coeff_scale=SIGN_SIM
        drift[tag] = np.array(L)
        print(f"  derive modale [{tag:12s}] f1={f1:.0f} f2={f2:.0f} : "
              f"limite combinee min = {min(L) * 1e3:.3f} mm")
    store['drift'] = drift

    np.savez_compressed(os.path.join(OUT, 'stability.npz'),
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           for kk, vv in v.items()})
    print(f"\n  -> results/stability.npz   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
