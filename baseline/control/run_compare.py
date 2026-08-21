"""
run_compare.py — Etape 2 : evaluation equitable des deux correcteurs optimises
==============================================================================
Tout est calcule a PLEINE resolution (Floquet m = 200), sur les MEMES grilles,
pour la boucle ouverte, le FOPID et l'ADRC-FOPID :

  1. lobes de stabilite : a_p,lim (minimum sur tout le bord superieur) en
     fonction de la vitesse de broche ;
  2. limites par position a la vitesse de synthese ;
  3. reponses temporelles a une profondeur ou la boucle ouverte broute :
     deplacement, tension, spectres ;
  4. robustesse : derive modale constatee au Section 5 du papier (+17 %, +9 %),
     amortissement reduit a 80 %, et verification sur 5 modes alors que la
     synthese n'en connait que 2 ;
  5. metriques frequentielles : marge de module, effort, Bode des correcteurs.

    python run_compare.py
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
from plate_model import build_plate, plant_vectors, plant_frf
from fopid import ss_frf
from objective import limits, frequency_metrics, nominal_poles
from simulate import MillingSimulation, amplitude_spectrum, mean_abs_amplitude
from sim_controller import LTIController

OUT = os.path.join(HERE, '..', 'results')


def load():
    d = np.load(os.path.join(OUT, 'pso.npz'), allow_pickle=True)
    out = {}
    for k in ('fopid', 'adrc'):
        out[k] = dict(ss=(d[f'{k}__A'], d[f'{k}__B'], d[f'{k}__C'],
                          d[f'{k}__D']),
                      keys=d[f'{k}__keys'], values=d[f'{k}__values'],
                      J=float(d[f'{k}__J']), n_par=int(d[f'{k}__n_par']),
                      n_states=int(d[f'{k}__n_states']))
    return out


def envelope(r, n=1200):
    """Enveloppe min/max par bloc + metriques a pleine resolution."""
    N = len(r['t'])
    k = max(1, N // n)
    nb = N // k
    o = dict(t=r['t'][:nb * k:k], diverged=r['diverged'],
             t_div=np.nan if r['t_div'] is None else r['t_div'],
             mean_u=mean_abs_amplitude(r['u']),
             mean_y=mean_abs_amplitude(r['y_mill']),
             max_u=float(np.abs(r['u']).max()),
             max_y=float(np.abs(r['y_mill']).max()))
    for key in ('y_mill', 'u'):
        v = r[key][:nb * k].reshape(nb, k)
        o[key + '_min'], o[key + '_max'] = v.min(axis=1), v.max(axis=1)
    return o


def main():
    t00 = time.time()
    print("=" * 74)
    print(" ETAPE 2 — COMPARAISON EQUITABLE : FOPID contre ADRC-FOPID")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE)
    ctl = load()
    cfgs = [('boucle ouverte', None), ('fopid', ctl['fopid']['ss']),
            ('adrc', ctl['adrc']['ss'])]
    store = {}

    # ---------------- 1. lobes de stabilite -------------------------------
    speeds = np.arange(3000, 7001, 200)
    lob = {}
    for name, ss in cfgs:
        v = []
        for rpm in speeds:
            L = limits(plate, ss, rpm, hi=4.0e-3, tol=2e-5)
            v.append(L.min())
        lob[name] = np.array(v)
        print(f"  lobes {name:14s} : moyenne {np.mean(v) * 1e3:.3f} mm,"
              f" min {np.min(v) * 1e3:.3f} mm  ({time.time() - t00:.0f} s)",
              flush=True)
    store['lobes'] = dict(rpm=speeds, **lob)

    # ---------------- 2. limites par position -----------------------------
    pos = {}
    for name, ss in cfgs:
        pos[name] = limits(plate, ss, C.RPM_DESIGN, hi=4.0e-3, tol=1e-5)
        print(f"  positions {name:14s} : {np.round(pos[name] * 1e3, 3)} mm"
              f"   min = {pos[name].min() * 1e3:.3f}", flush=True)
    store['positions'] = dict(x=np.array(C.POSITIONS), **pos)

    # ---------------- 3. reponses temporelles -----------------------------
    ap_test = 0.6e-3
    sim = MillingSimulation(plate, C.RPM_DESIGN, ap_test, n_modes=C.N_MODES,
                            n_sub=C.N_SUB, sign=C.SIGN_SIM)
    for name, ss in cfgs:
        c = None if ss is None else LTIController(ss, sim.dt)
        r = sim.run(controller=c, T=None)
        f, A = amplitude_spectrum(r['t'], r['y_mill'])
        fu, Au = amplitude_spectrum(r['t'], r['u'], scale=1.0)
        store[f'time_{name}'] = envelope(r)
        store[f'spec_{name}'] = dict(f=f, A=A, fu=fu, Au=Au)
        print(f"  temporel {name:14s} (a_p = {ap_test * 1e3:.2f} mm) : "
              f"{'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'stable'}"
              f"   |y|max = {np.abs(r['y_mill']).max() * 1e6:8.2f} um"
              f"   |u|max = {np.abs(r['u']).max():6.1f} V"
              f"   moy|u| = {mean_abs_amplitude(r['u']):5.2f} V", flush=True)
    store['time_meta'] = dict(ap=ap_test, rpm=C.RPM_DESIGN)

    # ---------------- 4. robustesse ---------------------------------------
    rob = {}
    cases = [('synthese 2 modes', dict(n_modes=2)),
             ('verif 5 modes', dict(n_modes=5)),
             ('derive +17/+9 %', dict(n_modes=2,
                                      freqs=[632.0, 1162.0]
                                      + C.F_THEORETICAL[2:])),
             ('amortissement x0.8', dict(n_modes=2, zeta=0.8))]
    for tag, kw in cases:
        pl = plate
        if 'freqs' in kw:
            pl = build_plate(C.PATCH_SIDE, calibrate=False)
            pl.calibrate_frequencies(kw['freqs'])
        if 'zeta' in kw:
            pl = build_plate(C.PATCH_SIDE)
            pl.zeta_modes = np.asarray(pl.zeta_modes, float) * kw['zeta']
        nm = kw['n_modes']
        row = {}
        for name, ss in cfgs:
            L = limits(pl, ss, C.RPM_DESIGN, n_modes=nm, hi=4.0e-3, tol=2e-5)
            row[name] = L.min()
        rob[tag] = row
        print(f"  robustesse [{tag:18s}] (n_modes={nm}) : " + "  ".join(
            f"{k} = {v * 1e3:.3f} mm" for k, v in row.items()), flush=True)
    store['robust'] = {t: np.array([rob[t][n] for n, _ in cfgs])
                       for t in rob}
    store['robust_labels'] = np.array([n for n, _ in cfgs])

    # ---------------- 5. metriques frequentielles -------------------------
    f = np.logspace(0.5, 4.1, 400)
    fr = dict(f=f)
    Pu, Pf = plant_frf(plate, f, C.N_MODES, x_force=0.0)
    fr['Pu'] = np.abs(Pu)
    for name, ss in cfgs:
        if ss is None:
            continue
        K = ss_frf(ss, 2 * np.pi * f)
        S = 1.0 / (1.0 - Pu * K)
        fr[f'K_{name}'] = np.abs(K)
        fr[f'S_{name}'] = np.abs(S)
        fr[f'U_{name}'] = np.abs(K * S * Pf)
        Ms, V = frequency_metrics(plate, ss)
        ev = nominal_poles(plate, ss)
        print(f"  frequentiel {name:8s} : Ms = {Ms:.3f}"
              f"   effort = {V:.0f} V/N   max Re(pole) = {ev.real.max():.1f}")
        store[f'metrics_{name}'] = np.array([Ms, V, ev.real.max()])
    store['freq'] = fr

    np.savez_compressed(os.path.join(OUT, 'compare.npz'),
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           if isinstance(v, dict) for kk, vv in v.items()},
                        **{k: v for k, v in store.items()
                           if not isinstance(v, dict)})
    print(f"\n  -> results/compare.npz   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main()
