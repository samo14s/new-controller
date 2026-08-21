"""
run_simulations.py — Etape 2 : simulations temporelles (Section 4 du papier)
============================================================================
Reproduit les conditions exactes des figures :

  Fig. 14/15 : condition S = 4900 tr/min, ae = 0.1 mm, ap = 0.3 mm,
               fz = 0.02 mm/dent, avalant ; passe complete de 20.408 s
               (l'outil parcourt les 100 mm du bord superieur a 4.90 mm/s) ;
               quatre cas : sans commande, retard seul (Eq. 30), robuste,
               robuste combine.
  Fig. 16    : meme condition avec 10 % de perturbation de masse et de
               raideur modales et 80 % de l'amortissement.
  Fig. 21    : condition T2 = 6100 tr/min, ap = 0.5 mm, passe de 16.4 s.

    python run_simulations.py [--quick]
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
from simulate import MillingSimulation, amplitude_spectrum, mean_abs_amplitude
from delay_control import CombinedController

OUT = os.path.join(HERE, '..', 'results')


def load_controller():
    d = np.load(os.path.join(OUT, 'controllers.npz'))
    return (d['A'], d['B'], d['C'], d['D']), (float(d['K_Pp']),
                                              float(d['K_Pd']))


def run_case(sim, ss, pd, T=None, split=False):
    ctl = None
    if ss is not None or pd is not None:
        ctl = CombinedController(sim.dt, sim.n_sub, robust=ss, pd=pd)
    return sim.run(controller=ctl, T=T, record_split=split)


def envelope(r, keys=('y_mill', 'u', 'y_obs'), n=1500):
    """Enveloppe min/max par bloc — c'est ainsi que se lisent les Figs. 14-15
    et 19-21 du papier (bandes d'oscillation). Un simple sous-echantillonnage
    serait faux : le pas decime (~5 ms) est presque commensurable avec la
    periode de dent (4.08 ms) et n'echantillonnerait que quelques phases.

    Les GRANDEURS CHIFFREES (moyennes, maxima, spectres) sont toutes calculees
    sur le signal a pleine resolution, avant tout traitement d'affichage.
    """
    N = len(r['t'])
    k = max(1, N // n)
    nb = N // k
    out = dict(t=r['t'][:nb * k:k], dt=r['dt'], tau=r['tau'],
               T_pass=r['T_pass'], diverged=r['diverged'],
               t_div=np.nan if r['t_div'] is None else r['t_div'],
               x_tool=r['x_tool'][:nb * k:k])
    for key in keys:
        if key not in r:
            continue
        v = r[key][:nb * k].reshape(nb, k)
        out[key + '_min'] = v.min(axis=1)
        out[key + '_max'] = v.max(axis=1)
        out[key] = v.mean(axis=1)
    # metriques a pleine resolution
    out['mean_abs_u'] = mean_abs_amplitude(r['u'])
    out['mean_abs_y'] = mean_abs_amplitude(r['y_mill'])
    out['max_abs_u'] = float(np.abs(r['u']).max())
    out['max_abs_y'] = float(np.abs(r['y_mill']).max())
    if 'u_delay' in r:
        out['mean_abs_upd'] = mean_abs_amplitude(r['u_delay'])
        out['mean_abs_urob'] = mean_abs_amplitude(r['u_robust'])
    return out


def main(quick=False):
    t00 = time.time()
    print("=" * 74)
    print(" ETAPE 2 — SIMULATIONS TEMPORELLES (Section 4)")
    print("=" * 74)
    plate = build_plate(C.PATCH_SIDE)
    ss, pd = load_controller()
    n_sub = 82 if quick else C.N_SUB
    store = {}

    # ---------------- Fig. 14 / 15 : condition S -------------------------
    print(f"  convention de couplage : SIGN_SIM = {C.SIGN_SIM:+.0f} "
          "(Section 4 du papier)")
    nm = C.N_MODES_PAPER          # modele de verification du papier
    print(f"  modele de la Section 4 : {nm} modes (celui du papier, Eq. 12 "
          "simplifiee)")
    sim = MillingSimulation(plate, C.RPM_S, C.AP_S, n_modes=nm,
                            n_sub=n_sub, sign=C.SIGN_SIM)
    pd_alone = pd          # Eq. (30) avec le gain de l'Eq. (31)
    cases = [('sans', None, None, None),
             ('retard', None, pd_alone, None),
             ('robuste', ss, None, None),
             ('combine', ss, pd, None)]
    for name, s_, p_, T in cases:
        t0 = time.time()
        r = run_case(sim, s_, p_, T=T, split=(p_ is not None))
        f, A = amplitude_spectrum(r['t'], r['y_mill'])
        fu, Au = amplitude_spectrum(r['t'], r['u'], scale=1.0)
        store[f'S_{name}'] = envelope(r)
        store[f'S_{name}_spec'] = dict(f=f, A=A, fu=fu, Au=Au)
        print(f"  S/{name:8s}: {'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'stable'}"
              f"   max|y| = {np.abs(r['y_mill']).max() * 1e6:8.2f} um"
              f"   moy|u| = {mean_abs_amplitude(r['u']):6.2f} V"
              f"   max|u| = {np.abs(r['u']).max():6.1f} V"
              f"   ({time.time() - t0:.0f} s)", flush=True)

    # Deux cas transposes au calage de NOTRE modele (limite BO a 4900 :
    # 0.331 mm ici contre < 0.30 mm dans le papier, ecart ~10 %) :
    #   * 0.35 mm sans commande -> la divergence lente du MODE 2 (f_c2),
    #     phenomene de la Fig. 14(a)-(b) ;
    #   * 0.70 mm retard seul   -> stable au milieu, instable aux deux bouts,
    #     phenomene de la Fig. 14(c)-(d).
    # poste FIXE au depart (x = 0) : c'est la lecture de la Fig. 14(a), qui
    # ne montre que 0.2 s — l'outil n'a pas le temps de quitter la zone
    # instable ; en passe MOBILE la fenetre instable pres de x = 0 est etroite
    # et la croissance s'arrete des que l'outil en sort.
    simd = MillingSimulation(plate, C.RPM_S, 0.35e-3, n_modes=nm,
                             n_sub=n_sub, sign=C.SIGN_SIM)
    r = simd.run(controller=None, T=2.5, moving=False, x0=0.0, stop_um=600.0)
    f, A = amplitude_spectrum(r['t'], r['y_mill'])
    store['S_sans035'] = envelope(r)
    store['S_sans035_spec'] = dict(f=f, A=A)
    print(f"  S/sans 0.35mm : {'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'croissance lente'}"
          f"   max|y| = {np.abs(r['y_mill']).max() * 1e6:8.2f} um", flush=True)
    sime = MillingSimulation(plate, C.RPM_S, 0.40e-3, n_modes=nm,
                             n_sub=n_sub, sign=C.SIGN_SIM)
    ctl = CombinedController(sime.dt, sime.n_sub, robust=None, pd=pd_alone)
    r = sime.run(controller=ctl, T=None, stop_um=300.0)
    f, A = amplitude_spectrum(r['t'], r['y_mill'])
    store['S_retard07'] = envelope(r)
    store['S_retard07_spec'] = dict(f=f, A=A)
    print(f"  S/retard 0.40mm : {'instable (bouts) des %.2f s' % r['t_div'] if r['diverged'] else 'stable'}"
          f"   max|y| = {np.abs(r['y_mill']).max() * 1e6:8.2f} um", flush=True)

    ur = float(store['S_robuste']['mean_abs_u'])
    uc = float(store['S_combine']['mean_abs_u'])
    yr = float(store['S_robuste']['mean_abs_y'])
    yc = float(store['S_combine']['mean_abs_y'])
    print(f"  -> tension combinee / robuste : {100 * (uc / ur - 1):+.1f} %"
          f"   vibration : {100 * (yc / yr - 1):+.1f} %"
          f"   (papier : -22 % en simulation)")

    # ---------------- Fig. 16 : systeme perturbe --------------------------
    # LECTURE LITTERALE : 10 % sur la masse ET sur la raideur modales.
    # Prises de meme signe elles laissent omega = sqrt(k/m) INCHANGE — ce que
    # le papier confirme lui-meme en trouvant f_c2p = f_c2 = 1135 Hz. Seul
    # l'amortissement change (x 0.8). La robustesse aux DECALAGES de
    # frequence, elle, est evaluee separement dans run_stability.py, sur les
    # +17 % / +9 % constates par le papier apres une serie d'essais.
    scale = np.full(C.N_MODES_SIM, 1.0)
    simp = MillingSimulation(plate, C.RPM_S, C.AP_S, n_modes=nm,
                             n_sub=n_sub, mode_scale=scale[:nm],
                             zeta_scale=0.8, sign=C.SIGN_SIM)
    for name, s_, p_, T in [('sans', None, None, None),
                            ('combine', ss, pd, None)]:
        t0 = time.time()
        r = run_case(simp, s_, p_, T=T, split=(p_ is not None))
        f, A = amplitude_spectrum(r['t'], r['y_mill'])
        store[f'P_{name}'] = envelope(r)
        store[f'P_{name}_spec'] = dict(f=f, A=A)
        print(f"  P/{name:8s}: {'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'stable'}"
              f"   max|y| = {np.abs(r['y_mill']).max() * 1e6:8.2f} um"
              f"   moy|u| = {mean_abs_amplitude(r['u']):6.2f} V"
              f"   ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- Fig. 21 : condition T2 ------------------------------
    sim2 = MillingSimulation(plate, C.RPM_T2, C.AP_T2, n_modes=nm,
                             n_sub=n_sub, sign=C.SIGN_SIM)
    for name, s_, p_, T in [('sans', None, None, None),
                            ('robuste', ss, None, None),
                            ('combine', ss, pd, None)]:
        t0 = time.time()
        r = run_case(sim2, s_, p_, T=T, split=(p_ is not None))
        f, A = amplitude_spectrum(r['t'], r['y_mill'])
        fu, Au = amplitude_spectrum(r['t'], r['u'], scale=1.0)
        store[f'T2_{name}'] = envelope(r)
        store[f'T2_{name}_spec'] = dict(f=f, A=A, fu=fu, Au=Au)
        print(f"  T2/{name:7s}: {'DIVERGE a %.3f s' % r['t_div'] if r['diverged'] else 'stable'}"
              f"   max|y| = {np.abs(r['y_mill']).max() * 1e6:8.2f} um"
              f"   moy|u| = {mean_abs_amplitude(r['u']):6.2f} V"
              f"   ({time.time() - t0:.0f} s)", flush=True)
    u2r = float(store['T2_robuste']['mean_abs_u'])
    u2c = float(store['T2_combine']['mean_abs_u'])
    print(f"  -> tension combinee / robuste (T2) : {100 * (u2c / u2r - 1):+.1f} %"
          f"   (papier : -11 % en experience)")

    np.savez_compressed(os.path.join(OUT, 'time_domain.npz'),
                        **{f'{k}__{kk}': vv for k, v in store.items()
                           for kk, vv in v.items()})
    print(f"\n  -> results/time_domain.npz   ({time.time() - t00:.0f} s)")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
