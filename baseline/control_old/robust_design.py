"""
robust_design.py — Chaine complete de synthese du correcteur robuste
=====================================================================
Assemble, dans l'ordre du papier :

    plaque (Section 2)  ->  systeme incertain (Eqs. 17-26)
                        ->  ponderations (Section 3.3)
                        ->  iteration D-K (Eqs. 28-29)
                        ->  reduction equilibree du correcteur
                        ->  K_Pmu(s)

POSITION DE L'ACTIONNEUR — le papier se contredit :
  * Section 3 : "The actuator position is set as the left lower corner of the
    plate according to the optimization results in [66]" ;
  * Section 5 : "The piezoelectric patch actuator is pasted in the right lower
    corner of the plate back, while the displacement sensor acquires the
    vibration of right upper corner on the plate back".
Les deux sont fournies (`patch='right'` / `'left'`). Elles ne sont PAS
equivalentes : au coin droit, tous les produits modaux D_obs(i) H_Pe(i) ont le
MEME signe (paire actionneur/capteur duale, somme -4.47), alors qu'au coin
gauche ils alternent (somme +0.30). Seule la configuration droite — celle des
experiences des Figs. 17-21, donc celle des resultats a reproduire — permet a
une contre-reaction de vitesse d'amortir tous les modes a la fois, et donc a
la commande a retard de l'Eq. (30) d'etre efficace.
"""
import numpy as np
import control as ct

from chebyshev_plate import ChebyshevPlate
from uncertain_plant import MillingUncertainSystem
from dk_synthesis import (prepare_plant, unscale_controller, dk_iteration,
                          mu_curve, reduce_controller)

F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]      # Tableau 4 (mesure)
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]    # Tableau 4 (modele)
PATCH = dict(left=dict(x1=0.000, x2=0.020, z1=0.0, z2=0.060),
             right=dict(x1=0.080, x2=0.100, z1=0.0, z2=0.060))


def build_plate(patch='left', PX=14, PZ=14, n_modes=5, calibrate=True,
                freqs=None):
    """Plaque du Tableau 1 avec la pastille du Tableau 2.

    freqs : liste de frequences cibles [Hz] pour le calage. Par defaut le
    modele THEORIQUE du Tableau 4 (537, 1101, ... Hz) — c'est celui que le
    papier emploie pour la conception ET la verification (Sections 3-4 : la
    frequence de broutement simulee f_c2 = 1135 Hz se refere a 1101 Hz, pas a
    la valeur mesuree 1068 Hz, et les lobes de la Fig. 13(b) ne se
    reproduisent qu'avec ce jeu). Passer freqs=F_MEASURED pour le modele
    recale sur les mesures (Fig. 12, verifications de robustesse).
    """
    p = ChebyshevPlate(PX=PX, PZ=PZ, n_modes=n_modes)
    p.add_piezo_patch(**PATCH[patch])
    if calibrate:
        tgt = list(F_THEORETICAL[:n_modes]) if freqs is None             else list(freqs)[:n_modes]
        p.calibrate_frequencies(tgt)
    p.patch_side = patch
    return p


def freq_grid(f_lo=10.0, f_hi=6000.0, n_log=50, n_hi=12):
    """Grille de frequences pour l'analyse mu (Hz)."""
    return np.concatenate([np.logspace(np.log10(f_lo), np.log10(4000), n_log),
                           np.linspace(4100.0, f_hi, n_hi)])


def design_robust(plate, rpm, ap, ae=None, alpha_coupling='paper',
                  wpf=None, wpu=None, wpn=None, n_iter=3, orders=(0, 1, 1, 1),
                  reduce_to=12, verbose=True, f_grid=None, sign=1.0):
    """Correcteur robuste K_Pmu(s) par iteration D-K (Eq. 29).

    `sign` : convention de couplage regeneratif (voir config.SIGN_*)."""
    from milling_dynamics import AE_NOM
    ae = AE_NOM if ae is None else ae
    U = MillingUncertainSystem(plate, rpm, ap, ae, n_modes=2,
                               alpha_coupling=alpha_coupling,
                               wpf=wpf, wpu=wpu, wpn=wpn, sign=sign)
    P = U.generalized_plant()
    # decalage spectral : la synthese est faite sur (A + alpha*I), ce qui
    # garantit Re(poles de la boucle fermee nominale) <= -alpha. Sans lui,
    # hinfsyn peut laisser un pole quasi marginal (constate : Re = -0.01 sur
    # la boucle [plante 2 modes + K], que le couplage de coupe periodique
    # pousse ensuite legerement a droite). alpha = 12 rad/s est negligeable
    # devant la dynamique (>= 2*pi*500 rad/s) mais suffit largement.
    alpha_shift = 12.0
    Ps, info = prepare_plant(P, alpha_shift=alpha_shift)
    f = freq_grid() if f_grid is None else np.asarray(f_grid, float)
    w = 2 * np.pi * f / info['w0']
    res = dk_iteration(Ps, P['blocks'], w, n_iter=n_iter, orders=orders,
                       verbose=verbose)
    K_full = unscale_controller(res['K'], info)
    K = reduce_controller(K_full, reduce_to) if reduce_to else K_full
    return dict(U=U, P=P, K=K, K_full=K_full, mu=res['mu'], mus=res['mus'],
                f=f, history=res['history'], info=info, gamma=res['gamma'],
                ss=(np.array(K.A), np.array(K.B), np.array(K.C),
                    np.array(K.D)))


def mu_of_controller(P, info, K_real, f):
    """Borne superieure de mu pour un correcteur donne (echelle reelle)."""
    Ps, _ = prepare_plant(P, w0=info['w0'], u0=info['u0'], y0=info['y0'])
    w0, u0, y0 = info['w0'], info['u0'], info['y0']
    A = np.array(K_real.A) / w0
    B = np.array(K_real.B) / w0 * y0
    C = np.array(K_real.C) / u0
    D = np.array(K_real.D) * y0 / u0
    T = ct.ss(ct.ss(Ps).lft(ct.ss(A, B, C, D), 1, 1))
    mus, _ = mu_curve(T, P['blocks'], 2 * np.pi * np.asarray(f) / w0)
    return mus


def closed_loop_modes(plate, ss_ctrl, n_modes=5):
    """Frequences et amortissements de la boucle fermee NOMINALE (sans coupe
    ni retard) — mesure directe de l'amortissement actif apporte."""
    Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in ss_ctrl]
    n, nc = n_modes, Ac.shape[0]
    K0 = np.diag(plate.omega_n[:n]**2)
    C0 = np.diag(2 * plate.zeta_modes[:n] * plate.omega_n[:n])
    H = np.asarray(plate.H_Pe_modal, float)[:n]
    Do = plate.D_row(plate.lp, plate.hp)[:n]
    M = np.zeros((2 * n + nc, 2 * n + nc))
    M[:n, n:2 * n] = np.eye(n)
    M[n:2 * n, :n] = -K0 + float(Dc[0, 0]) * np.outer(H, Do)
    M[n:2 * n, n:2 * n] = -C0
    if nc:
        M[n:2 * n, 2 * n:] = np.outer(H, Cc.ravel())
        M[2 * n:, :n] = np.outer(Bc.ravel(), Do)
        M[2 * n:, 2 * n:] = Ac
    ev = np.linalg.eigvals(M)
    ev = ev[np.imag(ev) > 0]
    f = np.abs(ev) / (2 * np.pi)
    z = -np.real(ev) / np.abs(ev)
    i = np.argsort(f)
    return f[i], z[i], float(np.max(np.real(np.linalg.eigvals(M))))
