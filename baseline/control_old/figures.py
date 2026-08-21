"""
figures.py — Reproduction des figures du papier
================================================
Trace, a partir des resultats calcules par run_design / run_simulations /
run_stability, les figures 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 18, 21 du papier
plus deux figures propres a la synthese (convergence D-K et courbe de mu,
Eqs. 28-29).

    python figures.py
"""
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'plant'), HERE]

import config as C
from robust_design import build_plate
from uncertain_plant import MillingUncertainSystem
from weights import W_PAF, W_PAU, weight_mag
from milling_dynamics import (alpha4_series, alpha4_average, dtd_paper_gauge,
                              N_TEETH)
from chebyshev_plate import ChebyshevPlate

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
DB = lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-300))


def load(name):
    """Recharge un fichier npz enregistre a plat (cle 'groupe__champ')."""
    d = np.load(os.path.join(OUT, name), allow_pickle=True)
    out = {}
    for k in d.files:
        if '__' in k:
            g, f = k.split('__', 1)
            out.setdefault(g, {})[f] = d[k]
        else:
            out[k] = d[k]
    return out


def modal_frf(plate, D_a, D_b, f):
    """Somme modale sum_i D_a(i) D_b(i)/(w_i^2 - w^2 + 2 j z_i w_i w)."""
    om = 2 * np.pi * np.asarray(f, float)
    den = (plate.omega_n[:, None]**2 - om[None, :]**2
           + 2j * plate.zeta_modes[:, None] * plate.omega_n[:, None]
           * om[None, :])
    return np.sum((np.asarray(D_a) * np.asarray(D_b))[:, None] / den, axis=0)


# ---------------------------------------------------------------------------
def fig4(plate):
    """Reponses frequentielles du bord superieur en toutes positions."""
    f = np.linspace(20, 5000, 500)
    xs = np.linspace(0, plate.lp, 41)
    H = np.asarray(plate.H_Pe_modal, float)
    Gf = np.empty((len(xs), len(f)))
    Gu = np.empty((len(xs), len(f)))
    for i, x in enumerate(xs):
        D = plate.D_row(x, plate.hp)
        Gf[i] = DB(modal_frf(plate, D, D, f))
        Gu[i] = DB(modal_frf(plate, D, H, f))
    F, X = np.meshgrid(f, xs * 1e3)
    fig = plt.figure(figsize=(12, 4.6))
    for k, (Z, ttl) in enumerate([(Gf, "(a) entree force de coupe (Ref 1 m/N)"),
                                  (Gu, "(b) entree tension (Ref 1 m/V)")]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        ax.plot_surface(F, X, Z, cmap='viridis', linewidth=0,
                        rcount=41, ccount=120, antialiased=True)
        ax.set_xlabel('Frequence (Hz)'); ax.set_ylabel('Position (mm)')
        ax.set_zlabel('Amplitude (dB)'); ax.set_title(ttl, fontsize=10)
        ax.view_init(24, -128)
    fig.suptitle("Fig. 4 — reponses du bord superieur en toutes positions")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig04_frf_surface.png', dpi=130)
    plt.close(fig)


def fig5(plate):
    """Reponses maximales sur toutes les positions et poids additifs."""
    f = np.linspace(20, 5000, 2000)
    xs = np.linspace(0, plate.lp, 41)
    H = np.asarray(plate.H_Pe_modal, float)
    mf = np.max([np.abs(modal_frf(plate, plate.D_row(x, plate.hp),
                                  plate.D_row(x, plate.hp), f)) for x in xs],
                axis=0)
    mu_ = np.max([np.abs(modal_frf(plate, plate.D_row(x, plate.hp), H, f))
                  for x in xs], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, m, w, ttl in [(ax[0], mf, W_PAF, "(a) entree force (Ref 1 m/N)"),
                         (ax[1], mu_, W_PAU, "(b) entree tension (Ref 1 m/V)")]:
        a.plot(f, DB(m), lw=1.2, label="reponse max toutes positions")
        a.plot(f, DB(weight_mag(w, f)), 'r:', lw=1.8,
               label="poids concu $W_{Pa}$")
        a.set_xlabel('Frequence (Hz)'); a.set_ylabel('Amplitude (dB)')
        a.set_title(ttl, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Fig. 5 — poids d'incertitude additive, Eqs. (18)-(19) "
                 "(valeurs du papier)")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig05_weights.png', dpi=130)
    plt.close(fig)


def fig6(st):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, tag, ttl in zip(ax, ['start', 'quarter', 'half'],
                           ['(a) position de depart', '(b) position 1/4',
                            '(c) position 1/2']):
        d = st[f'fig6_{tag}']
        a.plot(d['rpm'], d['0.3'] * 1e3, lw=1.4, label=r'$0.3\bar\alpha_4$')
        a.plot(d['rpm'], d['1.0'] * 1e3, 'r--', lw=1.4, label=r'$\bar\alpha_4$')
        a.plot(d['rpm'], d['a4(t)'] * 1e3, 'g-', lw=1.4, label=r'$\alpha_4(t)$')
        a.plot(d['rpm'], d['2.9'] * 1e3, lw=1.4, color='#d68910',
               label=r'$2.9\bar\alpha_4$')
        a.set_ylim(0, 1.3); a.set_xlabel('Vitesse de broche (tr/min)')
        a.set_title(ttl, fontsize=10); a.grid(alpha=.3)
    ax[0].set_ylabel('Profondeur axiale (mm)'); ax[0].legend(fontsize=8)
    fig.suptitle("Fig. 6 — lobes de stabilite selon le coefficient de coupe "
                 "(Eq. 23 : $\\alpha_4 \\in [0.3, 2.9]\\bar\\alpha_4$)")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig06_lobes.png', dpi=130)
    plt.close(fig)


def fig7():
    """Elements de D^T D le long du bord, dans la jauge du papier (Fig. 7)."""
    bare = ChebyshevPlate(PX=14, PZ=14)
    xs, DtD, DD0, LDD, s = dtd_paper_gauge(bare)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ttl = ['(a) $D^TD(1,1)$', '(b) $D^TD(1,2)$', '(c) $D^TD(2,1)$',
           '(d) $D^TD(2,2)$']
    for a, (i, j), t in zip(ax.ravel(), idx, ttl):
        v = DtD[:, i, j]
        a.plot(xs * 1e3, v, 'g-', lw=1.6, label='reel')
        a.axhline(DD0[i, j], color='b', ls='-.', lw=1.2, label='moyenne')
        a.axhline(DD0[i, j] + LDD[i, j], color='r', ls='--', lw=1.2,
                  label='max & min')
        a.axhline(DD0[i, j] - LDD[i, j], color='r', ls='--', lw=1.2)
        a.set_xlabel('Position (mm)'); a.set_title(t, fontsize=10)
        a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Fig. 7 — variation de $D_{Pr}^T D_{Pr}$ le long du bord "
                 "(jauge modale du papier), Eq. (24)")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig07_dtd.png', dpi=130)
    plt.close(fig)


def fig8(plate, n_draw=25, seed=3):
    """Systeme incertain (Eq. 26) contre systeme original, trois positions."""
    rng = np.random.default_rng(seed)
    f = np.linspace(20, 5000, 900)
    om = 2 * np.pi * f
    H = np.asarray(plate.H_Pe_modal, float)
    D_obs = plate.D_row(plate.lp, plate.hp)
    U = MillingUncertainSystem(plate, C.RPM_S, C.AP_S)
    p = U.par
    Waf, Wau = weight_mag(W_PAF, f), weight_mag(W_PAU, f)
    fig, ax = plt.subplots(3, 2, figsize=(11, 9))
    for r, frac in enumerate([0.0, 0.25, 0.5]):
        x = frac * plate.lp
        D = plate.D_row(x, plate.hp)
        for c, (src, ttl) in enumerate(
                [(D, f"force, position {frac * 100:.0f} %"),
                 (H, f"tension, position {frac * 100:.0f} %")]):
            ax[r, c].plot(f, DB(modal_frf(plate, D_obs, src, f)), 'b-', lw=1.3,
                          label='systeme original', zorder=5)
            for _ in range(n_draw):
                d = rng.uniform(-1, 1, 10)
                M = np.eye(2) + np.diag(p['L_Pm'] * d[:2])
                Cm = p['C0'] + np.diag(p['L_Pc'] * d[2:4])
                Kk = (p['K_eff'] + np.diag(p['L_Pk'] * d[4:6])
                      + np.array([[p['L_PD'][0, 0] * d[6],
                                   p['L_PD'][0, 1] * d[7]],
                                  [p['L_PD'][1, 0] * d[8],
                                   p['L_PD'][1, 1] * d[9]]]))
                G = np.empty(len(f), complex)
                b = np.linalg.solve(M, src[:2])
                co = D_obs[:2]
                for k, w in enumerate(om):
                    Z = -w**2 * np.eye(2) + 1j * w * np.linalg.solve(M, Cm) \
                        + np.linalg.solve(M, Kk)
                    G[k] = co @ np.linalg.solve(Z, b)
                dl = rng.uniform(-1, 1)
                G = G + dl * (Waf if c == 0 else Wau)
                ax[r, c].plot(f, DB(G), 'r--', lw=.5, alpha=.55)
            ax[r, c].set_title(ttl, fontsize=9); ax[r, c].grid(alpha=.3)
            ax[r, c].set_xlabel('Frequence (Hz)')
            ax[r, c].set_ylabel('Amplitude (dB)')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Fig. 8 — systeme incertain (Eq. 26, tirages aleatoires) "
                 "contre systeme original")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig08_uncertainty.png', dpi=130)
    plt.close(fig)


def fig12(plate):
    """Reponses modales : point d'impact = coin superieur droit (Fig. 12)."""
    f = np.linspace(20, 5000, 3000)
    D = plate.D_row(plate.lp, plate.hp)
    H = np.asarray(plate.H_Pe_modal, float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(f, DB(modal_frf(plate, D, D, f) * 1e6), lw=1.1)
    ax[0].set_title("(a) FRF au coin (Ref 1 $\\mu$m/N)", fontsize=10)
    ax[1].plot(f, DB(modal_frf(plate, D, H, f) * 1e8), lw=1.1)
    ax[1].set_title("(b) tension -> deplacement (Ref 0.01 $\\mu$m/V)",
                    fontsize=10)
    for a in ax:
        a.set_xlabel('Frequence (Hz)'); a.set_ylabel('Amplitude (dB)')
        a.grid(alpha=.3)
        for fn in C.F_MEASURED:
            a.axvline(fn, color='r', ls=':', lw=.7)
    fig.suptitle("Fig. 12 — reponses d'impact et de balayage (modele cale sur "
                 "le Tableau 4)")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig12_modal.png', dpi=130)
    plt.close(fig)


def fig13(st):
    d = st['fig13']
    R, X = np.meshgrid(d['rpm'], d['x'] * 1e3)
    fig = plt.figure(figsize=(12, 4.4))
    a = fig.add_subplot(1, 2, 1, projection='3d')
    a.plot_surface(X, R, d['S'] * 1e3, cmap='jet', linewidth=0,
                   rcount=len(d['x']), ccount=len(d['rpm']))
    a.set_xlabel('Position (mm)'); a.set_ylabel('Vitesse (tr/min)')
    a.set_zlabel('Profondeur axiale (mm)')
    a.set_title("(a) stabilite en toutes positions", fontsize=10)
    a.view_init(26, -60)
    b = fig.add_subplot(1, 2, 2)
    b.plot(d['rpm'], d['lowest'] * 1e3, lw=1.6)
    b.plot([C.RPM_S], [C.AP_S * 1e3], 'ko', ms=5)
    b.annotate('S', (C.RPM_S, C.AP_S * 1e3), textcoords='offset points',
               xytext=(6, 6))
    b.set_xlabel('Vitesse de broche (tr/min)')
    b.set_ylabel('Profondeur axiale (mm)')
    b.set_title("(b) minimum sur toutes les positions", fontsize=10)
    b.grid(alpha=.3)
    fig.suptitle("Fig. 13 — stabilite predite en boucle ouverte "
                 "(avalant, $a_e$ = 0.1 mm)")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig13_stability.png', dpi=130)
    plt.close(fig)


def _band(ax, s, key, lab, color, ylab, scale=1.0, ls='-'):
    """Bande d'oscillation min/max — c'est ainsi que se lisent les traces
    temporelles du papier."""
    t = s['t']
    if key + '_min' in s:
        ax.fill_between(t, s[key + '_min'] * scale, s[key + '_max'] * scale,
                        color=color, alpha=.85, lw=0, label=lab)
    else:
        ax.plot(t, s[key] * scale, lw=.5, color=color, label=lab)
    ax.set_xlabel('Temps (s)'); ax.set_ylabel(ylab)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right')


def _panel(ax, t, y, lab, color, ylab):
    ax.plot(t, y, lw=.5, color=color, label=lab)
    ax.set_xlabel('Temps (s)'); ax.set_ylabel(ylab)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right')


def fig14(td):
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 9))
    s = td['S_sans']
    _band(ax[0, 0], s, 'y_mill', 'sans commande', '#c8963e',
          'Deplacement ($\\mu$m)', 1e6)
    d = td['S_sans_spec']
    ax[0, 1].plot(d['f'], d['A'], color='#c8963e', lw=1.0)
    s = td['S_retard']
    _band(ax[1, 0], s, 'y_mill', 'commande a retard seule (Eq. 30)',
          '#c0392b', 'Deplacement ($\\mu$m)', 1e6)
    d = td['S_retard_spec']
    ax[1, 1].plot(d['f'], d['A'], color='#c0392b', lw=1.0)
    for name, col, lab in [('S_robuste', '#1a3f8f', 'commande robuste'),
                           ('S_combine', '#16a085', 'robuste + retard')]:
        s = td[name]
        ax[2, 0].fill_between(s['t'], s['y_mill_min'] * 1e6,
                              s['y_mill_max'] * 1e6, color=col, alpha=.6,
                              lw=0, label=lab)
        d = td[name + '_spec']
        ax[2, 1].plot(d['f'], d['A'], color=col, lw=1.1,
                      ls='-' if 'rob' in name else ':', label=lab)
    ax[2, 0].set_xlabel('Temps (s)'); ax[2, 0].set_ylabel('Deplacement ($\\mu$m)')
    ax[2, 0].grid(alpha=.3); ax[2, 0].legend(fontsize=8)
    ax[2, 1].legend(fontsize=8)
    for a in ax[:, 1]:
        a.set_xlabel('Frequence (Hz)'); a.set_ylabel('Amplitude ($\\mu$m)')
        a.grid(alpha=.3); a.set_xlim(0, 1600)
    for lbl, a in zip('abcdef', ax.ravel()):
        a.set_title(f'({lbl})', loc='left', fontsize=9)
    fig.suptitle("Fig. 14 — reponses a 4900 tr/min, $a_e$ = 0.1 mm, "
                 "$a_p$ = 0.3 mm")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig14_responses.png', dpi=130)
    plt.close(fig)


def fig15(td):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    s = td['S_retard']
    ax[0].fill_between(s['t'], s['u_min'], s['u_max'], color='#c0392b',
                       alpha=.85, lw=0, label='commande a retard seule')
    ax[0].set_title("(a) tension de la commande a retard", fontsize=10)
    for name, col, lab in [('S_robuste', '#1a3f8f', 'robuste'),
                           ('S_combine', '#16a085', 'robuste + retard')]:
        s = td[name]
        ax[1].fill_between(s['t'], s['u_min'], s['u_max'], color=col,
                           alpha=.6, lw=0, label=lab)
    ax[1].set_title("(b) tensions robuste et combinee", fontsize=10)
    for a in ax:
        a.set_xlabel('Temps (s)'); a.set_ylabel('Tension (V)')
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Fig. 15 — tensions de commande, condition S")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig15_voltages.png', dpi=130)
    plt.close(fig)


def fig16(td):
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 9))
    s = td['P_sans']
    _band(ax[0, 0], s, 'y_mill', 'sans commande (perturbe)', '#c8963e',
          'Deplacement ($\\mu$m)', 1e6)
    d = td['P_sans_spec']
    ax[0, 1].plot(d['f'], d['A'], color='#c8963e', lw=1.0)
    s = td['P_combine']
    _band(ax[1, 0], s, 'y_mill', 'robuste + retard (perturbe)', '#16a085',
          'Deplacement ($\\mu$m)', 1e6)
    d = td['P_combine_spec']
    ax[1, 1].plot(d['f'], d['A'], color='#16a085', lw=1.0)
    _band(ax[2, 0], s, 'u', 'tension (perturbe)', '#16a085', 'Tension (V)')
    r = td['S_combine']
    ax[2, 1].fill_between(r['t'], r['u_min'], r['u_max'], color='#1a3f8f',
                          alpha=.6, lw=0, label='nominal')
    ax[2, 1].fill_between(s['t'], s['u_min'], s['u_max'], color='#16a085',
                          alpha=.6, lw=0, label='perturbe')
    ax[2, 1].set_xlabel('Temps (s)'); ax[2, 1].set_ylabel('Tension (V)')
    ax[2, 1].grid(alpha=.3); ax[2, 1].legend(fontsize=8)
    for a in ax[:2, 1]:
        a.set_xlabel('Frequence (Hz)'); a.set_ylabel('Amplitude ($\\mu$m)')
        a.grid(alpha=.3); a.set_xlim(0, 1600)
    fig.suptitle("Fig. 16 — systeme perturbe : masse et raideur modales "
                 "+10 %, amortissement x 0.8")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig16_perturbed.png', dpi=130)
    plt.close(fig)


def fig18(st):
    d = st['fig18']
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    styles = [('sans', '#d68910', 'o-', 'sans commande (conv. simulation)'),
              ('sans_exp', '#8a6d3b', 'o--', 'sans commande (conv. exper.)'),
              ('retard', '#c0392b', 'v-', 'commande a retard seule'),
              ('robuste', '#1a3f8f', '*-', 'commande robuste'),
              ('combine', '#16a085', '*-', 'robuste + retard')]
    for key, col, mk, lab in styles:
        if key in d:
            ax.plot(d['rpm'], d[key] * 1e3, mk, color=col, lw=1.4, ms=8,
                    label=lab)
    for y, lab in [(0.1, 'papier : sans commande'),
                   (0.6, 'papier : robuste'),
                   (0.8, 'papier : robuste + retard')]:
        ax.axhline(y, color='gray', ls=':', lw=1)
        ax.annotate(lab, (4310, y + 0.02), fontsize=7, color='gray')
    ax.set_xlabel('Vitesse de broche (tr/min)')
    ax.set_ylabel('Profondeur axiale limite (mm)')
    ax.set_title("Fig. 18 — limite de coupe (minimum sur tout le bord "
                 "superieur)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig18_limits.png', dpi=130)
    plt.close(fig)


def fig21(td):
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 9))
    s = td['T2_sans']
    _band(ax[0, 0], s, 'y_mill', 'sans commande', '#c8963e',
          'Deplacement ($\\mu$m)', 1e6)
    ax[0, 1].plot(td['T2_sans_spec']['f'], td['T2_sans_spec']['A'],
                  color='#c8963e', lw=1.0)
    for name, col, lab in [('T2_robuste', '#1a3f8f', 'robuste'),
                           ('T2_combine', '#16a085', 'robuste + retard')]:
        s = td[name]
        ax[1, 0].fill_between(s['t'], s['y_mill_min'] * 1e6,
                              s['y_mill_max'] * 1e6, color=col, alpha=.6,
                              lw=0, label=lab)
        ax[1, 1].plot(td[name + '_spec']['f'], td[name + '_spec']['A'],
                      color=col, lw=1.1, label=lab)
        ax[2, 0].fill_between(s['t'], s['u_min'], s['u_max'], color=col,
                              alpha=.6, lw=0, label=lab)
        ax[2, 1].plot(td[name + '_spec']['fu'], td[name + '_spec']['Au'],
                      color=col, lw=1.1, label=lab)
    ax[1, 0].set_xlabel('Temps (s)'); ax[1, 0].set_ylabel('Deplacement ($\\mu$m)')
    ax[2, 0].set_xlabel('Temps (s)'); ax[2, 0].set_ylabel('Tension (V)')
    ax[1, 1].set_ylabel('Amplitude ($\\mu$m)')
    ax[2, 1].set_ylabel('Amplitude (V)')
    for a in ax[:, 1]:
        a.set_xlabel('Frequence (Hz)'); a.set_xlim(0, 1600); a.grid(alpha=.3)
    for a in ax[1:, 0]:
        a.grid(alpha=.3); a.legend(fontsize=8)
    for a in ax[1:, 1]:
        a.legend(fontsize=8)
    fig.suptitle("Fig. 21 — condition T2 : 6100 tr/min, $a_p$ = 0.5 mm")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig21_T2.png', dpi=130)
    plt.close(fig)


def fig_mu(ct):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].semilogx(ct['f_mu'], ct['mus'], lw=1.6)
    ax[0].axhline(1.0, color='r', ls='--', lw=1)
    ax[0].set_xlabel('Frequence (Hz)')
    ax[0].set_ylabel(r'borne sup. de $\mu$')
    ax[0].set_title(r"(a) $\sup_\omega \mu_\Delta(T_{P\mu})$, Eqs. (28)-(29)",
                    fontsize=10)
    ax[0].grid(alpha=.3, which='both')
    it = np.arange(len(ct['dk_mu']))
    ax[1].plot(it, ct['dk_gamma'], 'o-', label=r'$\gamma$ ($H_\infty$)')
    ax[1].plot(it, ct['dk_mu'], 's-', label=r'pic de $\mu$')
    ax[1].set_xlabel('iteration D-K'); ax[1].set_xticks(it)
    ax[1].set_title("(b) convergence de l'iteration D-K", fontsize=10)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle("Synthese $\\mu$ : borne structuree et convergence D-K")
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_mu_dk.png', dpi=130)
    plt.close(fig)


def fig_damping(ct):
    """Amortissement actif apporte — mesure directe de l'effet du correcteur."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    f, z = ct['cl_freq'], ct['cl_zeta']
    m = f < 4500
    ax.semilogy(f[m], 100 * z[m], 'o', ms=8, label='boucle fermee (robuste)')
    ax.semilogy(C.F_MEASURED, 100 * np.array(C.ZETA), 's', ms=8,
                label='boucle ouverte (Tableau 4)')
    ax.set_xlabel('Frequence (Hz)'); ax.set_ylabel('Amortissement (%)')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
    ax.set_title("Amortissement modal avant / apres commande", fontsize=10)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_damping.png', dpi=130)
    plt.close(fig)


def main():
    plate = build_plate(C.PATCH_SIDE)
    print("figures : modele ...", flush=True)
    fig4(plate); fig5(plate); fig7(); fig8(plate); fig12(plate)
    if os.path.exists(os.path.join(OUT, 'stability.npz')):
        st = load('stability.npz')
        print("figures : stabilite ...", flush=True)
        fig6(st); fig13(st); fig18(st)
    if os.path.exists(os.path.join(OUT, 'time_domain.npz')):
        td = load('time_domain.npz')
        print("figures : temporel ...", flush=True)
        fig14(td); fig15(td); fig16(td); fig21(td)
    if os.path.exists(os.path.join(OUT, 'controllers.npz')):
        ct = load('controllers.npz')
        fig_mu(ct); fig_damping(ct)
    print(f"  -> {FIG}")


if __name__ == '__main__':
    main()
