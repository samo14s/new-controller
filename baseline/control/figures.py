"""
figures.py — Figures de la comparaison FOPID / ADRC-FOPID
==========================================================
    python figures.py
"""
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'plant'), HERE]

import config as C

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

COL = {'boucle ouverte': '#c8963e', 'fopid': '#1a3f8f', 'adrc': '#16a085'}
LAB = {'boucle ouverte': 'sans commande', 'fopid': 'FOPID',
       'adrc': 'ADRC-FOPID'}


def load(name):
    d = np.load(os.path.join(OUT, name), allow_pickle=True)
    out = {}
    for k in d.files:
        if '__' in k:
            g, f = k.split('__', 1)
            out.setdefault(g, {})[f] = d[k]
        else:
            out[k] = d[k]
    return out


def fig_lobes(cp):
    d = cp['lobes']
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k in ('boucle ouverte', 'fopid', 'adrc'):
        ax.plot(d['rpm'], d[k] * 1e3, '-o', ms=4, color=COL[k], lw=1.6,
                label=LAB[k])
    ax.axvline(C.RPM_DESIGN, color='k', ls=':', lw=1)
    ax.annotate('vitesse de synthese', (C.RPM_DESIGN, 2.7), fontsize=8,
                rotation=90, va='top', ha='right')
    ax.set_xlabel('Vitesse de broche (tr/min)')
    ax.set_ylabel('Profondeur axiale limite (mm)')
    ax.set_title("Lobes de stabilite — minimum sur tout le bord superieur\n"
                 "(modele complet a 5 modes, Floquet m = 200)", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_lobes.png', dpi=130)
    plt.close(fig)


def fig_positions(cp):
    d = cp['positions']
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.26
    xs = np.arange(len(d['x']))
    for i, k in enumerate(('boucle ouverte', 'fopid', 'adrc')):
        ax.bar(xs + (i - 1) * w, d[k] * 1e3, w, color=COL[k], label=LAB[k])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v * 100:.0f} %" for v in d['x']])
    ax.set_xlabel("Position de l'outil sur le bord")
    ax.set_ylabel('Profondeur axiale limite (mm)')
    ax.set_title(f"Limites par position a {C.RPM_DESIGN} tr/min", fontsize=10)
    ax.grid(alpha=.3, axis='y')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_positions.png', dpi=130)
    plt.close(fig)


def fig_time(cp):
    ap = float(cp['time_meta']['ap']) * 1e3
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 9))
    for r, k in enumerate(('boucle ouverte', 'fopid', 'adrc')):
        s, sp = cp[f'time_{k}'], cp[f'spec_{k}']
        ax[r, 0].fill_between(s['t'], s['y_mill_min'] * 1e6,
                              s['y_mill_max'] * 1e6, color=COL[k], alpha=.85,
                              lw=0, label=LAB[k])
        ax[r, 0].set_ylabel('Deplacement ($\\mu$m)')
        ax[r, 0].legend(fontsize=8, loc='upper right')
        ax[r, 1].plot(sp['f'], sp['A'], color=COL[k], lw=1.0)
        ax[r, 1].set_ylabel('Amplitude ($\\mu$m)')
        ax[r, 1].set_xlim(0, 1600)
        for a in ax[r]:
            a.grid(alpha=.3)
    for a in ax[:, 0]:
        a.set_xlabel('Temps (s)')
    for a in ax[:, 1]:
        a.set_xlabel('Frequence (Hz)')
    fig.suptitle(f"Reponses a {C.RPM_DESIGN} tr/min, $a_p$ = {ap:.2f} mm "
                 "(passe complete)")
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_time.png', dpi=130)
    plt.close(fig)


def fig_voltage(cp):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for k in ('fopid', 'adrc'):
        s = cp[f'time_{k}']
        ax[0].fill_between(s['t'], s['u_min'], s['u_max'], color=COL[k],
                           alpha=.6, lw=0, label=LAB[k])
        ax[1].plot(cp[f'spec_{k}']['fu'], cp[f'spec_{k}']['Au'],
                   color=COL[k], lw=1.1, label=LAB[k])
    ax[0].set_xlabel('Temps (s)')
    ax[0].set_ylabel('Tension (V)')
    ax[0].set_title('(a) tension de commande', fontsize=10)
    ax[1].set_xlabel('Frequence (Hz)')
    ax[1].set_ylabel('Amplitude (V)')
    ax[1].set_xlim(0, 1600)
    ax[1].set_title('(b) spectre de la tension', fontsize=10)
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=9)
    fig.suptitle("Effort de commande — meme contrainte pour les deux "
                 "correcteurs")
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_voltage.png', dpi=130)
    plt.close(fig)


def fig_freq(cp):
    d = cp['freq']
    f = d['f']
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for k in ('fopid', 'adrc'):
        ax[0].loglog(f, d[f'K_{k}'], color=COL[k], lw=1.4, label=LAB[k])
        ax[1].semilogx(f, d[f'S_{k}'], color=COL[k], lw=1.4, label=LAB[k])
        ax[2].loglog(f, d[f'U_{k}'], color=COL[k], lw=1.4, label=LAB[k])
    ax[0].set_ylabel('$|K(j\\omega)|$ (V/m)')
    ax[0].set_title('(a) correcteurs optimises', fontsize=10)
    ax[1].axhline(C.MS_MAX, color='r', ls='--', lw=1,
                  label=f'contrainte $M_s$ = {C.MS_MAX}')
    ax[1].set_ylabel('$|S(j\\omega)|$')
    ax[1].set_title('(b) sensibilite', fontsize=10)
    ax[2].axhline(C.V_PER_N, color='r', ls='--', lw=1, label='contrainte')
    ax[2].set_ylabel('$|KSP_f|$ (V/N)')
    ax[2].set_title("(c) effort d'actionneur", fontsize=10)
    for a in ax:
        a.set_xlabel('Frequence (Hz)')
        a.grid(alpha=.3, which='both')
        a.legend(fontsize=8)
    fig.suptitle("Signatures frequentielles sous contraintes identiques")
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_frequency.png', dpi=130)
    plt.close(fig)


def fig_pso(ps):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ('fopid', 'adrc'):
        h = ps[k]['hist']
        it = np.arange(h.shape[1])
        for i in range(h.shape[0]):
            ax[0].plot(it, h[i], color=COL[k], lw=1.0, alpha=.55,
                       label=LAB[k] if i == 0 else None)
        ax[0].plot(it, h.max(axis=0), color=COL[k], lw=2.2)
    ax[0].set_xlabel('Iteration PSO')
    ax[0].set_ylabel('$J$ (marge de Floquet)')
    ax[0].set_title("(a) convergence, 3 graines par structure", fontsize=10)
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=9)
    ax[0].set_ylim(-1.0, 0.6)
    ks = ('fopid', 'adrc')
    xs = np.arange(len(ks))
    for i, k in enumerate(ks):
        js = ps[k]['J_seeds']
        ax[1].bar(xs[i], js.max(), 0.5, color=COL[k])
        ax[1].plot(np.full(len(js), xs[i]), js, 'ko', ms=6, zorder=5)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([LAB[k] for k in ks])
    ax[1].set_ylabel('$J$ final')
    ax[1].set_title("(b) meilleur et dispersion sur les graines", fontsize=10)
    ax[1].grid(alpha=.3, axis='y')
    ax[1].axhline(0, color='k', lw=.8)
    fig.suptitle("Optimisation PSO — budget identique "
                 f"({int(ps['fopid']['n_eval'])} evaluations chacun)")
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_pso.png', dpi=130)
    plt.close(fig)


def fig_robust(cp):
    labs = [str(x) for x in cp['robust_labels']]
    tags = [k for k in cp['robust']]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.26
    xs = np.arange(len(tags))
    for i, name in enumerate(labs):
        v = np.array([cp['robust'][t][i] for t in tags]) * 1e3
        ax.bar(xs + (i - 1) * w, v, w, color=COL[name], label=LAB[name])
    ax.set_xticks(xs)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel('Profondeur limite minimale (mm)')
    ax.set_title("Robustesse — la derniere colonne montre l'optimisme du "
                 "modele de synthese", fontsize=10)
    ax.grid(alpha=.3, axis='y')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_robust.png', dpi=130)
    plt.close(fig)


def main():
    ps = load('pso.npz')
    fig_pso(ps)
    if os.path.exists(os.path.join(OUT, 'compare.npz')):
        cp = load('compare.npz')
        fig_lobes(cp)
        fig_positions(cp)
        fig_time(cp)
        fig_voltage(cp)
        fig_freq(cp)
        fig_robust(cp)
    print(f"  -> {FIG}")


if __name__ == '__main__':
    main()
