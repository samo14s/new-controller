"""
figures.py — figures for Phases 12-14.
======================================
    python figures.py       -> results/figures_phase2/*.png
"""
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C

OUT = os.path.join(C.RESULTS, 'figures_phase2')
os.makedirs(OUT, exist_ok=True)
NAMES = dict(open='no control', pid='PID', lqr='LQR', smc='SMC',
             mu_paper='mu (published set)', mu_exact='mu (corrected set)',
             proposed='PB-RAC', **{'proposed*': 'PB-RAC scheduled'})
COL = dict(open='0.55', pid='tab:orange', lqr='tab:blue', smc='tab:green',
           mu_paper='tab:purple', mu_exact='tab:brown',
           proposed='tab:red', **{'proposed*': 'crimson'})
STY = {'proposed*': '--'}


def _keys(d, prefix):
    out = []
    for k in d.files:
        if k.startswith(prefix) and not k.endswith('_ratio'):
            name = k[len(prefix):]
            if name and '_' not in name.strip('*'):
                out.append(name)
    return [k for k in ('open', 'pid', 'lqr', 'smc', 'mu_paper', 'mu_exact',
                        'proposed', 'proposed*') if k in out]


def bar_scenario(d, prefix, xlabels, title, fname, xlabel=''):
    ks = _keys(d, prefix)
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(xlabels))
    w = 0.8 / max(len(ks), 1)
    for i, k in enumerate(ks):
        ax.bar(x + i * w - 0.4 + w / 2, np.asarray(d[prefix + k], float), w,
               label=NAMES.get(k, k), color=COL.get(k, None))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(r'$a_{p,\lim}$  (mm)')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def main():
    p = os.path.join(C.RESULTS, 'compare_phase2.npz')
    if not os.path.exists(p):
        print('no compare_phase2.npz yet')
        return
    d = np.load(p, allow_pickle=True)

    # S1: limit per position
    ks = [k for k in ('open', 'pid', 'lqr', 'smc', 'mu_paper', 'mu_exact',
                      'proposed') if f'S1_{k}_limits' in d.files]
    if ks:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        xs = np.array(C.POSITIONS) * 100.0
        for k in ks:
            ax.plot(xs, np.asarray(d[f'S1_{k}_limits'], float) * 1e3, 'o-',
                    label=NAMES.get(k, k), color=COL.get(k, None))
        ax.set_xlabel('tool position along the top edge (mm)')
        ax.set_ylabel(r'$a_{p,\lim}$  (mm)')
        ax.set_title(f'Scenario 1 - nominal, {C.RPM_S} rpm')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig_s1_positions.png'), dpi=150)
        plt.close(fig)

    etas = np.linspace(0.0, C.ETA_MAX, 5)
    bar_scenario(d, 'S2a_', [f'{e:.3f}' for e in etas],
                 'Scenario 2(a) - material removal, physics-based',
                 'fig_s2a_removal.png', xlabel=r'$\eta = V_{rem}/V_0$')
    bar_scenario(d, 'S2b_', ['-10%', '-5%', '0', '+5%', '+10%'],
                 "Scenario 2(b) - the paper's symmetric box",
                 'fig_s2b_box.png', xlabel=r'$\delta_m$')
    bar_scenario(d, 'S3_', ['z x0.8', 'z x1.2', 'K -10%', 'K +10%',
                            'K-10% z0.8'],
                 'Scenario 3 - stiffness and damping', 'fig_s3_stiff.png')
    bar_scenario(d, 'S4_', ['1.00', '1.25', '1.50', '1.75', '2.00'],
                 'Scenario 4 - delay', 'fig_s4_delay.png',
                 xlabel=r'$\tau/\tau_0$')
    print('figures ->', OUT)


if __name__ == '__main__':
    main()
