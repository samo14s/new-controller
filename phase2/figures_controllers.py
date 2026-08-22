"""
figures_controllers.py — the controller figures for Phase 3.
=============================================================
Five controllers designed under ONE protocol: the same plant, actuator, sensor,
sign, operating point, objective, constraints, optimiser and seeds.  The tables
say they were treated equally; these figures show what each one actually DOES,
and show the two constraints being met rather than asserted.

    fig_c1_bode      what each controller is: gain and phase against frequency
    fig_c2_limits    the two constraints drawn -- |S| against Ms <= 2, and the
                     actuator effort |K S P_f| against 450 V/N
    fig_c3_poles     the nominal closed loop without cutting: how much damping
                     each controller actually adds, mode by mode
    fig_c4_sweep     a_p,lim along the top edge
    fig_c5_time      the nominal cut in the time domain, displacement and
                     voltage, with the +/-450 V saturation drawn
    fig_c6_protocol  the tuning budget, and where each design ended up against
                     the two constraints

    python figures_controllers.py -> results/figures_controllers/*.png
"""
import os
import pickle
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from eval2 import nominal_poles
from fopid import ss_frf
from plate_model import build_plate, plant_frf
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = os.path.join(C.RESULTS, 'figures_controllers')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 1.7, 'figure.constrained_layout.use': True,
})
BOX = dict(fc='white', ec='0.86', lw=.7, pad=3.5)

SHOW = ['fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac']
LAB = dict(fopid='FOPID', lqg='LQG', mu_tdc='$\\mu$-TDC (Du 2024)',
           mu_phys_tdc='$\\mu$-TDC, physics set', ps_ac='PS-AC (proposed)',
           open='no control')
COL = dict(fopid='#C2410C', lqg='#7C3AED', mu_tdc='#0891B2',
           mu_phys_tdc='#15803D', ps_ac='#9F1239', open='0.6')
LS = dict(mu_phys_tdc='--')


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); print('  ', p)
    return p


def _ctrls():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    made = load_controllers(plate, plant)
    return plate, plant, {k: made[k](plant) for k in SHOW if k in made}


def _K_frf(c, om, x=0.0, tau=None):
    """Controller frequency response, delayed PD included when present."""
    ss, pd = c.at(x)
    K = ss_frf(ss, om) if ss is not None else np.zeros_like(om, complex)
    if pd is not None:
        tau = 60.0 / (3 * C.RPM_S) if tau is None else tau
        K = K + (float(pd[0]) + 1j * om * float(pd[1])) * np.exp(-1j * om * tau)
    return K


# ---------------------------------------------------------------------------
def fig_bode():
    plate, plant, cs = _ctrls()
    f = np.logspace(1.0, np.log10(6000), 2400)
    om = 2 * np.pi * f
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True,
                               height_ratios=[1.3, 1])
    for k, c in cs.items():
        K = _K_frf(c, om, x=0.5 * plate.lp)
        a.semilogx(f, 20 * np.log10(np.abs(K)), color=COL[k],
                   ls=LS.get(k, '-'), label=LAB[k])
        b.semilogx(f, np.angle(K) * 180 / np.pi, color=COL[k],
                   ls=LS.get(k, '-'), lw=1.3)
    for w, nm in zip(plate.omega_n[:2], ('mode 1', 'mode 2')):
        for ax in (a, b):
            ax.axvline(w / (2 * np.pi), color='0.35', lw=.9, ls=':')
        a.annotate(nm, (w / (2 * np.pi), a.get_ylim()[1]),
                   textcoords='offset points', xytext=(4, -12), fontsize=8,
                   color='0.35')
    a.set_ylabel('|K|  (dB, volts per metre)')
    a.set_title('What each controller is — gain')
    a.legend(loc='upper left', ncol=2, fontsize=8)
    b.set_ylabel('phase (deg, wrapped)'); b.set_xlabel('frequency (Hz)')
    b.set_yticks([-180, -90, 0, 90, 180])
    b.set_title('phase')
    b.text(0.985, 0.04,
           'the two $\\mu$ designs carry the Eq. (30) delayed PD; its\n'
           r'$e^{-j\omega\tau}$ turns through 360° every $1/\tau$ = 245 Hz,'
           '\nwhich is why only their phase cycles',
           transform=b.transAxes, fontsize=8, color='0.3', ha='right',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_c1_bode.png')


# ---------------------------------------------------------------------------
def fig_limits():
    plate, plant, cs = _ctrls()
    f = np.logspace(1.0, np.log10(6000), 1400)
    om = 2 * np.pi * f
    Pu, _ = plant_frf(plate, f, C.N_MODES)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for k, c in cs.items():
        K = _K_frf(c, om, x=0.5 * plate.lp)
        S = 1.0 / (1.0 - Pu * K)
        a.semilogx(f, np.abs(S), color=COL[k], ls=LS.get(k, '-'), label=LAB[k])
        v = np.zeros_like(f)
        for fr in C.POSITIONS_DESIGN:
            _, Pf = plant_frf(plate, f, C.N_MODES, x_force=fr * plate.lp)
            v = np.maximum(v, np.abs(K * S * Pf))
        b.loglog(f, v, color=COL[k], ls=LS.get(k, '-'), label=LAB[k])
    a.axhline(C.MS_MAX, color='k', lw=1.6, ls='--',
              label=f'constraint  $M_s \\leq$ {C.MS_MAX:.0f}')
    a.set_xlabel('frequency (Hz)'); a.set_ylabel('|S|')
    a.set_title('Modulus margin — the constraint, drawn')
    a.legend(loc='lower left', fontsize=7.6, ncol=2)
    a.set_ylim(0, 2.35)
    b.axhline(C.V_PER_N, color='k', lw=1.6, ls='--',
              label=f'constraint  {C.V_PER_N:.0f} V/N')
    b.set_xlabel('frequency (Hz)')
    b.set_ylabel('$|K S P_f|$  (volts per newton of cutting force)')
    b.set_title('Actuator effort — worst over the design positions')
    b.legend(loc='lower left', fontsize=7.6, ncol=2)
    b.set_ylim(3e-1, 1.2e3)
    b.text(0.5, 0.045,
           'the binding constraint is the SAME for every design: each sits on\n'
           'the $M_s$ bound and peaks near 150 V/N, a third of the effort bound',
           transform=b.transAxes, fontsize=8, color='0.3', ha='center',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_c2_limits.png')


# ---------------------------------------------------------------------------
def fig_poles():
    plate, plant, cs = _ctrls()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    rows, names = [], []
    for k, c in cs.items():
        ss, pd = c.at(0.5 * plate.lp)
        ev = nominal_poles(plate, ss, pd)
        ev = ev[np.imag(ev) > 0]
        fr = np.abs(ev) / (2 * np.pi)
        ze = -np.real(ev) / np.abs(ev)
        o = np.argsort(fr)
        keep = fr < 6000.0
        a.scatter(np.real(ev[keep]), np.imag(ev[keep]) / (2 * np.pi), s=34,
                  color=COL[k], label=LAB[k], zorder=3, alpha=.9)
        m = (fr[o] > 300) & (fr[o] < 5000)
        rows.append(100 * ze[o][m]); names.append(k)
    for w in plate.omega_n[:5]:
        a.scatter([-plate.zeta_modes[0] * w], [w / (2 * np.pi)], s=52,
                  facecolor='none', edgecolor='0.45', lw=1.2, zorder=2)
    a.axvline(0, color='k', lw=1.1)
    a.set_xlabel('Re $\\lambda$  (1/s)   — further left is better damped')
    a.set_ylabel('Im $\\lambda$ / 2$\\pi$  (Hz)')
    a.set_title('Closed-loop poles in the structural band, no cutting')
    a.legend(loc='lower left', fontsize=7.6)
    a.set_xscale('symlog', linthresh=1.0)
    a.set_ylim(0, 5200)
    a.text(0.985, 0.96, 'open circles = open-loop modes',
           transform=a.transAxes, fontsize=8, color='0.35', ha='right',
           va='top', bbox=BOX)

    n = min(min(len(r) for r in rows), 5)
    R = np.array([r[:n] for r in rows])
    x = np.arange(n); w = 0.8 / (len(rows) + 1)
    zopen = 100 * np.asarray(plate.zeta_modes[:n], float)
    b.bar(x - 0.4, zopen, w, color=COL['open'], label='open loop')
    for i, k in enumerate(names):
        b.bar(x - 0.4 + (i + 1) * w, R[i], w, color=COL[k], label=LAB[k])
    b.set_xticks(x, [f'mode {i+1}' for i in x])
    b.set_ylabel('closed-loop damping ratio (%)')
    b.set_yscale('log')
    b.set_title('Damping added, mode by mode')
    b.legend(loc='upper left', fontsize=7.4, ncol=2)
    b.set_ylim(1e-1, 3e2)
    b.text(0.5, 0.035,
           f'open-loop damping is {zopen[0]:.2f} % on mode 1 — a modal half-width\n'
           'of about 10 rad/s.  Every controller raises the first two modes by\n'
           'one to two DECADES; the higher modes are barely touched.',
           transform=b.transAxes, fontsize=8, color='0.3', ha='center',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_c3_poles.png')


# ---------------------------------------------------------------------------
def fig_sweep():
    with open(os.path.join(C.RESULTS, 'stage78.pkl'), 'rb') as fh:
        d = pickle.load(fh)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for k in ['open'] + SHOW:
        key = f'S2_{k}'
        if key not in d:
            continue
        y = np.asarray(d[key], float) * 1e3
        pos = np.linspace(0.0, 100.0, len(y))
        ax.plot(pos, y, 'o-', ms=4, color=COL[k], ls=LS.get(k, '-'),
                label=f'{LAB[k]}  —  min {y.min():.3f}, range {y.max()-y.min():.3f}')
    ax.set_xlabel('tool position along the top edge (% of $l_P$)')
    ax.set_ylabel('$a_{p,\\mathrm{lim}}$  (mm)')
    ax.set_title('Stability limit along the pass')
    ax.legend(loc='upper left', fontsize=8)
    ax.axvline(50, color='0.35', lw=.9, ls=':')
    ax.annotate('mode-2 node', (50, 0.05), textcoords='offset points',
                xytext=(4, 0), fontsize=8, color='0.35')
    ax.text(0.985, 0.06,
            'production is limited by the WORST position, not the mean:\n'
            'a flat curve is worth more than a high average',
            transform=ax.transAxes, fontsize=8, color='0.3', ha='right',
            va='bottom', bbox=BOX)
    return save(fig, 'fig_c4_sweep.png')


# ---------------------------------------------------------------------------
def fig_time():
    from simulate import MillingSimulation
    from sim_ctrl2 import ScheduledLTI
    plate, plant, cs = _ctrls()
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True,
                               height_ratios=[1.35, 1])
    umax = 0.0
    for k, c in list(cs.items()) + [('open', None)]:
        sim = MillingSimulation(plate, C.RPM_S, C.AP_S, ae=C.AE, fz=C.FZ,
                                sign=C.SIGN, n_modes=C.N_MODES, n_sub=C.N_SUB)
        ctl = None
        if c is not None:
            ctl = ScheduledLTI(c, sim.dt, plate.lp, tau=sim.tau,
                               feed=plant.feed_speed(), moving=False)
        r = sim.run(controller=ctl, T=0.06, moving=False)
        t = np.asarray(r['t'], float) * 1e3
        y = np.asarray(r['y_obs'], float) * 1e6
        u = np.asarray(r['u'], float)
        a.plot(t, y, color=COL[k], ls=LS.get(k, '-'), label=LAB[k], lw=1.4)
        if c is not None:
            b.plot(t, u, color=COL[k], ls=LS.get(k, '-'), lw=1.4,
                   label=f'{LAB[k]}  (peak {np.abs(u).max():.1f} V)')
            umax = max(umax, float(np.abs(u).max()))
    a.set_yscale('symlog', linthresh=3.0)
    a.set_ylabel('sensor displacement (μm)')
    a.set_title(f'Nominal cut — {C.RPM_S} rpm, $a_p$ = {C.AP_S*1e3:.1f} mm, '
                'tool held at $x_P$ = 0')
    a.legend(loc='upper left', fontsize=8, ncol=3)
    b.set_ylim(-1.9 * umax, 1.9 * umax)
    b.set_xlabel('time (ms)'); b.set_ylabel('control voltage (V)')
    b.set_title('Control effort')
    b.legend(loc='upper left', fontsize=7.6, ncol=3)
    b.text(0.985, 0.05,
           f'the $\\pm${C.U_SAT:.0f} V saturation is OFF SCALE by '
           f'{C.U_SAT/max(umax,1e-9):.0f}×\n'
           'at this depth — none of them comes near it',
           transform=b.transAxes, fontsize=8, color='0.3', ha='right',
           va='bottom', bbox=BOX)
    return save(fig, 'fig_c5_time.png')


# ---------------------------------------------------------------------------
def fig_protocol():
    from stage_common import n_params
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'rb') as fh:
        st = pickle.load(fh)
    ks = [k for k in SHOW if k in st]
    Ms = np.array([st[k]['Ms'] for k in ks])
    V = np.array([st[k]['V'] for k in ks])
    npar = np.array([n_params(k) for k in ks])
    order = np.array([st[k]['order'] for k in ks])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    y = np.arange(len(ks))
    a.barh(y + 0.19, Ms / C.MS_MAX, 0.34,
           color=[COL[k] for k in ks], label='$M_s$ / bound')
    a.barh(y - 0.19, V / C.V_PER_N, 0.34,
           color=[COL[k] for k in ks], alpha=.45, label='effort / bound')
    a.axvline(1.0, color='k', lw=1.8, ls='--', label='the constraint')
    a.set_yticks(y, [LAB[k] for k in ks], fontsize=8)
    a.set_xlabel('fraction of the constraint used')
    a.set_xlim(0, 1.42)
    a.set_ylim(-0.75, len(ks) - 0.25)
    a.set_title('Which constraint binds — and it is the same one for all')
    a.legend(loc='upper right', fontsize=7.8)
    for i in range(len(ks)):
        a.annotate(f'{Ms[i]:.3f}', (Ms[i] / C.MS_MAX, y[i] + 0.19),
                   textcoords='offset points', xytext=(4, -3), fontsize=7.6)
        a.annotate(f'{V[i]:.0f} V/N', (V[i] / C.V_PER_N, y[i] - 0.19),
                   textcoords='offset points', xytext=(4, -3), fontsize=7.6,
                   color='0.4')
    a.text(0.985, 0.035,
           'the modulus margin is ACTIVE for every design;\n'
           'the effort bound never is',
           transform=a.transAxes, fontsize=8, color='0.3', ha='right',
           va='bottom', bbox=BOX)

    x = np.arange(len(ks)); w = 0.38
    b.bar(x - w / 2, npar, w, color='#0891B2', label='tuned parameters')
    b.bar(x + w / 2, order, w, color='#C2410C', label='controller states')
    b.set_xticks(x, [LAB[k] for k in ks], fontsize=7.4, rotation=14)
    b.set_ylabel('count')
    b.set_title('Tuning budget and realisation cost')
    b.legend(loc='upper left', fontsize=8)
    for i in range(len(ks)):
        b.annotate(str(int(npar[i])), (i - w / 2, npar[i]),
                   textcoords='offset points', xytext=(0, 4), ha='center',
                   fontsize=8)
        b.annotate(str(int(order[i])), (i + w / 2, order[i]),
                   textcoords='offset points', xytext=(0, 4), ha='center',
                   fontsize=8)
    b.text(0.5, 0.60,
           f'PSO {C.OPT["n_particles"]}×{C.OPT["n_iter"]}, seeds {C.OPT["seeds"]},\n'
           'identical for every structure.\n'
           'PS-AC deliberately gets the SAME four\nparameters as LQG.',
           transform=b.transAxes, fontsize=8, color='0.3', ha='center', bbox=BOX)
    return save(fig, 'fig_c6_protocol.png')


def main():
    print('writing controller figures ->', OUT)
    fig_bode(); fig_limits(); fig_poles()
    fig_sweep(); fig_protocol(); fig_time()


if __name__ == '__main__':
    main()
