"""
Phase 1c -- Section 4 of the paper on the virtual testbed.
==========================================================
The three published control runs, each in "experiment" (truth + noise) and
"simulation" (identified model), with the paper's parameters mapped through
the calibrated mu_scale:

  Fig. 10/11  120 Hz, S2, single channel (A2 assumed), L = 128, mu = 5e-6
  Fig. 13     280 Hz, S1, single channel (A1 assumed), L = 128, mu = 5e-6
  Fig. 14     85 Hz,  S2, two channels A1 + A2,       L = 64,  mu = 1e-5
  + 10 repeated "experiments" (independent noise) for the decay-time spread
  + Fig. 12: noise injected into the simulation at SNR 50/40/30/20 dB.
Writes results/control.txt and figures/fig_control.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import Tee, identify, Testbed, FIGS
from control import Plants, run_case
import calib

log = Tee('control.txt')
cal = calib.load()
ZETA, KAPPA = cal['zeta'], cal['mu_scale']
log(f"calibration: zeta = {ZETA:.5f}, mu_scale = {KAPPA:.4f} "
    f"({'calibrated' if cal['calibrated'] else 'UNCALIBRATED'})")
T, M, _, _ = identify(Testbed(zeta=ZETA))
P = Plants(T, M)
fs = T.fs

CASES = [  # label, f, sensor, actuators, mu_paper, L, paper numbers
    ('Fig. 10-11: 120 Hz / S2 / A2', 120.0, 1, [1], 5e-6, 128,
     dict(t=(4.121, 3.763), amp=(0.272, 0.255), y=([1.570], [1.465]), db=70, red=98.16)),
    ('Fig. 13: 280 Hz / S1 / A1', 280.0, 0, [0], 5e-6, 128,
     dict(t=(1.832, 2.510), amp=(0.341, 0.339), y=([1.160], [1.195]), db=79, red=97.95)),
    ('Fig. 14: 85 Hz / S2 / A1+A2', 85.0, 1, [0, 1], 1e-5, 64,
     dict(t=(5.165, 4.577), amp=(0.410, 0.422), y=([1.901, 2.215], [1.718, 1.990]), db=67, red=99.51)),
]


def tone_amp(e, f, k0, k1):
    """Single-bin DFT amplitude at f over samples [k0, k1) (integer periods)."""
    k = np.arange(k0, k1)
    return 2 * np.abs(np.mean(e[k0:k1] * np.exp(-2j * np.pi * f * k / fs)))


fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for row, (label, f, s, acts, mu_p, L, pp) in enumerate(CASES):
    log(f"\n=== {label}: L = {L}, mu_paper = {mu_p:g} -> mu_testbed = {KAPPA * mu_p:.3e} ===")
    res = {}
    for which in ('truth', 'rom'):
        r = run_case(P, which, f, s, acts, KAPPA * mu_p, L, return_signals=True)
        e, k_on = r['e'][:, 0], r['k_on']
        a0 = tone_amp(e, f, 0, k_on)
        a1 = tone_amp(e, f, len(e) - int(2 * fs), len(e))
        res[which] = r
        db = 20 * np.log10(a0 / a1)
        red = 100 * (1 - r['amp_end'] / r['amp0'])
        log(f"  {which:5s}: decay time {r['t_c']:.3f} s | amplitude {r['amp0']:.4f} -> {r['amp_end']:.2e} V"
            f" ({red:.2f} % reduction) | tone reduction {db:.1f} dB | actuating peak "
            f"{np.round(r['y_peak'], 3)} V")
        ax = axes[row, 0 if which == 'truth' else 1]
        t = (np.arange(len(e)) - k_on) / fs
        ax.plot(t, e, lw=0.4, color='C0' if which == 'truth' else 'C3')
        ax.set_title(f"{label} -- {'virtual experiment' if which == 'truth' else 'virtual simulation'}"
                     f" (t5 = {r['t_c']:.2f} s)", fontsize=9)
        ax.set_xlabel('time after control on [s]')
        ax.set_ylabel('sensing voltage [V]')
    log(f"  paper: decay {pp['t'][0]} s (exp) / {pp['t'][1]} s (sim); amplitude {pp['amp'][0]} / {pp['amp'][1]} V;"
        f" actuating peaks {pp['y'][0]} / {pp['y'][1]} V; {pp['db']} dB; {pp['red']} % reduction")
    log(f"  ratio exp/sim decay time: testbed {res['truth']['t_c'] / res['rom']['t_c']:.3f}, "
        f"paper {pp['t'][0] / pp['t'][1]:.3f}")
    # 10 repeated experiments
    reps = [run_case(P, 'truth', f, s, acts, KAPPA * mu_p, L, seed=100 + i) for i in range(10)]
    tc = np.array([q['t_c'] for q in reps])
    log(f"  10 repeated virtual experiments: decay {tc.mean():.3f} +- {tc.std(ddof=1):.3f} s"
        "   (paper: +-0.014 / 0.015 / 0.031 s)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig_control.png'), dpi=130)

# Fig. 12: noise injected into the simulation
log("\n=== Fig. 12: noise injected in the simulation, 120 Hz / S2 ===")
r0 = run_case(P, 'rom', 120.0, 1, [1], KAPPA * 5e-6, 128)
for snr in (50, 40, 30, 20):
    sig = r0['amp0'] / np.sqrt(2) / 10**(snr / 20)
    r = run_case(P, 'rom', 120.0, 1, [1], KAPPA * 5e-6, 128, noise_sigma=sig, return_signals=True)
    e = r['e'][:, 0]
    a1 = tone_amp(e, 120.0, len(e) - int(2 * fs), len(e))
    log(f"  SNR {snr} dB: decay {r['t_c']:.2f} s (noise-free {r0['t_c']:.2f} s), residual tone "
        f"{a1:.2e} V, residual broadband RMS {np.sqrt(np.mean(e[-int(2 * fs):]**2)):.2e} V")
log("\nsaved figures/fig_control.png")
