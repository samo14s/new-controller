"""Forced actuator demand over speed and depth for the saturation experiments of Ozsoy et al. 2025.

Purpose.  The 2025 article predicts 'saturation islands' (its Figs. 10, 12): regions of speed and depth
where the settled forced vibration of the controlled cut asks the actuator for more than its 27 N.
Its frequency-domain model leaves out regeneration, which is exact for the tooth-periodic forced orbit
(x(t) - x(t - T) = 0 on it).  The same quantity in this repository is the forced demand of
src/feasibility.py: K_a g max |x_p'(t)| over the tooth period, from 40 tooth-passing harmonics of the
helical cutting force applied to the controlled FRF (N); its value over 27 N means that the forced
orbit's command exceeds the 9 V clip, and 27 N minus it is the residual authority in force units.

Map (re-clamped structure of the 2025 saturation tests: 128.1 Hz, 1.48 %, 1.13e7 N/m; DVF gain g):
  speeds 1700:10:2250 rpm, depths 0.25:0.25:28 mm; for each speed the linear limits a_lim(0) and a_lim(g)
  (src/zoa.py); the peak demand 'demand_N' and, as alternative readings of a measured 'amplitude', the
  amplitude of the tooth-passing harmonic of the demand ('h1_N') and sqrt(2) times its RMS ('rms2_N').
  Islands: demand_N > 27 N below a_lim(g).

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/ozsoy2025/forced_map.py [--g 253]
Writes verification/ozsoy2025/forced_map_g<g>.npz and .json (summary).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup2025 as S  # noqa: E402

import params as P  # noqa: E402
import zoa  # noqa: E402
import feasibility as F  # noqa: E402
from structure import Gcl  # noqa: E402


def demand_components(rpm, g, depths, n_harm=F.N_HARM, n_t=F.N_T):
    """Peak demand, tooth-harmonic amplitude and sqrt(2) RMS of K_a g x_p'(t) (N), per depth."""
    wT = 2 * np.pi * P.N_TEETH * rpm / 60.0
    m = np.arange(1, n_harm + 1)
    H = 1j * m * wT * Gcl(1j * m * wT, g)
    Fh = F.force_harmonics(depths, n_harm)[:, 1:]
    V = Fh * H[None, :] * P.GA_GAIN * g                   # complex harmonics of the demand (N), one-sided
    t = np.arange(n_t) / n_t
    E = np.exp(2j * np.pi * np.outer(m, t))
    d = 2 * np.real(V @ E)
    peak = np.abs(d).max(axis=1)
    h1 = 2 * np.abs(V[:, 0])
    rms2 = np.sqrt(2.0) * np.sqrt(np.sum(2 * np.abs(V) ** 2, axis=1))
    return peak, h1, rms2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=S.G_FIG6)
    ap.add_argument("--structure", default="reclamped", choices=["reclamped", "nominal"])
    a = ap.parse_args()
    S.apply_structure(**(S.RECLAMPED if a.structure == "reclamped" else S.NOMINAL))
    rpms = np.arange(1700.0, 2250.1, 10.0)
    depths = np.arange(0.25, 28.01, 0.25) * 1e-3
    peak = np.zeros((len(rpms), len(depths)))
    h1, rms2 = np.zeros_like(peak), np.zeros_like(peak)
    al0, alg = np.zeros(len(rpms)), np.zeros(len(rpms))
    for i, n in enumerate(rpms):
        peak[i], h1[i], rms2[i] = demand_components(n, a.g, depths)
        al0[i] = zoa.a_lim(n, 0.0)
        alg[i] = zoa.a_lim(n, a.g)
    # check against src/feasibility.forced_demand (the peak)
    chk = F.forced_demand(1900.0, a.g, np.array([5e-3, 10e-3]))
    i1900 = int(np.argmin(np.abs(rpms - 1900)))
    j5, j10 = int(np.argmin(np.abs(depths - 5e-3))), int(np.argmin(np.abs(depths - 10e-3)))
    stable = depths[None, :] < alg[:, None]
    island = stable & (peak > P.F_MAX)
    tag = f"{a.structure}_g{a.g:g}"
    np.savez_compressed(os.path.join(HERE, f"forced_map_{tag}.npz"), rpm=rpms, depth_mm=depths * 1e3,
                        demand_N=peak, h1_N=h1, rms2_N=rms2, a_lim0_mm=al0 * 1e3, a_lim_g_mm=alg * 1e3,
                        structure=json.dumps(S.structure()), g=a.g)
    summ = dict(note=__doc__.split("\n\n")[0], structure=S.structure(), g=a.g,
                check_peak_vs_feasibility=dict(here=[float(peak[i1900, j5]), float(peak[i1900, j10])],
                                               feasibility=[float(x) for x in chk]),
                island_cells=int(island.sum()), stable_cells=int(stable.sum()),
                island_rpm_range=[float(rpms[island.any(axis=1)].min()), float(rpms[island.any(axis=1)].max())]
                if island.any() else None,
                island_depth_mm_ranges=None)
    if island.any():
        cols = island.any(axis=0)
        dd = depths[cols] * 1e3
        # contiguous depth bands of the islands
        bands, start = [], None
        for k, c in enumerate(cols):
            if c and start is None:
                start = k
            if (not c or k == len(cols) - 1) and start is not None:
                end = k if c else k - 1
                bands.append([float(depths[start] * 1e3), float(depths[end] * 1e3)])
                start = None
        summ["island_depth_mm_ranges"] = bands
    with open(os.path.join(HERE, f"forced_map_{tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
