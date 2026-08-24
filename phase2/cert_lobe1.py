"""cert_lobe1.py — Stage B for the lobe-first champions: does the
whole-pass certificate survive the lobe push?

The machinery is the adiabatic (thin-cell) limit of the chain theorem,
exactly as the incumbent members were certified: solver-free point
certificates (sched_cert.certificate_direct) on a fine x-grid, scale-
balanced neighbour jump factors, v* = min_x 2 alpha / rho(x); the
constant-in-time uncertainties (a4, zeta) are SLICED — each realization
keeps one chain and the guarantee is the min over slices — and eta/xi
follow the frozen mid-pass tube (stated, not hidden).

Battery:
  * realizable champion at a_p = 0.10 mm (the incumbent's certified
    depth): 9 slices = a4 {lo, nom, hi} x zeta {0.8, 1.0, 1.2};
  * realizable champion, depth ladder 0.20 / 0.30 mm on the nominal
    slice (how far up does the certificate reach?);
  * filtered champion at a_p = 0.15 mm (its class's certified depth):
    a4 slices at nominal zeta, as the incumbent was reported.

alpha is FIXED at 2e-3 scaled (10 1/s) here, the incumbents' base rung;
the incumbents' final numbers used per-slice alpha optimisation, which
can only raise v* — so a v* that already clears the feed at fixed alpha
is a valid (conservative) certificate, and a shortfall is re-tried at
the alpha ladder before being called one.

Appends to results/log_lobe1.txt; writes results/cert_l.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import pickle
import sys
import time
import warnings

import numpy as np
from scipy.linalg import eigh

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers
import sched_cert as SC

NX = 121
ALPHAS = (2e-3, 1e-2)                 # base rung, then one rung up

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)


def slice_chain(cells, a_val, eta, xi, z, alpha, label):
    """Point certificates along one constant-parameter slice; returns
    v* in physical m/s, or None if any point certificate is missing."""
    xs = np.linspace(cells.edges[0], cells.edges[-1], NX)
    tau = cells.tau_scaled()
    Ps, Qs = [], []
    t0 = time.time()
    n_fail = 0
    for x in xs:
        ss, pd = cells.ctrl.at(float(x), 0.0)
        Acl, Adcl = cells._closed(ss, pd, x, eta, xi, z, a_val)
        r = SC.certificate_direct([(Acl, Adcl)], tau, alpha=alpha, eps=0.0,
                                  n_plant=cells.npl)
        if not r['feasible']:
            n_fail += 1
            Ps.append(None)
            Qs.append(None)
            continue
        Ps.append(r['P'])
        Qs.append(r['Q'])
    if n_fail:
        log(f'  {label}: {n_fail}/{NX} point certificates missing -- slice '
            f'not certifiable at this alpha  [{time.time() - t0:.0f}s]')
        return None
    h = xs[1] - xs[0]
    lnmu = []
    for k in range(NX - 1):
        m_fwd, m_bwd = 0.0, 0.0
        for Xa, Xb in ((Ps[k], Ps[k + 1]), (Qs[k], Qs[k + 1])):
            Xa = 0.5 * (Xa + Xa.T) + 1e-15 * np.eye(len(Xa))
            Xb = 0.5 * (Xb + Xb.T) + 1e-15 * np.eye(len(Xb))
            m_fwd = max(m_fwd, float(np.max(eigh(Xb, Xa,
                                                 eigvals_only=True))))
            m_bwd = max(m_bwd, float(np.max(eigh(Xa, Xb,
                                                 eigvals_only=True))))
        lnmu.append(0.5 * np.log(max(m_fwd * m_bwd, 1.0)))
    lnmu = np.asarray(lnmu)
    rho = lnmu / h
    v_star = float(np.min(2.0 * alpha / np.maximum(rho, 1e-300)))
    v_phys = v_star * cells.ws
    log(f'  {label}: rho max {rho.max():.3e} 1/m at '
        f'x = {xs[int(np.argmax(rho))] * 1e3:.1f} mm;  v* = '
        f'{v_phys * 1e3:.3f} mm/s  [{time.time() - t0:.0f}s]')
    return v_phys


def certify_member(kind, ap, slices, tag):
    pl = ControlledPlant(plate, ap=ap)
    mks = load_controllers(plate, pl)
    if kind not in mks:
        log(f'[{kind}] not on disk -- skipped')
        return None
    ct = mks[kind](pl)
    cells = SC.FamilyCells(pl, ct)
    v_feed = pl.feed_speed()
    a_lo, a_hi = cells._a_range()
    a_of = dict(lo=a_lo, nom=pl.a40, hi=a_hi)
    log(f'[{tag}]  {kind}  a_p = {ap * 1e3:.2f} mm  feed '
        f'{v_feed * 1e3:.2f} mm/s  ({len(slices)} slices, NX = {NX})')
    vs = {}
    for a_nm, z in slices:
        lbl = f'a4 {a_nm:3s} zeta {z:.1f}'
        v = None
        for alpha in ALPHAS:
            v = slice_chain(cells, a_of[a_nm], 0.5 * C.ETA_MAX, 0.5, z,
                            alpha, lbl + f' alpha {alpha:g}')
            if v is not None:
                break
        vs[f'{a_nm}_z{z:.1f}'] = np.nan if v is None else v
        if v is None:
            log(f'  {lbl}: NOT CERTIFIABLE at the alpha ladder')
    vals = [v for v in vs.values() if np.isfinite(v)]
    if len(vals) == len(vs):
        v_min = min(vals)
        log(f'  => {kind} @ {ap * 1e3:.2f} mm: v_cert(sliced) = '
            f'{v_min * 1e3:.3f} mm/s vs feed {v_feed * 1e3:.2f}: '
            + ('CERTIFIED at the actual feed'
               if v_min >= v_feed else
               f'short by {v_feed / max(v_min, 1e-300):.1f}x'))
    else:
        log(f'  => {kind} @ {ap * 1e3:.2f} mm: NOT fully certifiable '
            f'({len(vals)}/{len(vs)} slices)')
    return dict(v=vs, feed=v_feed)


if __name__ == '__main__':
    log('')
    log('=' * 78)
    log('STAGE B: WHOLE-PASS CERTIFICATE OF THE LOBE-FIRST CHAMPIONS  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 78)
    champs = pickle.load(open(os.path.join(C.RESULTS,
                                           'lobe_champions.pkl'), 'rb'))
    log(f'champions on record: {champs}')
    out = {}
    ri = champs['realizable']
    nine = [(a, z) for a in ('lo', 'nom', 'hi') for z in (0.8, 1.0, 1.2)]
    r = certify_member(ri, 0.10e-3, nine, 'RI champion, full 9-slice')
    if r:
        for k, v in r['v'].items():
            out[f'{ri}_010_{k}'] = v
        out[f'{ri}_feed'] = r['feed']
    for ap in (0.20e-3, 0.30e-3):
        r = certify_member(ri, ap, [('nom', 1.0)], 'RI champion, ladder')
        if r:
            out[f'{ri}_{int(ap * 1e6):03d}_nom'] = r['v']['nom_z1.0']
    rfa = champs['filtered']
    r = certify_member(rfa, 0.15e-3, [(a, 1.0) for a in ('lo', 'nom',
                                                         'hi')],
                       'filtered champion')
    if r:
        for k, v in r['v'].items():
            out[f'{rfa}_015_{k}'] = v
    np.savez_compressed(os.path.join(C.RESULTS, 'cert_l.npz'), **out)
    log('-> results/cert_l.npz')
    log('cert_lobe1 done.')
