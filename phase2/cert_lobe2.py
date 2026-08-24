"""cert_lobe2.py — per-slice alpha optimisation of the RFA-L whole-pass
certificate, by the incumbents' own protocol.

cert_lobe1 reported the champion at the fixed base rate alpha = 2e-3
scaled (10 1/s): v* = 49.5 mm/s on the three a4 slices.  The incumbent
members' final numbers used a per-slice alpha ladder (2e-3, 5e-3, 1e-2,
2e-2), keeping the best feasible v* per slice — v* scales with alpha
while the point certificates stay feasible, so the ladder can only
raise the number.  This script applies exactly that ladder to the
lobe-first champion at a_p = 0.15 mm over the full 9 slices
(a4 {lo, nom, hi} x zeta {0.8, 1.0, 1.2}), then:

  * validates the winning slice chains: every point certificate is
    re-verified at its interval midpoint (static interval validity, the
    reading of Theorem 2's hypothesis checks);
  * confirms grid convergence of the binding slice at NX = 121/241/481.

v_cert(final) = min over slices of the per-slice best v*.

Appends to results/log_lobe1.txt; writes results/cert_l2.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
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

KIND = 'ps_ac_rfa_l2'
AP = 0.15e-3
ALPHAS = (2e-3, 5e-3, 1e-2, 2e-2)

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)


def chain(cells, a_val, z, alpha, NX, validity=False):
    """One slice chain; returns dict(v_star, rho_max, x_worst, valid) in
    scaled units, or None if any point certificate is missing."""
    xs = np.linspace(cells.edges[0], cells.edges[-1], NX)
    tau = cells.tau_scaled()
    eta, xi = 0.5 * C.ETA_MAX, 0.5
    Ps, Qs, lams, cls = [], [], [], []
    for x in xs:
        ss, pd = cells.ctrl.at(float(x), 0.0)
        Acl, Adcl = cells._closed(ss, pd, x, eta, xi, z, a_val)
        r = SC.certificate_direct([(Acl, Adcl)], tau, alpha=alpha, eps=0.0,
                                  n_plant=cells.npl)
        if not r['feasible']:
            return None
        Ps.append(r['P'])
        Qs.append(r['Q'])
        lams.append(r['lam'])
        cls.append((Acl, Adcl))
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
    out = dict(v_star=float(np.min(2.0 * alpha / np.maximum(rho, 1e-300))),
               rho_max=float(rho.max()),
               x_worst=float(xs[int(np.argmax(rho))]))
    if validity:
        ok = 0
        for k in range(NX - 1):
            xm = 0.5 * (xs[k] + xs[k + 1])
            ss, pd = cells.ctrl.at(float(xm), 0.0)
            Aclm, Adclm = cells._closed(ss, pd, xm, eta, xi, z, a_val)
            v = SC.verify_certificate([(Aclm, Adclm)], tau, alpha, 0.0,
                                      cells.npl, Ps[k], Qs[k], lams[k])
            ok += bool(v['phi_max'] < 0.0 and v['p_min'] > 0.0
                       and v['q_min'] > 0.0)
        out['valid'] = ok / (NX - 1)
    return out


if __name__ == '__main__':
    log('')
    log('=' * 78)
    log('ALPHA OPTIMISATION OF THE RFA-L WHOLE-PASS CERTIFICATE  '
        + time.strftime('%Y-%m-%d %H:%M'))
    log('=' * 78)
    pl = ControlledPlant(plate, ap=AP)
    ct = load_controllers(plate, pl)[KIND](pl)
    cells = SC.FamilyCells(pl, ct)
    v_feed = pl.feed_speed()
    a_lo, a_hi = cells._a_range()
    a_of = dict(lo=a_lo, nom=pl.a40, hi=a_hi)
    log(f'{KIND} @ {AP * 1e3:.2f} mm, feed {v_feed * 1e3:.2f} mm/s, '
        f'NX = 121, alpha ladder {ALPHAS} (scaled; x{cells.ws:.0f} 1/s '
        'physical)')
    out = {}
    worst = (np.inf, None, None)
    for a_nm in ('lo', 'nom', 'hi'):
        for z in (0.8, 1.0, 1.2):
            t0 = time.time()
            best = (None, None)
            for al in ALPHAS:
                r = chain(cells, a_of[a_nm], z, al, 121)
                if r is not None and (best[0] is None
                                      or r['v_star'] > best[0]):
                    best = (r['v_star'], al)
            v_phys = None if best[0] is None else best[0] * cells.ws
            out[f'{a_nm}_z{z:.1f}_v'] = np.nan if v_phys is None else v_phys
            out[f'{a_nm}_z{z:.1f}_alpha'] = (np.nan if best[1] is None
                                             else best[1])
            log(f'  a4 {a_nm:3s} zeta {z:.1f}: '
                + ('no feasible alpha' if best[0] is None else
                   f'v* = {v_phys * 1e3:7.3f} mm/s at alpha = '
                   f'{best[1]:.0e} ({best[1] * cells.ws:.0f} 1/s)')
                + f'  [{time.time() - t0:.0f}s]')
            if v_phys is not None and v_phys < worst[0]:
                worst = (v_phys, (a_nm, z), best[1])
    if worst[1] is not None:
        (a_nm, z), al = worst[1], worst[2]
        log(f'binding slice: a4 {a_nm} zeta {z:.1f} at alpha = {al:.0e}')
        r = chain(cells, a_of[a_nm], z, al, 241, validity=True)
        log(f'  NX = 241: '
            + ('infeasible' if r is None else
               f'v* = {r["v_star"] * cells.ws * 1e3:.3f} mm/s, midpoint '
               f'validity {r["valid"] * 100:.0f}%'))
        r4 = chain(cells, a_of[a_nm], z, al, 481)
        log(f'  NX = 481: '
            + ('infeasible' if r4 is None else
               f'v* = {r4["v_star"] * cells.ws * 1e3:.3f} mm/s'))
        out['binding_v241'] = (np.nan if r is None
                               else r['v_star'] * cells.ws)
        out['binding_valid241'] = np.nan if r is None else r['valid']
        out['binding_v481'] = (np.nan if r4 is None
                               else r4['v_star'] * cells.ws)
        vals = [v for k, v in out.items()
                if k.endswith('_v') and np.isfinite(v)]
        v_cert = min(vals)
        log(f'=> v_cert(alpha-opt, 9 slices) = {v_cert * 1e3:.3f} mm/s vs '
            f'feed {v_feed * 1e3:.2f} mm/s '
            f'({v_cert / v_feed:.1f}x rate margin)')
        out['v_cert'] = v_cert
        out['v_feed'] = v_feed
    np.savez_compressed(os.path.join(C.RESULTS, 'cert_l2.npz'), **out)
    log('-> results/cert_l2.npz')
    log('cert_lobe2 done.')
