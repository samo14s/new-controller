"""run_psacrf.py — design and judge the actuator-aware scheduled member.

The design story lives in docs/09 sec. 5-bis and in this module's own log,
round by round; the short of it:

  * no fixed filter works on the STORED gains (broadband 9.7e7 high-band
    gain; the scan is the record), so gains and filter are designed together;
  * the member is ps_ac_rfa (act_filter.py): the filter INSIDE the design
    model -- augmented LQR on [filter; plate] solved in scaled coordinates,
    plate-only observer driven by the exact filtered command -- the one of
    three structures that stands (the series member breaks certainty
    equivalence and collapsed parametrically twice; the observer-consistent
    shortcut has an unstable internal observer/filter loop);
  * the search is the stage-3 PSO exactly (same J, constraints, optimiser,
    seeds, budget) plus this member's DECLARED synthesis pressure, as the
    D-K machinery is mu-TDC's: the 8-corner envelope screens and the
    structured screen MuScreen -- the G1 quantity itself, scalar-D at 19
    frequencies on pre-prepared plants (rounds 1-2 proved cheaper proxies
    do not track mu; round 3's inverted penalty ladder is the recorded
    lesson in shaping the ladder strictly);
  * the complex reading floors near mu ~ 2.4 for this structure (the
    filter's phase at mode 2 eats the active damping the parametric test
    demands -- the open-loop COMPLEX mu of the plate is ~40); the mixed
    real/complex re-reading of the same quantity lives in mu_real.py.

Success criteria, PRE-DECLARED (docs/09 sections 2 and 5-bis):

    G1  sup over the 21 nodes of mu_RS(point, ACTUATOR INCLUDED)      < 1
    G2  the same with the interpolation cell of position              < 1
    G3  S1 nominal floor (stage-7 protocol)             >= 0.6821 mm (mu-TDC)
    G4  a_p^inf certified by crossing                   >= 0.3954 mm (PS-TDC-R)
    G5  no scenario of the four collapses               (> 0.05 mm)
    G6  whole-pass certificate (sched_cert): alpha > 0 with v_cert >= the
        actual feed speed at the reference depth

Anything short is reported as exactly that.

    python phase2/run_psacrf.py                 (design + judge + certify)
    python phase2/run_psacrf.py design          (stages singly)
    python phase2/run_psacrf.py judge21 certify whole_pass scenarios verdict

Appends to results/log_psacrf.txt; stores in the shared stage pkls + npz.
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from act_filter import band_proxy
from design2 import Design2
from eval2 import evaluate
from pso import pso

OUT = C.RESULTS
KIND = 'ps_ac_rfa'
MU_SCR_MAX = 0.90
PROXY_MAX = 0.55
RE_VERT = -0.5          # corner poles: delay-free loop alive on the envelope
MS_VERT = 2.0           # corner modulus margin: same bar as the nominal one
                        # (the stored PS-AC-R sits at 1.14 on the corners, the
                        #  fragile round-1 winner at 2.73 -- room and contrast)
TARGET = dict(g3_floor=0.6821e-3, g4_apinf=0.3954e-3, g4_amb=0.4364e-3)
LOG = open(os.path.join(OUT, 'log_psacrf.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def upd(fname, patch):
    p = os.path.join(OUT, fname)
    store = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
    store.update(patch)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    return store


def proxy_of(plate, ctrl):
    """The actuator-band proxy, worst over the three design positions."""
    v = 0.0
    for fr in C.POSITIONS_DESIGN:
        ss, _ = ctrl.at(fr * plate.lp)
        hi, lo = band_proxy(plate, ss)
        v = max(v, hi, lo)
    return v


# -- the envelope screens: the physics corners, cheaply -----------------------
_F_SCREEN = np.logspace(0.5, 4.1, 140)


def vertex_screens(plate, plant, ctrl):
    """(worst max-Re, worst Ms) of the closed loop over the eight corners of
    the physics set -- removal x force coefficient x damping -- at the three
    design positions.  The first PSO round showed why this pressure is
    needed: J plus the band proxy alone steered the search into designs
    whose PARAMETRIC mu collapsed at mode 2 (1.7-2.1 against the stored
    member's 0.58) while every nominal screen looked fine.  The corners are
    the same family the mu judge certifies, so this is envelope design made
    explicit, at FRF cost."""
    om = 2 * np.pi * _F_SCREEN
    n = plant.n
    worst_re, worst_ms = -np.inf, 0.0
    corners = [(eta, am, z) for eta in (0.0, C.ETA_MAX)
               for am in (C.ALPHA_LO / 1.6, C.ALPHA_HI / 1.6)
               for z in (C.ZETA_LO, C.ZETA_HI)]
    for frx in C.POSITIONS_DESIGN:
        x = frx * plate.lp
        ss, _ = ctrl.at(x)
        Ac, Bc, Cc, Dc = [np.atleast_2d(np.asarray(m, float)) for m in ss]
        Bc = Bc.reshape(-1, 1); Cc = Cc.reshape(1, -1)
        nc = Ac.shape[0]
        K = ss_frf_cached(ss, om)
        for eta, am, z in corners:
            A, _, B, _, Cy = plant.matrices(x_pos=x,
                                            a4=am * 1.6 * plant.abar4,
                                            eta=eta, zeta_scale=z)
            ncl = A.shape[0] + nc
            M = np.zeros((ncl, ncl))
            M[:2 * n, :2 * n] = A + float(Dc[0, 0]) * (B @ Cy)
            M[:2 * n, 2 * n:] = B @ Cc
            M[2 * n:, :2 * n] = Bc @ Cy
            M[2 * n:, 2 * n:] = Ac
            worst_re = max(worst_re, float(np.max(np.linalg.eigvals(M).real)))
            if worst_re > 0.0:
                return worst_re, np.inf          # already dead: stop paying
            Pu = _frf_siso(A, B, Cy, om)
            worst_ms = max(worst_ms, float(np.max(np.abs(
                1.0 / (1.0 - Pu * K)))))
    return worst_re, worst_ms


def _frf_siso(A, B, Cy, om):
    n = A.shape[0]
    I = np.eye(n)
    out = np.empty(len(om), complex)
    for k, w in enumerate(om):
        out[k] = (Cy @ np.linalg.solve(1j * w * I - A, B))[0, 0]
    return out


def ss_frf_cached(ss, om):
    from fopid import ss_frf
    return ss_frf(ss, om)


# -- the direct structured screen: scalar-D mu at key frequencies -------------
class MuScreen:
    """The G1 quantity itself, made cheap enough for the search loop.

    Rounds 1-2 taught the lesson twice: neither the band proxy nor the
    corner screens track the structured test -- designs cleared both and
    collapsed under mu (1.7-2.8 at the modes).  So the screen IS mu now:
    the scaled generalized plants of the judge, prepared ONCE per node, and
    the scalar-D upper bound evaluated at a fixed handful of frequencies
    around the modes and across the actuator band (~0.3 s per candidate).
    Scalar-D over-estimates the tight bound and a 15-point grid can miss a
    peak, so the full judge still has the last word on the winner."""

    F_SCR = np.array([520.0, 565.0, 610.0, 700.0, 900.0, 1050.0, 1100.0,
                      1178.0, 1300.0, 1600.0, 2000.0, 2400.0, 2790.0,
                      3000.0, 3350.0, 3700.0, 4000.0, 4120.0, 4400.0])

    def __init__(self, plate, nodes=(0.0, 0.5)):
        import control as ct
        from mu_scheduled import plant_at, POS
        from dk_synthesis import prepare_plant
        self.plate = plate
        self.nodes = nodes
        self.data = []
        for frx in nodes:
            U = plant_at(plate, frx * plate.lp)
            P = U.generalized_plant()
            Ps, info = prepare_plant(P)
            keep = [j for j, (nm, _) in enumerate(U.blocks_used)
                    if nm not in POS]
            ch = np.concatenate([U.idx[U.blocks_used[j][0]] for j in keep])
            rows = np.concatenate([[0, 1], 2 + ch]).astype(int)
            cols = np.concatenate([[0], 1 + ch]).astype(int)
            blocks = ([('F', 2, 1)]
                      + [('S', U.blocks_used[j][1], U.blocks_used[j][1])
                         for j in keep])
            self.data.append(dict(Ps=ct.ss(Ps), info=info, rows=rows,
                                  cols=cols, blocks=blocks,
                                  ncon=P['ncon'], nmeas=P['nmeas']))

    def __call__(self, ctrl, limit=None):
        import control as ct
        from cross_mu import frf
        from dk_synthesis import mu_upper_bound
        worst = 0.0
        for frx, d in zip(self.nodes, self.data):
            ss, _ = ctrl.at(frx * self.plate.lp)
            A, B, Cm, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
            w0, u0, y0 = d['info']['w0'], d['info']['u0'], d['info']['y0']
            Ks = ct.ss(A / w0, B.reshape(-1, 1) / w0 * y0,
                       Cm.reshape(1, -1) / u0, D * y0 / u0)
            T = ct.ss(d['Ps'].lft(Ks, d['ncon'], d['nmeas']))
            w = 2 * np.pi * self.F_SCR / w0
            H = frf(T, w)[:, d['rows']][:, :, d['cols']]
            d0 = None
            for k in range(len(w)):
                m, d0 = mu_upper_bound(H[k], d['blocks'], d0)
                worst = max(worst, m)
                if limit is not None and worst > limit:
                    return worst
        return worst


# ---------------------------------------------------------------------------
def design(plate, plant):
    log('')
    log(f'--- DESIGN {KIND}: stage-3 PSO + the member\'s declared design '
        'pressure ---')
    log('  member: the augmented structure (filter INSIDE the design model,')
    log('  certainty equivalence intact -- act_filter.ps_ac_rfa)')
    log(f'  screen 1 (corners): poles <= {RE_VERT} 1/s, Ms <= {MS_VERT} on '
        'the 8 physics corners x 3 positions')
    log(f'  screen 2 (structured): scalar-D mu_RS, actuator included, on the '
        f'{len(MuScreen.F_SCR)}-frequency grid at 2 nodes <= {MU_SCR_MAX}')
    log('  (rounds 1-2, recorded above, taught this twice: the band proxy')
    log('   and the corner screens do not track the structured test -- the')
    log('   screen is mu itself now.  All of it is this member\'s declared')
    log('   synthesis pressure, as the D-K machinery is mu-TDC\'s.)')
    d = Design2(KIND, plant, plate, None)
    scr = MuScreen(plate)

    def fit(u):
        # a strict penalty LADDER: passing a gate always pays, whatever the
        # state of the next one.  Round 3 taught this the hard way: with the
        # mu penalty reaching -173 against the corner-screen floor of -60,
        # the swarm learned to stand at the corner-screen boundary forever
        # (all three seeds ended at exactly -60.0x).
        try:
            c = d.build(u)
        except Exception:
            return -1e4
        try:
            re_v, ms_v = vertex_screens(plate, plant, c)
        except Exception:
            return -1e4
        pen = 0.0
        if re_v > RE_VERT:
            pen += 10.0 * min((re_v - RE_VERT) / 10.0, 10.0)
        if ms_v > MS_VERT:
            pen += 10.0 * min(ms_v / MS_VERT - 1.0, 10.0)
        if pen > 0.0:
            return -300.0 - pen               # worst band: [-410, -300]
        try:
            mu_s = scr(c, limit=10.0)
        except Exception:
            return -1e4
        if mu_s > MU_SCR_MAX:
            return -200.0 - 10.0 * min(mu_s / MU_SCR_MAX - 1.0, 8.0)
            #                                   middle band: [-280, -200]
        J, info = evaluate(plate, c, detail=True)
        if not info['feasible']:
            return max(-140.0, -100.0 + J / 50.0)   # band: [-140, -100]
        return J                                    # the real objective

    t0 = time.time()
    best = (None, -np.inf)
    for sd in C.OPT['seeds']:
        t = time.time()
        x, J, info = pso(fit, d.n, seed=sd, verbose=False)
        log(f'  seed {sd}: J = {J:+.4f}   [{time.time()-t:.0f}s]')
        if J > best[1]:
            best = (x, J)
    x, Jc = best
    c = d.build(x)
    J, info = evaluate(plate, c, detail=True)
    p = proxy_of(plate, c)
    mu_s = scr(c)
    params = d.decode(x)
    log(f'  winner: J = {J:+.5f}  mu_screen = {mu_s:.3f}  proxy = {p:.3f}  '
        f'order {c.order}  {c.n_params} parameters  [{time.time()-t0:.0f}s]')
    log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
        f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in params.items()))
    upd('stage3_controllers.pkl',
        {KIND: dict(x=x, J=float(J), params=params, n_params=c.n_params,
                    order=c.order, Ms=float(info['Ms']), V=float(info['V']),
                    proxy=float(p), mu_screen=float(mu_s))})
    log('  -> stage3_controllers.pkl')


# ---------------------------------------------------------------------------
def design_real(plate, plant):
    """Round 5 -- M1-real: recover the performance under the REAL pressures.

    The mixed reading showed the complex mu screen was overcharging: every
    filtered member passes G1(real) at 0.25-0.49 while the complex screen
    priced them at 2.4-10.  So the design pressure drops to what the real
    physics actually bills:

      * the corner screens (real-parameter envelope: poles and Ms on the 8
        physics corners x 3 positions) -- unchanged;
      * the actuator-band proxy RELAXED to hi <= 0.85, lo <= 1.6 (the
        actuator block stays complex under the mixed reading, so the
        high band still needs real pressure; the mode band was the J killer
        at 0.55 and the 2x-3x G1(real) margins of rounds 2/4 buy the slack);
      * realisability: controller poles <= 2*pi*20 kHz, so the winner stays
        certifiable by construction (the legacy members' 1.4-8.8 MHz
        parasitic modes are what locked them out of the SDP);
      * J and the three shared constraints exactly as stage 3.

    The mixed judge has the last word on the winner (judge stages).  The
    round-4 entry is preserved as ps_ac_rfa_r4 before this round overwrites
    the live key, so every number in docs/09 stays reconstructible.
    """
    PROXY_HI, PROXY_LO, POLE_MAX = 0.85, 1.6, 2 * np.pi * 20e3
    log('')
    log(f'--- DESIGN ROUND 5 ({KIND}) - M1-real: J under the real '
        'pressures ---')
    log(f'  corners: poles <= {RE_VERT} 1/s, Ms <= {MS_VERT} (8 corners x 3'
        ' positions)')
    log(f'  actuator proxy: hi <= {PROXY_HI} (2.2-4.8k), lo <= {PROXY_LO} '
        '(0.3-1.4k), 3 positions')
    log(f'  realisability: controller poles <= {POLE_MAX:.0f} rad/s')
    log('  (the complex mu screen is gone -- the mixed reading showed it')
    log('   overcharging by 5-20x; the mixed judge rules on the winner.)')
    d = Design2(KIND, plant, plate, None)

    def fit(u):
        try:
            c = d.build(u)
        except Exception:
            return -1e4
        try:
            ss, _ = c.at(0.05)
            ev = np.linalg.eigvals(np.atleast_2d(np.asarray(ss[0], float)))
            if np.abs(ev).max() > POLE_MAX:
                return -400.0 - min(np.abs(ev).max() / POLE_MAX, 50.0)
            re_v, ms_v = vertex_screens(plate, plant, c)
        except Exception:
            return -1e4
        pen = 0.0
        if re_v > RE_VERT:
            pen += 10.0 * min((re_v - RE_VERT) / 10.0, 10.0)
        if ms_v > MS_VERT:
            pen += 10.0 * min(ms_v / MS_VERT - 1.0, 10.0)
        if pen > 0.0:
            return -300.0 - pen
        try:
            hi = lo = 0.0
            for fr in C.POSITIONS_DESIGN:
                h, l = band_proxy(plate, c.at(fr * plate.lp)[0])
                hi, lo = max(hi, h), max(lo, l)
        except Exception:
            return -1e4
        pen = 0.0
        if hi > PROXY_HI:
            pen += 10.0 * min(hi / PROXY_HI - 1.0, 8.0)
        if lo > PROXY_LO:
            pen += 10.0 * min(lo / PROXY_LO - 1.0, 8.0)
        if pen > 0.0:
            return -200.0 - pen
        J, info = evaluate(plate, c, detail=True)
        if not info['feasible']:
            return max(-140.0, -100.0 + J / 50.0)
        return J

    t0 = time.time()
    best = (None, -np.inf)
    for sd in C.OPT['seeds']:
        t = time.time()
        x, J, info = pso(fit, d.n, seed=sd, verbose=False)
        log(f'  seed {sd}: J = {J:+.4f}   [{time.time()-t:.0f}s]')
        if J > best[1]:
            best = (x, J)
    x, _ = best
    c = d.build(x)
    J, info = evaluate(plate, c, detail=True)
    p = proxy_of(plate, c)
    params = d.decode(x)
    ev = np.linalg.eigvals(np.atleast_2d(np.asarray(c.at(0.05)[0][0], float)))
    log(f'  winner: J = {J:+.5f}  proxy = {p:.3f}  max|pole| = '
        f'{np.abs(ev).max():.2e} rad/s  order {c.order}  '
        f'{c.n_params} parameters  [{time.time()-t0:.0f}s]')
    log(f'  Ms = {info["Ms"]:.3f}   effort = {info["V"]:.1f} V/N   '
        f'slowest nominal pole = {info["max_re"]:.1f} 1/s')
    log('  parameters: ' + ', '.join(f'{k}={v:.4g}'
                                     for k, v in params.items()))
    store = pickle.load(open(os.path.join(OUT, 'stage3_controllers.pkl'),
                             'rb'))
    if KIND in store and KIND + '_r4' not in store:
        store[KIND + '_r4'] = store[KIND]      # keep round 4 reconstructible
    store[KIND] = dict(x=x, J=float(J), params=params, n_params=c.n_params,
                       order=c.order, Ms=float(info['Ms']),
                       V=float(info['V']), proxy=float(p))
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'wb') as f:
        pickle.dump(store, f)
    log('  -> stage3_controllers.pkl (round 4 preserved as ps_ac_rfa_r4)')


def judge21(plate, plant):
    """G1/G2: mu_RS per scheduling node, ACTUATOR INCLUDED, at all 21 nodes
    (point) and with the interpolation cell of position (cell) -- the gate
    quantity of docs/09, judged by the same machinery as every design."""
    from mu_scheduled import plant_at, mu_rs, POS, N_NODES
    from robust_design import freq_grid
    from stage_common import load_controllers

    ctrl = load_controllers(plate, plant)[KIND](plant)
    f = freq_grid()
    xs = np.linspace(0.0, plate.lp, N_NODES)
    cell_h = plate.lp / (N_NODES - 1) / 2.0

    log('')
    log('--- JUDGE ps_ac_rf: mu_RS at the 21 nodes, actuator block INCLUDED')
    t0 = time.time()
    pt, cl = [], []
    for x in xs:
        ss, _ = ctrl.at(float(x))
        pt.append(mu_rs(plant_at(plate, x), ss, f, drop=POS,
                        actuator=True)[0])
        cl.append(mu_rs(plant_at(plate, x, cell_h), ss, f,
                        actuator=True)[0])
        log(f'    x = {x*1e3:5.1f} mm   point {pt[-1]:7.3f}   '
            f'cell {cl[-1]:7.3f}')
    pt, cl = np.array(pt), np.array(cl)
    log(f'  G1 sup(point) = {pt.max():.3f}  at x = {xs[pt.argmax()]*1e3:.1f} '
        f'mm   {"< 1: MET" if pt.max() < 1 else ">= 1: NOT MET"}')
    log(f'  G2 sup(cell)  = {cl.max():.3f}  at x = {xs[cl.argmax()]*1e3:.1f} '
        f'mm   {"< 1: MET" if cl.max() < 1 else ">= 1: NOT MET"}')
    log(f'  (references: best fixed design 1.205; scheduled LQG members 85.8;'
        f' chain reproducible optimum 1.73)   [{time.time()-t0:.0f}s]')
    np.savez(os.path.join(OUT, 'psacrf_mu.npz'), x=xs, point=pt, cell=cl,
             cell_h=cell_h)
    log('  -> results/psacrf_mu.npz')


# ---------------------------------------------------------------------------
def certify(plate):
    """Crossing + margins + common-P LK: the exact stage-56 block."""
    import certify2 as CF2
    from stage_common import load_controllers

    tau0 = 60.0 / (3 * C.RPM_S)
    etas = tuple(np.linspace(0.0, C.ETA_MAX, 3))
    zetas = (C.ZETA_LO, C.ZETA_HI)
    full = dict(n_pos=9, etas=etas, zetas=zetas, xis=(0.5, 1.0))
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    mk = load_controllers(plate, plant0)[KIND]
    c0 = mk(plant0)

    log('')
    log('--- CERTIFY ps_ac_rf: same vertex family and bisections as '
        'stage 5-6 ---')
    t = time.time()
    tm, peak, ok = CF2.analyse(plant0, c0, **full)
    ratio = 'inf' if np.isinf(tm) else f'{tm/tau0:.2f}'
    log(f'  peak = {peak:.3f}   tau_max/tau0 = {ratio}   '
        f'[{time.time()-t:.0f}s]')

    def plant_of(ap):
        return ControlledPlant(plate, ap=ap)

    light = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    t = time.time()
    ap_inf = CF2.depth_bisect(plant_of, mk, n_iter=16, **light)
    log(f'  a_p^inf = {ap_inf*1e3:.4f} mm   (G4 target {TARGET["g4_apinf"]*1e3:.4f},'
        f' ambition {TARGET["g4_amb"]*1e3:.4f})   [{time.time()-t:.0f}s]')
    t = time.time()
    dmax = CF2.margin_bisect(plant0, c0, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    log(f'  delta_max = {dmax:.2f}   (mu-TDC: 1.54, PS-TDC-R: 1.40)   '
        f'[{time.time()-t:.0f}s]')
    pl_lk = ControlledPlant(plate, ap=max(ap_inf, 1e-6))
    lk = CF2.lk_common_P(pl_lk, mk(pl_lk), n_pos=3,
                         etas=(0.0, C.ETA_MAX), zetas=zetas, xis=(1.0,))
    lk_txt = ('yes' if lk['feasible']
              else ('no' if lk.get('attempted', True) else 'n/a'))
    log(f'  common-P LK at a_p^inf: {lk_txt}')
    upd('stage56.pkl',
        {KIND: dict(peak=peak, tau_max=tm, tau_ratio=tm / tau0,
                    ap_inf=ap_inf, delta_max=dmax, lk=bool(lk['feasible']),
                    lk_attempted=bool(lk.get('attempted', True)))})
    log('  -> stage56.pkl')


# ---------------------------------------------------------------------------
def whole_pass(plate, ap=None):
    """G6: the exponential-LK + dwell certificate over the scheduling tube.
    `ap` (defaults to the reference depth) lets the same machinery probe the
    largest whole-pass-certified depth."""
    import sched_cert as SC
    from stage_common import load_controllers

    plant = ControlledPlant(plate, ap=C.AP_S if ap is None else ap)
    ctrl = load_controllers(plate, plant)[KIND](plant)
    v_feed = plant.feed_speed()
    log('')
    log(f'--- WHOLE-PASS {KIND}: exponential DI-LK per cell + dwell '
        f'theorem (a_p = {plant.ap*1e3:.2f} mm) ---')
    t = time.time()
    cells = SC.FamilyCells(plant, ctrl)
    ws = cells.ws
    a_star, r = 0.0, None
    for a in (2e-3, 1e-2):                     # 10 and 50 1/s
        ri = SC.certify_family(cells, a, verbose=True, log=log)
        if not ri['feasible']:
            log(f'  alpha = {a*ws:.0f} 1/s: infeasible at cell '
                f'{ri["failed_cell"]}')
            break
        a_star, r = a, ri
    if r is None:
        r0 = SC.certify_family(cells, 0.0, verbose=False, log=log)
        if r0['feasible']:
            log('  cells certify at alpha = 0 only: no decay margin, so no '
                'dwell statement; G6 NOT met')
            upd('stage56.pkl', {KIND + '_pass': dict(feasible=True,
                                                     alpha=0.0)})
        else:
            log(f'  even alpha = 0 INFEASIBLE (cell {r0["failed_cell"]}) -- '
                'G6 not met at the reference depth; reported as exactly that')
            upd('stage56.pkl', {KIND + '_pass': dict(feasible=False)})
        log(f'  [{time.time()-t:.0f}s]')
        return
    vc = r['v_cert'] * ws
    worst_lmi = max(c['worst_eig'] for c in r['certs'])
    log(f'  every cell certified at alpha = {a_star*ws:.1f} 1/s '
        f'(verified: worst LMI eigenvalue {worst_lmi:.2e})')
    log(f'  jump factors: min {r["mus"].min():.3f}  max {r["mus"].max():.3f}')
    log(f'  G6 certified traversal speed v_cert = '
        + ('unbounded' if np.isinf(vc) else f'{vc*1e3:.2f} mm/s')
        + f'  vs actual feed {v_feed*1e3:.2f} mm/s  -> '
        + ('MET' if (np.isinf(vc) or vc >= v_feed) else 'NOT MET'))
    upd('stage56.pkl',
        {KIND + '_pass': dict(feasible=True, alpha=a_star * ws,
                              mus=np.asarray(r['mus']),
                              v_cert=vc, v_feed=v_feed)})
    np.savez(os.path.join(OUT, 'psacrf_pass_cert.npz'),
             alpha=a_star * ws, mus=np.asarray(r['mus']),
             v_cert=vc, v_feed=v_feed,
             P=np.array([c['P'] for c in r['certs']]),
             Q=np.array([c['Q'] for c in r['certs']]),
             lam=np.array([c['lam'] for c in r['certs']]),
             edges=cells.edges, worst_eig=np.array([c['worst_eig']
                                                    for c in r['certs']]))
    log('  certificate matrices -> results/psacrf_pass_cert.npz '
        '(re-checkable from disk)')
    log(f'  [{time.time()-t:.0f}s]')


# ---------------------------------------------------------------------------
def scenarios(plate):
    import run_stage78 as S78
    from stage_common import load_controllers
    plant = ControlledPlant(plate)
    mk = load_controllers(plate, plant)[KIND]
    log('')
    log('--- SCENARIOS ps_ac_rf: stages 7-8, same code path as everyone ---')
    p = os.path.join(OUT, 'stage78.pkl')
    store = pickle.load(open(p, 'rb'))
    for fn in (S78.s1, S78.s2, S78.s3, S78.s4):
        fn(plate, plant, {KIND: mk}, store)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    log('  -> stage78.pkl')


# ---------------------------------------------------------------------------
def verdict():
    s56 = pickle.load(open(os.path.join(OUT, 'stage56.pkl'), 'rb'))
    s78 = pickle.load(open(os.path.join(OUT, 'stage78.pkl'), 'rb'))
    mu = np.load(os.path.join(OUT, 'psacrf_mu.npz'))
    r = s56.get(KIND, {})
    ps = s56.get(KIND + '_pass', {})
    S1 = np.asarray(s78[f'S1_{KIND}']['limits'], float)
    S3 = np.asarray(s78[f'S3_{KIND}'], float)
    S4 = np.asarray(s78[f'S4_{KIND}'], float)
    checks = [
        ('G1  sup mu_RS(point, actuator) < 1',
         f'{mu["point"].max():.3f}', mu['point'].max() < 1.0),
        ('G2  sup mu_RS(cell, actuator) < 1',
         f'{mu["cell"].max():.3f}', mu['cell'].max() < 1.0),
        ('G3  S1 floor >= 0.6821 mm (mu-TDC)',
         f'{S1.min()*1e3:.4f}', S1.min() >= TARGET['g3_floor']),
        ('G4  a_p^inf >= 0.3954 mm (PS-TDC-R)',
         f'{r.get("ap_inf", 0)*1e3:.4f}',
         r.get('ap_inf', 0) >= TARGET['g4_apinf']),
        ('G5  no scenario collapse (> 0.05)',
         f'{min(S3.min(), S4.min()):.4f}', min(S3.min(), S4.min()) > 0.05),
        ('G6  whole-pass: alpha > 0, v_cert >= feed',
         (f'a={ps.get("alpha", 0):.1f}/s v={ps.get("v_cert", 0)*1e3:.1f}mm/s'
          if ps.get('feasible') else 'infeasible'),
         bool(ps.get('feasible')) and ps.get('alpha', 0) > 0
         and (np.isinf(ps.get('v_cert', 0))
              or ps.get('v_cert', 0) >= ps.get('v_feed', np.inf))),
    ]
    log('')
    log('--- VERDICT PS-AC-RF against the pre-declared criteria ' + '-' * 16)
    for name, val, ok in checks:
        log(f'  {"PASS" if ok else "FAIL"}  {name:<40s} {val}')
    n = sum(ok for _, _, ok in checks)
    log(f'  {n}/6 criteria met'
        + ('' if n == 6 else ' -- reported as exactly that'))


# ---------------------------------------------------------------------------
STAGES = dict(design=lambda pl, pt: design(pl, pt),
              design_real=lambda pl, pt: design_real(pl, pt),
              judge21=lambda pl, pt: judge21(pl, pt),
              certify=lambda pl, pt: certify(pl),
              whole_pass=lambda pl, pt: whole_pass(pl),
              scenarios=lambda pl, pt: scenarios(pl),
              verdict=lambda pl, pt: verdict())


def main(args):
    t0 = time.time()
    which = [a for a in args if a in STAGES] or list(STAGES)
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    log('=' * 74)
    log('PS-AC-RF - THE ACTUATOR-AWARE SCHEDULED MEMBER (docs/09 sec. 5-bis)')
    log('=' * 74)
    for name in which:
        STAGES[name](plate, plant)
    log(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])
