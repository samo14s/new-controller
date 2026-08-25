"""
g0_apinf_sens.py — G0b: what sets a_p^inf for the scheduled family, and which
                   knob moves it?
==============================================================================
D1 (a_p^inf >= 0.4364 mm, mu-TDC's value) is the one axis no scheduled design
has ever won, and results/margin_landscape.txt already proved the delayed pair
alone cannot deliver it on the PS-AC base (627 points, 89 feasible, 0 reaching
the target).  This script measures, BEFORE anyone tunes anything, two things:

PART 1 — WHERE THE CONSTRAINT BINDS.
    a_p^inf is the largest depth at which the delay-independence test
    sup_omega rho((j omega I - A_cl)^-1 A_d,cl) < 1 holds at EVERY vertex of
    the family the Stage 5-6 bisection uses:

        light = n_pos 5, etas (0, ETA_MAX), zetas (0.8, 1.2), xis (1.0,),
                alpha_4 ends (0.3, 2.9) * abar4        -> 40 vertices

    At each design's own a_p^inf the script enumerates those 40 vertices in the
    exact order certify2.vertices produces them, records the peak and the
    frequency at which the peak sits (the grid lives in scaled time; Hz =
    w_scaled * ws / 2pi with ws = 2 pi 800), and names the ARGMAX vertex:
    position, alpha_4 end, eta, zeta, xi.  Designs probed: mu_tdc, ps_ac,
    ps_ac_obs, ps_ac_full, ps_tdc_r.
    results/envelope_probe.txt found, at the TRUE alpha_4 peak (12.43 abar4),
    that the LQG family binds near 505 Hz and mu-TDC near 322 Hz.  Here the
    same question is asked at the DESIGN alpha_4 band, and answered either way.

PART 2 — WHAT MOVES IT.
    One-knob-at-a-time ladders around the STORED ps_ac point (a_p^inf =
    0.3836 mm), 7 points spanning each knob's full BOUNDS2 interval plus the
    base value: log_q_pos, log_q_vel, log_r, log_ratio, and a4_mult (the
    fifth parameter PS-AC-R exposes; on the a4_mult ladder the structure is
    therefore PS-AC-R, stated).  At every ladder point: a_p^inf by the same
    depth_bisect(n_iter=16) on the same light family, delta_max by the same
    margin_bisect(n_iter=14, n_pos=5, xis=(1,)), and the protocol screen
    (nominal poles <= -1 /s over POSITIONS_DESIGN, Ms <= 2, V <= 450 V/N)
    evaluated by eval2, i.e. the identical code the design stage used.

PRE-DECLARED SUCCESS CRITERIA (written before the run):

  S1  The binding-vertex identification counts as MEANINGFUL only if a second,
      independent pass at the SAME depth — fresh plate, fresh plant, fresh
      controller objects — returns the SAME vertex index and a peak agreeing to
      1e-10 relative.  PASS/FAIL is reported per design, whatever it is.  A
      FAIL means the identification is numerical noise and must not be used.

  S2  A knob "can plausibly deliver D1" only if some point on its measured
      ladder reaches a_p^inf >= 0.4364 mm WHILE the protocol constraints hold
      at that point (max Re <= -1 /s, Ms <= 2.0, V <= 450 V/N).  The
      constraints are measured at every ladder point, not assumed.  A ladder
      point that reaches the depth with a violated constraint is reported as
      REACHED-BUT-INFEASIBLE and does NOT count.

  S3  Knobs whose ladder moves a_p^inf the WRONG way (below the base 0.3836 mm)
      are reported in the same table and with the same prominence as any knob
      that moves it up; the signed one-sided slopes at the base are printed for
      every knob.

WHAT THIS SCRIPT DOES NOT DO (declared up front):
  * No two-knob interactions: the ladders are strictly one-at-a-time, so a
    combination that beats every single knob would not be seen here.
  * No re-optimisation, no PSO: this is a measurement, not a design.
  * The bisections use the same REDUCED (light) family as Stage 5-6, so the
    depths are comparable to the stored scoreboard and carry the same
    upper-bound caveat that run_stage56 states.
  * delta_max is measured only on the ladders, not for Part 1.

Writes results/g0_apinf_sens.txt and results/g0_apinf_sens.npz.  Nothing else.
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import certify2 as CF2
import ctrl2 as K2
import eval2 as E2
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers

OUT = C.RESULTS
TARGET = 0.4364e-3                      # D1: mu-TDC's a_p^inf
BASE_AP = 0.3836e-3                     # stored ps_ac a_p^inf, for reference

LIGHT = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=(C.ZETA_LO, C.ZETA_HI),
             xis=(1.0,))
MARGIN_KW = dict(n_pos=5, xis=(1.0,))
KINDS1 = ('mu_tdc', 'ps_ac', 'ps_ac_obs', 'ps_ac_full', 'ps_tdc_r')

PS_AC_BASE = dict(log_q_pos=18.80271200842692, log_q_vel=4.404518846606756,
                  log_r=-10.013785054801193, log_ratio=10.470924868454862,
                  a4_mult=1.0)
KNOB_BOUNDS = dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                   log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0),
                   a4_mult=(0.4, 2.0))
KNOBS = ('log_q_pos', 'log_q_vel', 'log_r', 'log_ratio', 'a4_mult')


# ---------------------------------------------------------------------------
def vertex_labels():
    """(x_frac, eta, xi, zeta, alpha_end) in certify2.vertices' emission order."""
    lab = []
    for ix in range(LIGHT['n_pos']):
        for eta in LIGHT['etas']:
            for xi in LIGHT['xis']:
                for z in LIGHT['zetas']:
                    for a in ('lo', 'hi'):
                        lab.append((ix, eta, xi, z, a))
    return lab


def scan_vertices(plant, ctrl):
    """Per-vertex (peak, peak frequency in Hz) over the light family."""
    V, npl, ws = CF2.vertices(plant, ctrl, **LIGHT)
    peaks = np.zeros(len(V))
    fpk = np.full(len(V), np.nan)
    for i, (A, Ad) in enumerate(V):
        w = CF2._wgrid(A)
        I = np.eye(A.shape[0])
        pk, wat = 0.0, np.nan
        for wk in w:
            try:
                r = float(np.max(np.abs(np.linalg.eigvals(
                    np.linalg.solve(1j * wk * I - A, Ad)))))
            except np.linalg.LinAlgError:
                continue
            if r > pk:
                pk, wat = r, wk
        peaks[i] = pk
        fpk[i] = wat * ws / (2 * np.pi)
    return peaks, fpk, ws


def part1_one(kind):
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    p0 = ControlledPlant(plate, ap=C.AP_S)
    mk = load_controllers(plate, p0)[kind]
    ap = CF2.depth_bisect(lambda a: ControlledPlant(plate, ap=a), mk,
                          n_iter=16, **LIGHT)
    p = ControlledPlant(plate, ap=max(ap, 1e-6))
    peaks, fpk, ws = scan_vertices(p, mk(p))
    # S1: second, independent pass at the SAME depth
    plate_b = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    p0b = ControlledPlant(plate_b, ap=C.AP_S)
    mkb = load_controllers(plate_b, p0b)[kind]
    pb = ControlledPlant(plate_b, ap=max(ap, 1e-6))
    peaks_b, fpk_b, _ = scan_vertices(pb, mkb(pb))
    return dict(kind=kind, ap=ap, peaks=peaks, fpk=fpk,
                peaks_b=peaks_b, fpk_b=fpk_b, lp=plate.lp,
                order=mk(p).order)


# ---------------------------------------------------------------------------
def make_ps_ac(u):
    def mk(plant):
        c = K2.ps_ac(plant, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                     10 ** u['log_r'], 10 ** u['log_ratio'],
                     sched_K=True, sched_L=False,
                     a4_mult=float(u['a4_mult']), name='PS_AC(knob)')
        return c
    return mk


def screen(plate, ctrl):
    """The protocol screen, by the same code eval2 uses at design time."""
    mre = -np.inf
    Ms, V = -np.inf, -np.inf
    for fr in C.POSITIONS_DESIGN:
        ss, pd = ctrl.at(fr * plate.lp)
        mre = max(mre, float(np.max(E2.nominal_poles(plate, ss, pd).real)))
        a, b = E2.frequency_metrics(plate, ss, pd, C.POSITIONS_DESIGN)
        Ms, V = max(Ms, a), max(V, b)
    return mre, Ms, V


def ladder_point(job):
    knob, val = job
    u = dict(PS_AC_BASE)
    u[knob] = val
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    p0 = ControlledPlant(plate, ap=C.AP_S)
    mk = make_ps_ac(u)
    try:
        mre, Ms, V = screen(plate, mk(p0))
        ap = CF2.depth_bisect(lambda a: ControlledPlant(plate, ap=a), mk,
                              n_iter=16, **LIGHT)
        dmax = CF2.margin_bisect(p0, mk(p0), n_iter=14, base_kw=dict(MARGIN_KW))
    except Exception as exc:                                  # noqa: BLE001
        return dict(knob=knob, val=val, ap=np.nan, dmax=np.nan, mre=np.nan,
                    Ms=np.nan, V=np.nan, err=f'{type(exc).__name__}: {exc}')
    return dict(knob=knob, val=val, ap=ap, dmax=dmax, mre=mre, Ms=Ms, V=V,
                err='')


def ladders():
    jobs = []
    for k in KNOBS:
        lo, hi = KNOB_BOUNDS[k]
        vals = sorted(set(np.round(
            np.concatenate([np.linspace(lo, hi, 7), [PS_AC_BASE[k]]]), 10)))
        for v in vals:
            jobs.append((k, float(v)))
    return jobs


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    out = ['=' * 84,
           'G0b -- WHAT SETS a_p^inf FOR THE SCHEDULED FAMILY, AND WHICH KNOB '
           'MOVES IT?',
           '=' * 84,
           f'  target D1            : {TARGET*1e3:.4f} mm (mu-TDC)',
           f'  family (both parts)  : n_pos {LIGHT["n_pos"]}, '
           f'etas {tuple(round(float(e),4) for e in LIGHT["etas"])}, '
           f'zetas {LIGHT["zetas"]}, xis {LIGHT["xis"]}, '
           f'alpha_4 ends {{{C.ALPHA_LO}, {C.ALPHA_HI}}} abar4 '
           f'-> {2*LIGHT["n_pos"]*len(LIGHT["etas"])*len(LIGHT["zetas"])} '
           'vertices',
           '  bisections           : depth n_iter 16, margin n_iter 14 -- the '
           'Stage 5-6 settings',
           '']

    # ---------------------------------------------------------------- part 1
    out += ['-' * 84,
            'PART 1 -- WHERE THE CONSTRAINT BINDS, at each design\'s own '
            'a_p^inf',
            '-' * 84]
    with Pool(min(4, len(KINDS1))) as pool:
        r1 = pool.map(part1_one, KINDS1)
    lp = r1[0]['lp']
    labs = vertex_labels()
    xs = np.linspace(0.0, lp, LIGHT['n_pos'])
    modes = np.asarray(C.F_MEASURED, float)

    out.append(f'  plate modes (Hz): '
               + ' '.join(f'{f:.1f}' for f in modes))
    out.append('')
    out.append(f'{"design":<12}{"a_p^inf mm":>11}{"peak":>8}{"f_peak Hz":>11}'
               f'{"x mm":>8}{"alpha4":>8}{"eta":>8}{"zeta":>7}{"xi":>5}'
               f'{"ord":>5}')
    store1 = {}
    for r in r1:
        i = int(np.argmax(r['peaks']))
        ix, eta, xi, z, a = labs[i]
        out.append(f'{r["kind"]:<12}{r["ap"]*1e3:>11.4f}{r["peaks"][i]:>8.4f}'
                   f'{r["fpk"][i]:>11.1f}{xs[ix]*1e3:>8.1f}'
                   f'{(C.ALPHA_LO if a=="lo" else C.ALPHA_HI):>8.1f}'
                   f'{eta:>8.4f}{z:>7.2f}{xi:>5.1f}{r["order"]:>5d}')
        store1[r['kind']] = r
    out.append('  alpha4 column is the vertex end in units of abar4 '
               '(design band 0.3 .. 2.9)')
    out.append('')

    out.append('  the five worst vertices per design (peak, f Hz, x mm, '
               'alpha4, eta, zeta)')
    for r in r1:
        out.append(f'  {r["kind"]}:')
        for i in np.argsort(-r['peaks'])[:5]:
            ix, eta, xi, z, a = labs[int(i)]
            out.append(f'      {r["peaks"][i]:.4f}  {r["fpk"][i]:9.1f}  '
                       f'{xs[ix]*1e3:6.1f}  '
                       f'{(C.ALPHA_LO if a=="lo" else C.ALPHA_HI):4.1f}  '
                       f'{eta:.4f}  {z:.2f}')
    out.append('')

    # S1
    out.append('  [S1] same depth, second independent pass: same vertex, and '
               'peak to 1e-10 relative?')
    s1_all = True
    for r in r1:
        i = int(np.argmax(r['peaks']))
        j = int(np.argmax(r['peaks_b']))
        rel = abs(r['peaks'][i] - r['peaks_b'][j]) / max(r['peaks'][i], 1e-30)
        ok = (i == j) and rel <= 1e-10
        s1_all &= ok
        out.append(f'      {r["kind"]:<12} vertex {i:>3} vs {j:>3}   '
                   f'rel peak diff {rel:.2e}   '
                   f'f {r["fpk"][i]:.1f} vs {r["fpk"][j]:.1f} Hz   '
                   f'{"PASS" if ok else "FAIL"}')
    out.append(f'      [S1] {"PASS" if s1_all else "FAIL"} -- the binding-vertex'
               f' identification is '
               f'{"reproducible" if s1_all else "NOT reproducible; do not use it"}')
    out.append('')
    out.append('  margin of the binding vertex over the runner-up (how sharply '
               'the argmax is defined)')
    for r in r1:
        s = np.sort(r['peaks'])[::-1]
        out.append(f'      {r["kind"]:<12} top {s[0]:.4f}  2nd {s[1]:.4f}  '
                   f'gap {100*(s[0]-s[1])/s[0]:.3f} %   '
                   f'spread over the 40 vertices '
                   f'{s[0]-s[-1]:.4f}')
    out.append('')
    out.append('  comparison with results/envelope_probe.txt, which asked the '
               'same question at the')
    out.append('  TRUE alpha_4 peak (12.43 abar4) and found mu-TDC binding at '
               '321.6 Hz and the LQG')
    out.append('  family at ~505 Hz:')
    for r in r1:
        i = int(np.argmax(r['peaks']))
        out.append(f'      {r["kind"]:<12} design-band f_peak '
                   f'{r["fpk"][i]:8.1f} Hz   nearest plate mode '
                   f'{modes[int(np.argmin(np.abs(modes - r["fpk"][i])))]:.1f} Hz'
                   f'  ({100*(r["fpk"][i]/modes[int(np.argmin(np.abs(modes - r["fpk"][i])))]-1):+.1f} %)')
    out.append('')

    # ---------------------------------------------------------------- part 2
    out += ['-' * 84,
            'PART 2 -- WHAT MOVES IT: one-knob ladders around the stored ps_ac '
            'point',
            '-' * 84,
            '  base: ' + ', '.join(f'{k}={PS_AC_BASE[k]:.4f}' for k in KNOBS),
            '  the a4_mult ladder is the PS-AC-R structure (5 parameters); the '
            'other four are PS-AC (4).',
            '']
    jobs = ladders()
    with Pool(4) as pool:
        r2 = pool.map(ladder_point, jobs)

    by = {k: [r for r in r2 if r['knob'] == k] for k in KNOBS}
    for k in KNOBS:
        rows = sorted(by[k], key=lambda r: r['val'])
        out.append(f'  {k}   (bounds {KNOB_BOUNDS[k]}, base '
                   f'{PS_AC_BASE[k]:.4f})')
        out.append(f'      {"value":>10}{"a_p^inf mm":>12}{"delta_max":>11}'
                   f'{"max Re":>10}{"Ms":>8}{"V/N":>9}  feasible')
        for r in rows:
            feas = (np.isfinite(r['mre']) and r['mre'] <= -1.0
                    and r['Ms'] <= C.MS_MAX and r['V'] <= C.V_PER_N)
            tag = 'yes' if feas else 'NO'
            mark = ' <- base' if abs(r['val'] - PS_AC_BASE[k]) < 1e-9 else ''
            if r['err']:
                out.append(f'      {r["val"]:>10.4f}   {r["err"]}')
                continue
            out.append(f'      {r["val"]:>10.4f}{r["ap"]*1e3:>12.4f}'
                       f'{r["dmax"]:>11.3f}{r["mre"]:>10.2f}{r["Ms"]:>8.3f}'
                       f'{r["V"]:>9.1f}  {tag}{mark}')
        aps = np.array([r['ap'] for r in rows])
        fe = np.array([(np.isfinite(r['mre']) and r['mre'] <= -1.0
                        and r['Ms'] <= C.MS_MAX and r['V'] <= C.V_PER_N)
                       for r in rows])
        span = (np.nanmin(aps), np.nanmax(aps))
        out.append(f'      span of a_p^inf along this knob alone: '
                   f'{span[0]*1e3:.4f} .. {span[1]*1e3:.4f} mm '
                   f'(all points)')
        if fe.any():
            out.append(f'      span among FEASIBLE points only          : '
                       f'{np.nanmin(aps[fe])*1e3:.4f} .. '
                       f'{np.nanmax(aps[fe])*1e3:.4f} mm '
                       f'({int(fe.sum())} of {len(fe)} feasible)')
        else:
            out.append('      no feasible point on this ladder')
        # signed one-sided slopes at the base
        vals = np.array([r['val'] for r in rows])
        ib = int(np.argmin(np.abs(vals - PS_AC_BASE[k])))
        sl = []
        if ib > 0:
            sl.append(f'down {1e3*(aps[ib]-aps[ib-1])/(vals[ib]-vals[ib-1]):+.5f} mm/unit')
        if ib < len(vals) - 1:
            sl.append(f'up {1e3*(aps[ib+1]-aps[ib])/(vals[ib+1]-vals[ib]):+.5f} mm/unit')
        out.append('      signed slope at the base: ' + '; '.join(sl))
        # S2 verdict
        hit = [r for r in rows if np.isfinite(r['ap']) and r['ap'] >= TARGET]
        hit_f = [r for r, f in zip(rows, fe)
                 if np.isfinite(r['ap']) and r['ap'] >= TARGET and f]
        if hit_f:
            out.append(f'      [S2] REACHES D1 FEASIBLY at '
                       + ', '.join(f'{r["val"]:.4f}' for r in hit_f))
        elif hit:
            out.append('      [S2] REACHED-BUT-INFEASIBLE at '
                       + ', '.join(f'{r["val"]:.4f}' for r in hit)
                       + ' -- does NOT count')
        else:
            out.append(f'      [S2] does NOT reach {TARGET*1e3:.4f} mm '
                       'anywhere on this ladder')
        # S3 wrong-way flag
        below = int(np.sum(aps < BASE_AP))
        out.append(f'      [S3] points BELOW the base a_p^inf '
                   f'({BASE_AP*1e3:.4f} mm): {below} of {len(rows)}'
                   + ('  -- this knob mostly moves a_p^inf the WRONG way'
                      if below > len(rows) / 2 else ''))
        out.append('')

    # global verdicts
    all_ap = np.array([r['ap'] for r in r2], float)
    all_f = np.array([(np.isfinite(r['mre']) and r['mre'] <= -1.0
                       and r['Ms'] <= C.MS_MAX and r['V'] <= C.V_PER_N)
                      for r in r2])
    out.append('-' * 84)
    out.append('VERDICT')
    out.append('-' * 84)
    out.append(f'  ladder points evaluated : {len(r2)}   '
               f'feasible: {int(all_f.sum())}')
    out.append(f'  best a_p^inf over ALL ladder points      : '
               f'{np.nanmax(all_ap)*1e3:.4f} mm')
    if all_f.any():
        out.append(f'  best a_p^inf over FEASIBLE ladder points : '
                   f'{np.nanmax(all_ap[all_f])*1e3:.4f} mm '
                   f'({100*(np.nanmax(all_ap[all_f])/TARGET-1):+.2f} % of D1)')
    reach = [k for k in KNOBS
             if any(np.isfinite(r['ap']) and r['ap'] >= TARGET
                    and np.isfinite(r['mre']) and r['mre'] <= -1.0
                    and r['Ms'] <= C.MS_MAX and r['V'] <= C.V_PER_N
                    for r in by[k])]
    out.append('  [S2] knobs that reach D1 feasibly by themselves: '
               + (', '.join(reach) if reach else 'NONE'))
    out.append(f'  total {time.time()-t0:.0f}s')

    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'g0_apinf_sens.txt'), 'w') as f:
        f.write(txt)
    print(txt)

    np.savez(os.path.join(OUT, 'g0_apinf_sens.npz'),
             p1_kinds=np.array(KINDS1),
             p1_ap=np.array([store1[k]['ap'] for k in KINDS1]),
             p1_peaks=np.array([store1[k]['peaks'] for k in KINDS1]),
             p1_fpk=np.array([store1[k]['fpk'] for k in KINDS1]),
             p1_peaks_b=np.array([store1[k]['peaks_b'] for k in KINDS1]),
             p1_fpk_b=np.array([store1[k]['fpk_b'] for k in KINDS1]),
             p1_order=np.array([store1[k]['order'] for k in KINDS1]),
             vertex_x_mm=np.array([xs[l[0]] * 1e3 for l in labs]),
             vertex_eta=np.array([l[1] for l in labs]),
             vertex_xi=np.array([l[2] for l in labs]),
             vertex_zeta=np.array([l[3] for l in labs]),
             vertex_alpha=np.array([C.ALPHA_LO if l[4] == 'lo' else C.ALPHA_HI
                                    for l in labs]),
             modes_hz=modes,
             p2_knob=np.array([r['knob'] for r in r2]),
             p2_val=np.array([r['val'] for r in r2]),
             p2_ap=all_ap,
             p2_dmax=np.array([r['dmax'] for r in r2], float),
             p2_mre=np.array([r['mre'] for r in r2], float),
             p2_Ms=np.array([r['Ms'] for r in r2], float),
             p2_V=np.array([r['V'] for r in r2], float),
             p2_feasible=all_f,
             base_params=np.array([PS_AC_BASE[k] for k in KNOBS]),
             base_knobs=np.array(KNOBS),
             target=TARGET, base_ap=BASE_AP)


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


# ---------------------------------------------------------------------------
def addendum():
    """POST-HOC block, appended after the measurement run.

    Everything here is DERIVED from results/g0_apinf_sens.npz -- no new
    physics is computed, no threshold is moved.  It is reported separately
    and labelled because it was NOT pre-declared: the rank agreement between
    the binding frequency and a_p^inf was noticed in the Part 1 table after
    the fact, and the bisection-resolution caveat is a property of the
    protocol that the table makes visible.

        python g0_apinf_sens.py addendum
    """
    d = np.load(os.path.join(OUT, 'g0_apinf_sens.npz'), allow_pickle=True)
    kinds = list(d['p1_kinds'])
    ap = d['p1_ap']
    pk = d['p1_peaks']
    fp = d['p1_fpk']
    ib = [int(np.argmax(p)) for p in pk]
    f_bind = np.array([fp[i][j] for i, j in enumerate(ib)])

    # bisection resolution actually delivered
    quantum = 2e-6      # certify2.depth_bisect tol, in metres
    out = ['', '-' * 84,
           'ADDENDUM -- derived from the npz AFTER the run, NOT pre-declared',
           '-' * 84,
           f'  [A1] depth resolution.  certify2.depth_bisect stops at '
           f'tol = {quantum*1e3:.4f} mm, so any',
           f'       difference below {quantum*1e3:.4f} mm in the Part 2 tables '
           'is NOT resolved by the',
           '       instrument.  The four knobs whose feasible span is '
           '0.3817 .. 0.3836 mm move',
           f'       a_p^inf by exactly one quantum, i.e. by nothing the '
           'bisection can see.  The',
           f'       gap the campaign must close, 0.3836 -> 0.4364 mm, is '
           f'{(TARGET-BASE_AP)/quantum:.0f} quanta wide, so it is',
           '       far above the resolution floor and the negative result is '
           'not a numerical artefact.',
           '']
    order = np.argsort(ap)
    out.append('  [A2] binding frequency vs a_p^inf, ranked (5 designs, '
               'post-hoc observation)')
    out.append(f'       {"design":<12}{"a_p^inf mm":>12}{"f_bind Hz":>12}'
               f'{"x mm":>8}')
    xs_all = d['vertex_x_mm']
    for i in order:
        out.append(f'       {kinds[i]:<12}{ap[i]*1e3:>12.4f}{f_bind[i]:>12.1f}'
                   f'{xs_all[ib[i]]:>8.1f}')
    ra = np.argsort(np.argsort(ap))
    rf = np.argsort(np.argsort(-f_bind))
    rho = float(np.corrcoef(ra, rf)[0, 1])
    out.append(f'       Spearman rank correlation between a_p^inf and '
               f'(-f_bind) over these 5 designs: {rho:+.3f}')
    out.append('       n = 5.  This is an OBSERVED ordering, not a mechanism '
               'and not a prediction;')
    out.append('       it says the designs that certify deeper are the ones '
               'whose crossing peak sits')
    out.append('       lower in frequency, and nothing about causation.')
    out.append('')
    out.append('  [A3] what is common to all five binding vertices: '
               'alpha_4 = 2.9 abar4 (the UPPER')
    out.append('       end of the design band) and zeta = 0.80 (the LOW '
               'damping end), 5 of 5.  The')
    out.append('       binding position is x = 0 mm for all four LQG-family '
               'designs and x = 100 mm')
    out.append('       (the far edge) for mu-TDC; eta is not shared.')
    txt = '\n'.join(out) + '\n'
    with open(os.path.join(OUT, 'g0_apinf_sens.txt'), 'a') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'addendum':
    addendum()
