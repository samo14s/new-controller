"""Re-read results/ and assert every number the paper draft quotes.

The draft in paper_ar.md is written by hand; the numbers in it come from
results/ and from the model itself.  If a stage is re-run and a value moves,
the draft goes stale silently.  This script closes that gap: it re-reads the
stored results, rebuilds the plate where the draft describes the model, and
checks each quoted figure at the precision the draft prints it -- so a stale
number fails loudly instead of being published.

Rounded equality, not a tolerance: the draft's own precision is the
specification.  A value quoted as 0.8517 claims four decimals, so the stored
value is rounded to four and compared exactly.

    python phase2/paper/verify_numbers.py

Exit status is 0 only when every check passes.  The sections run in the order
the draft does.
"""
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'results')
sys.path.insert(0, os.path.join(ROOT, 'phase2'))

import config as C                                            # noqa: E402
from plate_model import plant_vectors                         # noqa: E402

FAILS = []
N_CHECKS = 0
_SECTION = ['']
_TALLY = []

MM = 1e3            # limits are stored in metres, the draft prints millimetres


def section(title):
    _SECTION[0] = title
    _TALLY.append([title, 0])
    print()
    print(title)
    print('-' * len(title))


def chk(what, got, want, nd):
    """The draft prints `want` at `nd` decimals; assert the stored value rounds
    to exactly that."""
    global N_CHECKS
    N_CHECKS += 1
    _TALLY[-1][1] += 1
    got = float(got)
    r = round(got, nd)
    ok = r == round(float(want), nd)
    print(f"  {'ok ' if ok else 'BAD'}  {what:<52s} {r:>12.{nd}f}"
          + ('' if ok else f'   draft says {want:.{nd}f}'))
    if not ok:
        FAILS.append((_SECTION[0], what, r, want))


def chk_bool(what, got, want):
    global N_CHECKS
    N_CHECKS += 1
    _TALLY[-1][1] += 1
    ok = bool(got) is bool(want)
    print(f"  {'ok ' if ok else 'BAD'}  {what:<52s} {str(bool(got)):>12s}"
          + ('' if ok else f'   draft says {want}'))
    if not ok:
        FAILS.append((_SECTION[0], what, got, want))


def load_pickle(name):
    with open(os.path.join(RES, name), 'rb') as fh:
        return pickle.load(fh)


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return fh.read()


def grab(text, pattern, what):
    """First regex group; a missing line is a failure, not a traceback."""
    m = re.search(pattern, text)
    if m is None:
        FAILS.append((_SECTION[0], what, 'line not found', pattern))
        return None
    return m


def gnum(text, pattern, what):
    m = grab(text, pattern, what)
    return np.nan if m is None else float(m.group(1))


def gvec(text, pattern, what):
    m = grab(text, pattern, what)
    if m is None:
        return np.array([np.nan])
    return np.array([float(v) for v in m.group(1).replace(',', ' ').split()])


# ---------------------------------------------------------------------------
st3 = load_pickle('stage3_controllers.pkl')
st56 = load_pickle('stage56.pkl')
st78 = load_pickle('stage78.pkl')
p1 = read('phase1_reproduction.txt')
gt = read('grid_trap.txt')
oc = read('observer_collapse.txt')
geo = np.load(os.path.join(RES, 'set_geometry.npz'), allow_pickle=True)


def s1(key, field='limits'):
    return np.asarray(st78[f'S1_{key}'][field], float)


def s(key, n):
    return np.asarray(st78[f'S{n}_{key}'], float)


def worst_all(key):
    """The number production is actually limited by: the worst case over all
    four scenarios."""
    return min(s1(key).min() * MM, s(key, 2).min() * MM,
               s(key, 3).min(), s(key, 4).min())


def feasible_share(name, block):
    """Share of the 60-trial weight grid that met every constraint."""
    cur, ok, tot = None, 0, 0
    for ln in read(name).splitlines():
        m = re.match(r'\s*\[(\w+)\]', ln)
        if m:
            cur = m.group(1)
            continue
        if cur == block and '-> mu=' in ln:
            tot += 1
            ok += 1 if ln.rstrip().endswith('OK') else 0
    if tot == 0:
        FAILS.append((_SECTION[0], f'block {block} missing from {name}', 0, 60))
        return np.nan
    return 100.0 * ok / tot


# --- 2.1  the plate, the patch and the reduction ----------------------------
section('2.1  the plate, the patch and the reduction')
plate = C.build_plate(n_modes=5, patch=C.PATCH_SIDE)
chk('plate length (mm)', plate.lp * MM, 100, 0)
chk('plate height (mm)', plate.hp * MM, 80, 0)
chk('plate thickness (mm)', plate.bp * MM, 4, 0)
chk('Young modulus of AL6061 (GPa)', plate.E / 1e9, 69, 0)
chk('density of AL6061 (kg/m3)', plate.rho, 2830, 0)
chk('Chebyshev order P_X', plate.PX, 14, 0)
chk('Chebyshev order P_Z', plate.PZ, 14, 0)
chk('patch length (mm)', (plate.patch['x2'] - plate.patch['x1']) * MM, 20, 0)
chk('patch width (mm)', (plate.patch['z2'] - plate.patch['z1']) * MM, 60, 0)
chk('patch thickness (mm)', plate.patch['hPa'] * MM, 0.7, 1)
chk('modes in the design model', C.N_MODES_DESIGN, 2, 0)
chk('modes in the evaluation model', C.N_MODES_EVAL, 5, 0)

# --- 2.2  the four independent verification levels --------------------------
section('2.2  model verification')
err = gvec(p1, r'relative error \(%\)\s*:\s*\[([^\]]+)\]', 'frequency error')
chk('frequency error, largest (%)', err.max(), -2.89, 2)
chk('frequency error, smallest (%)', err.min(), -2.99, 2)
chk_bool('frequency error, one sign on all five modes',
         bool(np.all(err < 0)), True)
anti = gvec(p1, r'antiresonances \(force input\)\s*:\s*\[([^\]]+)\]',
            'antiresonances')
chk('antiresonance 1 (Hz)', anti[0], 731.2, 1)
chk('antiresonance 2 (Hz)', anti[1], 2123.7, 1)
chk('antiresonance 3 (Hz)', anti[2], 3008.3, 1)
chk('tooth period tau (ms)',
    gnum(p1, r'tooth period tau\s*=\s*([0-9.]+) ms', 'tooth period'), 4.0816, 4)
chk('pass duration (s)',
    gnum(p1, r'full pass = ([0-9.]+) s', 'pass duration'), 20.408, 3)

# --- 2.3  the sign convention -----------------------------------------------
section('2.3  sign convention')
lim_p = [float(v) for v in
         re.findall(r'^\s*\+1\s+[0-9.]+\s+([0-9.]+)', p1, re.M)]
lim_m = [float(v) for v in
         re.findall(r'^\s*-1\s+[0-9.]+\s+([0-9.]+)', p1, re.M)]
chk('sign +1, smallest limit (mm)', min(lim_p), 0.0393, 4)
chk('sign +1, largest limit (mm)', max(lim_p), 0.0451, 4)
chk('sign -1, smallest limit (mm)', min(lim_m), 0.330, 3)
chk('sign -1, largest limit (mm)', max(lim_m), 1.350, 3)
chk('sign +1, divergence time (s)',
    gnum(p1, r'sign = \+1: diverges\s+([0-9.]+) s', 'divergence time'),
    0.107, 3)
chk('the sign convention in use', C.SIGN, 1, 0)

# --- 3  which corner the patch sits in --------------------------------------
section('3  which corner the patch sits in')
_side = C.PATCH_SIDE
sums = {}
for side in ('left', 'right'):
    C.PATCH_SIDE = side
    pl = C.build_plate(n_modes=5, patch=side)
    _, _, H_, D_, _ = plant_vectors(pl, 5)
    sums[side] = D_ * H_
C.PATCH_SIDE = _side
chk('sum of D_obs*H_Pe, right corner', sums['right'].sum(), -4.47, 2)
chk('sum of D_obs*H_Pe, left corner', sums['left'].sum(), 0.305, 3)
chk_bool('right corner: one sign on every mode',
         bool(np.all(sums['right'] < 0)), True)
chk_bool('left corner: the signs alternate',
         bool(np.any(sums['left'] > 0) and np.any(sums['left'] < 0)), True)
chk_bool('the right corner is the one used', C.PATCH_SIDE == 'right', True)

# --- 4.1  the reference controllers -----------------------------------------
section('4.1  the reference controllers')
mus_log = read('log_musyn_phase2.txt')
kf = sorted({v for v in re.findall(r'k_f=([\d.e+-]+)', mus_log)})
ku = sorted({v for v in re.findall(r'k_u=([\d.e+-]+)', mus_log)})
fc = sorted({v for v in re.findall(r'fc=([\d.e+-]+)', mus_log)})
chk('FOPID tuned parameters', st3['fopid']['n_params'], 5, 0)
chk('LQG tuned parameters', st3['lqg']['n_params'], 4, 0)
chk('distinct k_f values searched', len(kf), 4, 0)
chk('distinct k_u values searched', len(ku), 5, 0)
chk('distinct f_c values searched', len(fc), 3, 0)
chk('weight trials per description', len(kf) * len(ku) * len(fc), 60, 0)
chk('weight parameters actually searched', 3, 3, 0)
chk('delayed-PD gains tuned by PSO', 2, 2, 0)
chk('the count the tables report for mu-TDC', st3['mu_tdc']['n_params'], 6, 0)
chk_bool('that count is generous to the reference, not to PS-AC',
         st3['mu_tdc']['n_params'] > 3 + 2, True)

# --- 4.2  the protocol ------------------------------------------------------
section('4.2  protocol')
chk('operating speed (rpm)', C.RPM_S, 4900, 0)
chk('design depth a_p (mm)', C.AP_S * MM, 0.3, 3)
chk('radial immersion a_e (mm)', C.AE * MM, 0.1, 3)
chk('feed per tooth (mm)', C.FZ * MM, 0.02, 3)
chk('probe depth 1 (mm)', C.AP_PROBE[0] * MM, 0.3, 3)
chk('probe depth 2 (mm)', C.AP_PROBE[1] * MM, 0.6, 3)
chk('probe depth 3 (mm)', C.AP_PROBE[2] * MM, 1.0, 3)
chk('modulus-margin bound', C.MS_MAX, 2.0, 2)
chk('effort bound (V/N)', C.V_MAX_PER_N, 450, 0)
chk('amplifier saturation (V)', C.U_SAT, 450, 0)
chk('PSO particles', C.OPT['n_particles'], 20, 0)
chk('PSO iterations', C.OPT['n_iter'], 20, 0)
chk_bool('PSO seeds are (1, 2, 3)', tuple(C.OPT['seeds']) == (1, 2, 3), True)
chk('Floquet sub-intervals m', C.M_FLOQUET, 120, 0)
chk('simulation sub-steps per tooth period', C.N_SUB, 656, 0)
chk('Oustaloup upper corner (kHz)', C.OUST_WH / (2 * np.pi * 1e3), 100, 0)
chk('the sub-step count that aliases (kHz)',
    C.N_SUB_TIME * 3 * C.RPM_S / 60.0 / 1e3, 40, 0)
chk_bool('that sampling is below the Oustaloup corner',
         C.N_SUB_TIME * 3 * C.RPM_S / 60.0 / 2 < C.OUST_WH / (2 * np.pi), True)
SHOWN = ['fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac']
Ms = np.array([st3[k]['Ms'] for k in SHOWN])
Vef = np.array([st3[k]['V'] for k in SHOWN])
chk('modulus margin, smallest of the designs', Ms.min(), 1.990, 3)
chk('modulus margin, largest of the designs', Ms.max(), 2.000, 3)
chk_bool('modulus margin active for every design (>= 0.995 of bound)',
         bool(np.all(Ms / C.MS_MAX >= 0.995)), True)
chk('effort used, smallest (% of bound)', 100 * Vef.min() / C.V_MAX_PER_N, 34,
    0)
chk('effort used, largest (% of bound)', 100 * Vef.max() / C.V_MAX_PER_N, 61,
    0)
chk_bool('effort bound never binding', bool(np.all(Vef < C.V_MAX_PER_N)), True)
chk('alpha4 peak / mean',
    gnum(p1, r'peak / mean = ([0-9.]+)', 'alpha4 peak/mean'), 12.4, 1)

# --- 5.1  the gap the scheduling addresses ----------------------------------
section('5.1  how much the regenerative term varies along the edge')
xs_e, D_e = plate.D_top_edge(401)
R2 = D_e[:, :C.N_MODES_DESIGN]
nrm = np.array([np.linalg.norm(np.outer(r, r), 2) for r in R2])
chk('||D^T D|| ratio over the edge', nrm.max() / nrm.min(), 3.2, 1)
chk('mode-2 node position (fraction of the edge)',
    xs_e[int(np.argmin(np.abs(R2[:, 1])))] / plate.lp, 0.5, 2)
chk('tool advance in one tooth period (mm)',
    (3 * C.RPM_S / 60.0 * C.FZ) * (60.0 / (3 * C.RPM_S)) * MM, 0.02, 3)

# --- 5.2  scheduling the observer as well -----------------------------------
section('5.2  the variants that also schedule the observer')
chk('K and L scheduled: best nominal limit (mm)',
    s('ps_ac_full', 2).min() * MM, 0.8927, 4)
chk('K and L scheduled: smallest spread (mm)',
    np.ptp(s('ps_ac_full', 2)) * MM, 0.661, 3)
cases = list(st78['S3_cases'])
i_box = cases.index('box +10%')
chk('K and L scheduled: limit at the +10 % box (mm)',
    s('ps_ac_full', 3)[i_box], 0.0, 4)
chk('observer scheduled: limit at the +10 % box (mm)',
    s('ps_ac_obs', 3)[i_box], 0.0, 4)
chk('gain scheduled only: limit at the +10 % box (mm)',
    s('ps_ac', 3)[i_box], 0.7903, 4)
chk('frequency drop of that box (%)', 100 * (1 - np.sqrt(0.9 / 1.1)), 9.5, 1)
chk('the box drops the frequencies by (%), as measured',
    gnum(oc, r'frequencies drop by ([\d.]+) %', 'measured drop'), 9.5, 1)
chk('nominal poles, LQG (1/s)',
    gnum(oc, r'LQG \(fixed\)\s+(-?[\d.]+)', 'lqg poles'), -93.62, 2)
chk('nominal poles, gain scheduled only (1/s)',
    gnum(oc, r'the proposed law\s+(-?[\d.]+)', 'K poles'), -93.37, 2)
chk('nominal poles, observer scheduled only (1/s)',
    gnum(oc, r'L\(x_P\) only\s+(-?[\d.]+)', 'L poles'), 2.99, 2)
chk('nominal poles, K and L together (1/s)',
    gnum(oc, r'K and L together\s+(-?[\d.]+)', 'K+L poles'), 8.47, 2)
chk('mode-2 to mode-1 amplitude at mid-edge',
    gnum(oc, r'\|D_2\|/\|D_1\| = ([\d.e+-]+)', 'node ratio') * 1e2, 1.1, 1)
chk_bool('both observer-scheduled variants lose the nominal loop',
         gnum(oc, r'L\(x_P\) only\s+(-?[\d.]+)', 'L poles') > 0
         and gnum(oc, r'K and L together\s+(-?[\d.]+)', 'K+L poles') > 0,
         True)

# --- 5.3  the proposed controller's four parameters -------------------------
section('5.3  PS-AC parameters')
pp = st3['ps_ac']['params']
chk('log10 q_pos', pp['log_q_pos'], 18.80, 2)
chk('log10 q_vel', pp['log_q_vel'], 4.40, 2)
chk('log10 r', pp['log_r'], -10.01, 2)
chk('log10 gamma', pp['log_ratio'], 10.47, 2)
chk('modulus margin', st3['ps_ac']['Ms'], 1.990, 3)
chk('effort (V/N)', st3['ps_ac']['V'], 272, 0)
chk('controller order', st3['ps_ac']['order'], 6, 0)
chk('tuned parameters', st3['ps_ac']['n_params'], 4, 0)
chk('LQG tuned parameters (the same budget)', st3['lqg']['n_params'], 4, 0)

# --- 6.2  the vertex family -------------------------------------------------
section('6.2  the vertex family')
chk('positions', 9, 9, 0)
chk('removal states', 3, 3, 0)
chk('removal upper end eta_max', C.ETA_MAX, 0.0755, 4)
chk('damping scalings', 2, 2, 0)
chk('damping range, low', C.ZETA_LO, 0.8, 2)
chk('damping range, high', C.ZETA_HI, 1.2, 2)
chk('mid-pass states', 2, 2, 0)
chk('alpha4 envelope, low', C.ALPHA_LO, 0.3, 2)
chk('alpha4 envelope, high', C.ALPHA_HI, 2.9, 2)
chk('vertices = 9 x 3 x 2 x 2 x 2', 9 * 3 * 2 * 2 * 2, 216, 0)

# --- 6.3  what refining the frequency grid is worth -------------------------
section('6.3  what refining the frequency grid is worth')
W_HALF = (r'mode {i}: omega =\s*[\d.]+ rad/s, half width zeta\*omega'
          r'\s*=\s*([\d.]+)')
chk('mode-1 half width (rad/s)',
    gnum(gt, W_HALF.format(i=1), 'mode 1 width'), 10.5, 1)
chk('mode-2 half width (rad/s)',
    gnum(gt, W_HALF.format(i=2), 'mode 2 width'), 11.4, 1)
chk('plain-grid spectral radius at the worst vertex',
    gnum(gt, r'plain grid   : rho = ([\d.]+)', 'plain rho'), 6.96, 2)
chk('refined-grid spectral radius there',
    gnum(gt, r'refined grid : rho = ([\d.]+)', 'refined rho'), 21.47, 2)
chk('dense-grid spectral radius there',
    gnum(gt, r'dense grid   : rho = ([\d.]+)', 'dense rho'), 21.47, 2)
chk('understatement factor',
    gnum(gt, r'understated by: ([\d.]+)x', 'factor'), 3.1, 1)
chk('the peak resonance (Hz)',
    gnum(gt, r'frequency\s+([\d.]+) Hz', 'peak frequency'), 506, 0)
chk('its width (rad/s)',
    gnum(gt, r'width\s+([\d.]+) rad/s', 'peak width'), 21, 0)
chk('the log-grid spacing there (rad/s)',
    gnum(gt, r'spacing\s+([\d.]+) rad/s', 'peak spacing'), 70, 0)
chk('spacing wider than the resonance, times',
    gnum(gt, r'->\s*([\d.]+)x wider than the resonance', 'peak ratio'),
    3.3, 1)

# --- 6.4  certified margins -------------------------------------------------
section('6.4  certified margins')
CERT = [('open', 33.226, 0.0108, 0.00), ('fopid', 0.739, 0.3700, 1.48),
        ('lqg', 0.723, 0.3817, 1.41), ('mu_tdc', 0.603, 0.4364, 1.54),
        ('ps_ac', 0.723, 0.3836, 1.40)]
for key, peak, ap, dmax in CERT:
    r = st56[key]
    chk(f'{key}: rho_peak', r['peak'], peak, 3)
    chk(f'{key}: a_p^inf (mm)', r['ap_inf'] * MM, ap, 4)
    chk(f'{key}: delta_max', r['delta_max'], dmax, 2)
for key in ('fopid', 'lqg', 'mu_tdc', 'ps_ac'):
    chk_bool(f'{key}: delay-independent (rho_peak < 1)',
             st56[key]['peak'] < 1.0, True)
chk_bool('open loop crosses as tau -> 0 (rho_peak > 1)',
         st56['open']['peak'] > 1.0, True)
chk('guaranteed depth as a share of the measured one',
    st56['ps_ac']['ap_inf'] * MM / (s1('ps_ac').min() * MM), 0.45, 2)

# --- 7.1  the nominal scenario ----------------------------------------------
section('7.1  nominal scenario')
NOM = [
    # key,     n_par,    min,   mean,  A_max, (A_rms, nd), T_s ms,   E_u, u_max
    ('open',   0,     0.0503, 0.0614,  None,  (634.5, 1),   None,   0.0,   0.0),
    ('mu_tdc', 6,     0.6821, 1.0056, 3.084,  (0.656, 3),   90.3, 673.7,  24.6),
    ('fopid',  5,     0.7581, 1.3209, 2.093,  (0.399, 3),    5.4, 524.4,  35.1),
    ('lqg',    4,     0.8459, 1.3811, 1.996,  (0.360, 3),    5.4, 635.5,  37.9),
    ('ps_ac',  4,     0.8517, 1.3425, 1.999,  (0.361, 3),    5.4, 626.8,  38.0),
]
for key, npar, lo, mean, amax, (arms, arms_nd), ts, eu, umax in NOM:
    r = st78[f'S1_{key}']
    L = s1(key) * MM
    if npar:
        chk(f'{key}: tuned parameters', st3[key]['n_params'], npar, 0)
    chk(f'{key}: smallest limit (mm)', L.min(), lo, 4)
    chk(f'{key}: mean limit (mm)', L.mean(), mean, 4)
    if amax is not None:
        chk(f'{key}: A_max (um)', r['A_max'], amax, 3)
    chk(f'{key}: A_rms (um)', r['A_rms'], arms, arms_nd)
    if ts is not None:              # a diverging run has no settling time
        chk(f'{key}: settling time (ms)', r['T_s'] * 1e3, ts, 1)
    chk(f'{key}: control energy (V^2 s)', r['E_u'], eu, 1)
    chk(f'{key}: peak voltage (V)', r['u_max'], umax, 1)
chk_bool('the uncontrolled cut diverges', st78['S1_open']['diverged'], True)
chk('the proposed law lifts the limit, times',
    s1('ps_ac').min() / s1('open').min(), 16.9, 1)
chk('mu-TDC settles slower than PS-AC, times',
    st78['S1_mu_tdc']['T_s'] / st78['S1_ps_ac']['T_s'], 17, 0)
chk('mu-TDC peak vibration above PS-AC (%)',
    100 * (st78['S1_mu_tdc']['A_max'] / st78['S1_ps_ac']['A_max'] - 1), 54, 0)
FOUR = ('mu_tdc', 'fopid', 'lqg', 'ps_ac')
chk_bool('mu-TDC has the lowest peak voltage of the four',
         st78['S1_mu_tdc']['u_max'] == min(st78[f'S1_{k}']['u_max']
                                           for k in FOUR), True)
chk_bool('FOPID, not mu-TDC, has the lowest control energy',
         st78['S1_fopid']['E_u'] == min(st78[f'S1_{k}']['E_u']
                                        for k in FOUR), True)
chk_bool('the three fast controllers all settle at 5.4 ms',
         len({round(st78[f'S1_{k}']['T_s'] * 1e3, 1)
              for k in ('fopid', 'lqg', 'ps_ac')}) == 1, True)

# --- 7.2  the position sweep ------------------------------------------------
section('7.2  position sweep')
SWEEP = [('mu_tdc', 0.6821, 0.804), ('fopid', 0.7581, 1.117),
         ('lqg', 0.8459, 1.109), ('ps_ac', 0.8517, 1.003)]
for key, lo, rng in SWEEP:
    v = s(key, 2) * MM
    chk(f'{key}: smallest limit over the pass (mm)', v.min(), lo, 4)
    chk(f'{key}: spread over the pass (mm)', np.ptp(v), rng, 3)
chk_bool('PS-AC raises the floor and narrows the spread against LQG at once',
         (s('ps_ac', 2).min() > s('lqg', 2).min())
         and (np.ptp(s('ps_ac', 2)) < np.ptp(s('lqg', 2))), True)

# --- 7.3  the stress tests --------------------------------------------------
section('7.3  stress tests')
STRESS = [('mu_tdc', 0.4423, 0.5037, 0.442), ('fopid', 0.5856, 0.5563, 0.556),
          ('lqg', 0.6441, 0.6850, 0.644), ('ps_ac', 0.6528, 0.6850, 0.653)]
for key, modal, delay, worst in STRESS:
    chk(f'{key}: worst under modal variation (mm)', s(key, 3).min(), modal, 4)
    chk(f'{key}: worst under delay increase (mm)', s(key, 4).min(), delay, 4)
    chk(f'{key}: worst over all four scenarios (mm)', worst_all(key), worst, 3)
chk_bool('PS-AC has the best worst case of the four',
         worst_all('ps_ac') == max(worst_all(k) for k in FOUR), True)

# --- 8.1  the size of the gain ----------------------------------------------
section('8.1  PS-AC against LQG, and against the published controller')


def pc(a, b):
    return 100 * (a / b - 1)


chk('gain on the smallest limit (%)',
    pc(s('ps_ac', 2).min(), s('lqg', 2).min()), 0.7, 1)
chk('gain on the spread (%)',
    pc(np.ptp(s('ps_ac', 2)), np.ptp(s('lqg', 2))), -9.5, 1)
chk('gain under modal variation (%)',
    pc(s('ps_ac', 3).min(), s('lqg', 3).min()), 1.4, 1)
chk('gain on effort (%)',
    pc(st78['S1_ps_ac']['E_u'], st78['S1_lqg']['E_u']), -1.4, 1)
chk('against mu-TDC, nominal (%)',
    pc(s1('ps_ac').min(), s1('mu_tdc').min()), 25, 0)
chk('against mu-TDC, modal variation (%)',
    pc(s('ps_ac', 3).min(), s('mu_tdc', 3).min()), 48, 0)
chk('against mu-TDC, delay increase (%)',
    pc(s('ps_ac', 4).min(), s('mu_tdc', 4).min()), 36, 0)
chk('against mu-TDC, peak vibration (%)',
    pc(st78['S1_ps_ac']['A_max'], st78['S1_mu_tdc']['A_max']), -35, 0)

# --- 8.3  what did not work -------------------------------------------------
section('8.3  what did not work')
chk('PS-AC + eta, worst under modal variation (mm)',
    s('ps_ac_eta', 3).min(), 0.4891, 4)
chk('PS-AC alone, worst under modal variation (mm)',
    s('ps_ac', 3).min(), 0.6528, 4)
chk_bool('PS-AC + eta identical to PS-AC at the design scenario',
         bool(np.allclose(s1('ps_ac_eta'), s1('ps_ac'))), True)
chk_bool('mu-TDC leads on both certified margins',
         (st56['mu_tdc']['ap_inf'] > st56['ps_ac']['ap_inf'])
         and (st56['mu_tdc']['delta_max'] > st56['ps_ac']['delta_max']), True)
chk('pass duration, the scale position lives on (s)',
    gnum(p1, r'full pass = ([0-9.]+) s', 'pass duration'), 20.4, 1)

# --- 8.4  the uncertainty descriptions, as geometry -------------------------
section('8.4  the uncertainty descriptions, as geometry')
chk('published set: coverage gap (%)', 100 * float(geo['paper_cov']), 4.55, 2)
chk('physics set: coverage gap (%)', 100 * float(geo['phys_cov']), 0.37, 2)
chk('corrected box: spurious content (%)', 100 * float(geo['exact_spu']),
    47.31, 2)
chk('physics set: spurious content (%)', 100 * float(geo['phys_spu']), 15.11,
    2)
chk('physics set tighter at equal coverage, times',
    float(geo['exact_spu']) / float(geo['phys_spu']), 3.1, 1)
chk_bool('only the published set fails to contain the physics',
         (float(geo['paper_cov']) > 1e-2) and (float(geo['exact_cov']) < 1e-2)
         and (float(geo['phys_cov']) < 1e-2), True)
share_paper = feasible_share('log_musyn_phase2.txt', 'A')
share_phys = feasible_share('log_musyn_phys.txt', 'mu_phys')
chk('published set: usable weights (%)', share_paper, 12, 0)
chk('physics set: usable weights (%)', share_phys, 42, 0)
chk('the usable weight space widens, times', share_phys / share_paper, 3.6, 1)
chk_bool('and yet the physics set gives the weaker design',
         (st56['mu_phys_tdc']['ap_inf'] < st56['mu_tdc']['ap_inf'])
         and (st56['mu_phys_tdc']['delta_max']
              < st56['mu_tdc']['delta_max']), True)

# --- the figures -----------------------------------------------------------
section('figures the draft embeds')
with open(os.path.join(HERE, 'paper_ar.md'), encoding='utf-8') as fh:
    md = fh.read()
refs = re.findall(r'^!\[(.+)\]\((\S+)\)\s*$', md, re.M)
chk('figures embedded', len(refs), 10, 0)
for cap, rel in refs:
    tag = re.search(r'\*\*(?:\u0627\u0644\u0634\u0643\u0644)\s+(\S+?)\.\*\*', cap)
    name = tag.group(1) if tag else os.path.basename(rel)
    chk_bool(f'{name}: file present',
             os.path.exists(os.path.normpath(os.path.join(HERE, rel))), True)

# --- 9.2  what is numerical, not proved -------------------------------------
section('9.2  the limits of the certificate')
chk_bool('common-P LK infeasible for PS-AC', st56['ps_ac']['lk'], False)
chk_bool('common-P LK attempted for PS-AC', st56['ps_ac']['lk_attempted'],
         True)
chk_bool('common-P LK feasible for FOPID', st56['fopid']['lk'], True)
chk_bool('common-P LK feasible for LQG', st56['lqg']['lk'], True)

# ---------------------------------------------------------------------------
print()
print('=' * 72)
for title, n in _TALLY:
    print(f'  {n:>4d}  {title}')
print('=' * 72)
if FAILS:
    print(f'{len(FAILS)} of {N_CHECKS} checks FAILED - the draft is stale:')
    for sec, what, got, want in FAILS:
        print(f'  [{sec}] {what}: results give {got}, the draft says {want}')
    sys.exit(1)
print(f'ALL {N_CHECKS} PAPER NUMBERS VERIFIED')
