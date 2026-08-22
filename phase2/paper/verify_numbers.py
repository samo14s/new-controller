"""Re-read results/ and assert every number the paper draft quotes.

The draft in paper_ar.md is written by hand; the numbers in it come from
results/.  If a stage is re-run and a value moves, the draft goes stale
silently.  This script closes that gap: it re-reads the stored results and
checks each quoted figure at the precision the draft prints it, so a stale
number fails loudly instead of being published.

    python phase2/paper/verify_numbers.py

Exit status is 0 only when every check passes.
"""
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'results')

FAILS = []
N_CHECKS = 0
_SECTION = ['']


def section(title):
    _SECTION[0] = title
    print()
    print(title)
    print('-' * len(title))


def chk(what, got, want, nd):
    """The draft prints `want` at `nd` decimals; assert the stored value rounds
    to exactly that.  Rounded equality rather than a tolerance, because the
    draft's own precision is the specification."""
    global N_CHECKS
    N_CHECKS += 1
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
    ok = bool(got) is bool(want)
    print(f"  {'ok ' if ok else 'BAD'}  {what:<52s} {str(bool(got)):>12s}"
          + ('' if ok else f'   draft says {want}'))
    if not ok:
        FAILS.append((_SECTION[0], what, got, want))


def load_pickle(name):
    with open(os.path.join(RES, name), 'rb') as fh:
        return pickle.load(fh)


def phase1_text():
    with open(os.path.join(RES, 'phase1_reproduction.txt')) as fh:
        return fh.read()


def grab(text, pattern, what):
    """First regex group as a float; a missing line is a failure, not a crash
    with a confusing traceback."""
    m = re.search(pattern, text)
    if m is None:
        FAILS.append((_SECTION[0], what, 'line not found in '
                      'phase1_reproduction.txt', pattern))
        return np.nan
    return m


def grab_vec(text, pattern, what):
    m = grab(text, pattern, what)
    if m is np.nan:
        return np.array([np.nan])
    return np.array([float(v) for v in m.group(1).replace(',', ' ').split()])


# ---------------------------------------------------------------------------
st3 = load_pickle('stage3_controllers.pkl')
st56 = load_pickle('stage56.pkl')
st78 = load_pickle('stage78.pkl')
p1 = phase1_text()

MM = 1e3            # limits are stored in metres, the draft prints millimetres
V_PER_N = 450.0     # the effort bound of the protocol
MS_MAX = 2.0


def s1(key, field='limits'):
    return np.asarray(st78[f'S1_{key}'][field], float)


def s(key, n):
    return np.asarray(st78[f'S{n}_{key}'], float)


def worst_all(key):
    """The number production is actually limited by: worst case over all four
    scenarios."""
    return min(s1(key).min() * MM, s(key, 2).min() * MM,
               s(key, 3).min(), s(key, 4).min())


# --- 2.2  the four independent verification levels --------------------------
section('2.2  model verification')
err = grab_vec(p1, r'relative error \(%\)\s*:\s*\[([^\]]+)\]', 'frequency error')
chk('frequency error, largest (%)', err.max(), -2.89, 2)
chk('frequency error, smallest (%)', err.min(), -2.99, 2)
chk_bool('frequency error, one sign on all five modes',
         bool(np.all(err < 0)), True)
anti = grab_vec(p1, r'antiresonances \(force input\)\s*:\s*\[([^\]]+)\]',
                'antiresonances')
chk('antiresonance 1 (Hz)', anti[0], 731.2, 1)
chk('antiresonance 2 (Hz)', anti[1], 2123.7, 1)
chk('antiresonance 3 (Hz)', anti[2], 3008.3, 1)
m = grab(p1, r'tooth period tau\s*=\s*([0-9.]+) ms', 'tooth period')
chk('tooth period tau (ms)', m.group(1) if m is not np.nan else np.nan,
    4.0816, 4)
m = grab(p1, r'full pass = ([0-9.]+) s', 'pass duration')
chk('pass duration (s)', m.group(1) if m is not np.nan else np.nan, 20.408, 3)

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
m = grab(p1, r'sign = \+1: diverges\s+([0-9.]+) s', 'divergence time')
chk('sign +1, divergence time (s)', m.group(1) if m is not np.nan else np.nan,
    0.107, 3)

# --- 4.2  the protocol ------------------------------------------------------
section('4.2  protocol')
SHOWN = ['fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac']
Ms = np.array([st3[k]['Ms'] for k in SHOWN])
Vef = np.array([st3[k]['V'] for k in SHOWN])
chk('modulus margin, smallest of the designs', Ms.min(), 1.990, 3)
chk('modulus margin, largest of the designs', Ms.max(), 2.000, 3)
chk_bool('modulus margin active for every design (>= 0.995 of bound)',
         bool(np.all(Ms / MS_MAX >= 0.995)), True)
chk('effort used, smallest (% of bound)', 100 * Vef.min() / V_PER_N, 34, 0)
chk('effort used, largest (% of bound)', 100 * Vef.max() / V_PER_N, 61, 0)
chk_bool('effort bound never binding', bool(np.all(Vef < V_PER_N)), True)
m = grab(p1, r'peak / mean = ([0-9.]+)', 'alpha4 peak/mean')
chk('alpha4 peak / mean', m.group(1) if m is not np.nan else np.nan, 12.4, 1)

# --- 5.2  scheduling the observer as well -----------------------------------
section('5.2  the variant that also schedules the observer')
chk('K and L scheduled: best nominal limit (mm)', s(  # noqa: E501
    'ps_ac_full', 2).min() * MM, 0.8927, 4)
chk('K and L scheduled: smallest spread (mm)',
    (s('ps_ac_full', 2).max() - s('ps_ac_full', 2).min()) * MM, 0.661, 3)
cases = list(st78['S3_cases'])
i_box = cases.index('box +10%')
chk('K and L scheduled: limit at the +10 % box (mm)',
    s('ps_ac_full', 3)[i_box], 0.0, 4)
chk('observer scheduled: limit at the +10 % box (mm)',
    s('ps_ac_obs', 3)[i_box], 0.0, 4)
chk('gain scheduled only: limit at the +10 % box (mm)',
    s('ps_ac', 3)[i_box], 0.7903, 4)
chk('frequency drop of that box (%)',
    100 * (1 - np.sqrt(0.9 / 1.1)), 9.5, 1)

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
chk('vertices = 9 positions x 3 eta x 2 zeta x 2 xi x 2 alpha4',
    9 * 3 * 2 * 2 * 2, 216, 0)

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

# --- 7.1  the nominal scenario ----------------------------------------------
section('7.1  nominal scenario')
NOM = [
    # key,     n_par,    min,   mean,  A_max, (A_rms, nd), T_s ms,   E_u, u_max
    ('open',   0,     0.0503, 0.0614,  None,  (634.5, 1),  174.2,   0.0,   0.0),
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
    chk(f'{key}: settling time (ms)', r['T_s'] * 1e3, ts, 1)
    chk(f'{key}: control energy (V^2 s)', r['E_u'], eu, 1)
    chk(f'{key}: peak voltage (V)', r['u_max'], umax, 1)
chk_bool('the uncontrolled cut diverges', st78['S1_open']['diverged'], True)
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

# --- 7.2  the position sweep ------------------------------------------------
section('7.2  position sweep')
SWEEP = [('mu_tdc', 0.6821, 0.804), ('fopid', 0.7581, 1.117),
         ('lqg', 0.8459, 1.109), ('ps_ac', 0.8517, 1.003)]
for key, lo, rng in SWEEP:
    v = s(key, 2) * MM
    chk(f'{key}: smallest limit over the pass (mm)', v.min(), lo, 4)
    chk(f'{key}: spread over the pass (mm)', v.max() - v.min(), rng, 3)
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
         worst_all('ps_ac') == max(worst_all(k) for k in
                                   ('mu_tdc', 'fopid', 'lqg', 'ps_ac')), True)

# --- 8.1  the size of the gain ----------------------------------------------
section('8.1  PS-AC against LQG, and against the published controller')
pc = lambda a, b: 100 * (a / b - 1)                                 # noqa: E731
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
if FAILS:
    print(f'{len(FAILS)} of {N_CHECKS} checks FAILED - the draft is stale:')
    for sec, what, got, want in FAILS:
        print(f'  [{sec}] {what}: results give {got}, the draft says {want}')
    sys.exit(1)
print(f'ALL {N_CHECKS} PAPER NUMBERS VERIFIED')
