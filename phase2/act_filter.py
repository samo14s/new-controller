"""act_filter.py — the actuator-aware scheduled member: a D-K-free road to G1.

The anatomy of the G1 blocker (docs/09, measured here in this container):

    PS-AC-R at mid-span   mu_RS full 85.79 at 4000 Hz;  actuator-only 85.76;
                          parametric-only 0.577 at 639 Hz;
                          max |K| over 2.4-4.6 kHz = 9.7e7
    mu_paper (stored)     mu_RS full 1.157 at 1178 Hz;  actuator-only 0.651;
                          parametric-only 0.829 at 1178 Hz;
                          max |K| over 2.4-4.6 kHz = 1.0e6

So the scheduled member's failure is ONE thing: controller gain in the band of
the truncated modes (2787 / 3351 / 4122 Hz), where the additive actuator
weights W_Pau, W_Paf are large.  Its parametric reserve (0.577) is far better
than the fixed D-K's (0.829) -- scheduling bought it.  What D-K buys, and the
LQG structure lacks, is the 100x gain reduction in that band.

This module supplies that reduction as STRUCTURE instead of synthesis: a fixed
series filter

    F(s) = notch(f3) notch(f4) notch(f5) x lowpass2(fr)

with the notch centres AT the truncated modes -- the physically motivated
spillover guard -- and everything parametrized by three numbers (notch depth d,
notch width q, rolloff corner fr).  The member is then

    PS-AC-RF :  u = -K(x_P) xhat,  filtered by F(s),

built by the SAME closed-form Riccati chain as PS-AC-R: deterministic,
byte-reproducible, no D-scale fit, no reproducibility gap.  Whether it reaches
G1 is decided by the same judge as every other member.

The cheap in-loop proxy for the G1 actuator excess is

    max over the band  |W_Pau(jw) K(jw) S(jw)| ,  S = 1/(1 - P_u K)

(the y_Paf row of the RS cut is zero, so the actuator block measures exactly
|W_Pau K S|); the full structured-mu judge runs only on the shortlist.

    python phase2/act_filter.py scan     (filter grid on the stored gains)

Writes results/log_act_filter.txt; the retune stage lives in run_psacrf.py.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from ctrl2 import series, ps_ac, Ctrl
from fopid import rolloff_ss, ss_frf

OUT = C.RESULTS
F_TRUNC = tuple(C.F_MEASURED[2:5])          # 2787, 3351, 4122 Hz
BAND_HI = (2200.0, 4800.0)                  # where the additive weights live
BAND_LO = (300.0, 1400.0)                   # the working band of modes 1-2


def notch_ss(f0, depth, q):
    """H(s) = (s^2 + 2 zz w s + w^2) / (s^2 + 2 zp w s + w^2), zz = depth*zp:
    |H(j w0)| = depth, unity far away.  Balanced 2-state realisation."""
    w = 2 * np.pi * float(f0)
    zp = 1.0 / (2.0 * float(q))
    zz = float(depth) * zp
    a1 = 2 * zp * w
    b1 = 2 * (zz - zp) * w
    A = np.array([[0.0, w], [-w, -a1]])
    B = np.array([[0.0], [1.0]])
    Cm = np.array([[0.0, b1]])
    D = np.array([[1.0]])
    return A, B, Cm, D


def lead_ss(f_l, k):
    """Phase-lead L(s) = k (s + z)/(s + p), z = w_l/sqrt(k), p = w_l sqrt(k):
    unit DC gain, +asin((k-1)/(k+1)) phase at f_l, HF gain k.  One state."""
    w = 2 * np.pi * float(f_l)
    rk = np.sqrt(float(k))
    z, p = w / rk, w * rk
    return (np.array([[-p]]), np.array([[1.0]]),
            np.array([[k * (z - p)]]), np.array([[float(k)]]))


def act_filter_ss(depth, q, fr, orders=2, lead=None):
    """The fixed actuator-band filter: three notches + a second-order rolloff,
    optionally with a phase-lead (f_l, k) restoring phase at the working
    modes -- the named candidate against the J ceiling."""
    ss = notch_ss(F_TRUNC[0], depth, q)
    for f0 in F_TRUNC[1:]:
        ss = series(ss, notch_ss(f0, depth, q))
    if fr:
        ss = series(ss, rolloff_ss(float(fr), orders))
    if lead is not None:
        ss = series(ss, lead_ss(*lead))
    return ss


def ps_ac_rf(plant, q_pos, q_vel, r, ratio, a4_mult, depth, q, fr,
             name='PS_AC_RF'):
    """PS-AC-R with the actuator filter in SERIES: same scheduled Riccati law,
    plus the fixed F(s).  Parameter count 5 + 3, reported as exactly that.

    KEPT AS THE RECORD of the first two design rounds (log_psacrf.txt): the
    appended filter breaks certainty equivalence -- the observer believes its
    command reaches the plate while F(s) distorts it -- and the constrained
    PSO landed on parametrically fragile designs twice (mu 1.7-2.1 at mode 2,
    then 2.2-2.8 at mode 1, against 0.58 for the unfiltered member).  The
    member that fixes this is ps_ac_rfa below.
    """
    inner = ps_ac(plant, q_pos, q_vel, r, ratio, sched_K=True, sched_L=False,
                  a4_mult=a4_mult)
    Fss = act_filter_ss(depth, q, fr)

    def build(x, e):
        ss, _ = inner.at(x, e)
        return series(ss, Fss), None

    return Ctrl(name, 8, builder=build, scheduled=True,
                meta=dict(grid=inner.meta['grid'], depth=depth, q=q, fr=fr,
                          a4_mult=a4_mult))


# ---------------------------------------------------------------------------
def ps_ac_rfa(plant, q_pos, q_vel, r, ratio, a4_mult, depth, q, fr,
              name='PS_AC_RFA', lead=None):
    """The actuator-aware member with the filter INSIDE the design model.

    u_cmd -> F(s) -> u_p -> plate.  The filter is part of the controller, so
    its states are KNOWN -- nothing about them is estimated -- and certainty
    equivalence survives:

      * LQR on the augmented model [x_F; x_p] at each scheduling position
        (cut model, a4_mult * alpha_40, the envelope convention), weighting
        the plate states and the command; the filter states get a relative
        regularisation 1e-9 q_pos so the augmented Riccati is well posed.
        The Riccati is solved in SCALED coordinates -- filter states carry
        volt-sized signals and plate states metre-sized ones, eight decades
        apart, and the raw 12-state ordqz dies at the high-gain weights the
        mu physics demands; a diagonal similarity (x_F scaled by 1e-8) and
        a time scale fix that EXACTLY (the gain is invariant);
      * the observer estimates the PLATE alone, position-blind
        (sched_L=False), driven by the exact filtered command u_p.  (An
        'observer-consistent' variant that kept the plain plate LQR and
        only re-wired the observer was tried and is structurally wrong:
        with a fast observer, the internal observer <-> filter loop is
        itself unstable -- separation does not survive lag in the command
        path.  The augmented LQR internalises that lag; this is the one
        structure of the three that stands.);
      * u_cmd = -K_aug(x_P) [x_F; xhat_p].

    Same F(s) structure (three notches at the truncated modes + second-order
    rolloff), same 5 + 3 = 8 searched parameters, same outer realisation
    convention (the common 8 kHz rolloff).
    """
    n = plant.n
    Q = np.diag(np.concatenate([q_pos * np.ones(n), q_vel * np.ones(n)]))
    xs = np.linspace(0.0, plant.plate.lp, 21)
    Af, Bf, Cf, Df = [np.atleast_2d(np.asarray(m, float))
                      for m in act_filter_ss(depth, q, fr, lead=lead)]
    Bf = Bf.reshape(-1, 1)
    Cf = Cf.reshape(1, -1)
    nf = Af.shape[0]
    Wbar = _mean_noise(plant)
    SF, WSC = 1e-8, 2.0 * np.pi * 2000.0

    def build(x_pos, eta=0.0):
        from scipy.linalg import solve_continuous_are
        x = 0.5 * plant.plate.lp if x_pos is None else float(x_pos)
        x = float(np.clip(x, xs[0], xs[-1]))
        A, _, B, E, Cy = plant.matrices(x_pos=x, eta=0.0,
                                        a4=a4_mult * plant.a40)
        A0, B0, _ = plant.structure_only(0.0)
        # augmented design model [x_F ; x_p], scaled: x_F' = SF x_F, t' = WSC t
        Aa = np.block([[Af, np.zeros((nf, 2 * n))],
                       [B @ Cf, A]])
        Ba = np.vstack([Bf, B @ Df])
        Qa = np.zeros((nf + 2 * n, nf + 2 * n))
        Qa[nf:, nf:] = Q
        T = np.diag(np.concatenate([SF * np.ones(nf), np.ones(2 * n)]))
        Ti = np.diag(1.0 / np.diag(T))
        As, Bs = T @ Aa @ Ti / WSC, T @ Ba / WSC
        Qs = Ti @ Qa @ Ti / WSC
        # relative regularisation of the (cost-free) filter states, in the
        # solver's own coordinates, so the Riccati pencil is well posed
        Qs[:nf, :nf] = (1e-9 * np.trace(Qs[nf:, nf:]) / (2 * n)
                        * np.eye(nf))
        Pa = solve_continuous_are(As, Bs, Qs, np.atleast_2d(r / WSC))
        Ka = np.linalg.solve(np.atleast_2d(r), Ba.T @ (T @ Pa @ T))
        Kf, Kp = Ka[:, :nf], Ka[:, nf:]
        Po = solve_continuous_are(A0.T, Cy.T,
                                  ratio * Wbar + 1e-12 * np.eye(2 * n),
                                  np.atleast_2d(1.0))
        L = Po @ Cy.T
        # controller realisation, y -> u_p, states [x_F ; xhat_p]
        Ac = np.block([
            [Af - Bf @ Kf, -Bf @ Kp],
            [B0 @ (Cf - Df @ Kf), A0 - B0 @ (Df @ Kp) - L @ Cy]])
        Bc = np.vstack([np.zeros((nf, 1)), L])
        Cc = np.hstack([Cf - Df @ Kf, -Df @ Kp])
        Dc = np.zeros((1, 1))
        return _ro((Ac, Bc, Cc, Dc)), None

    return Ctrl(name, 8 if lead is None else 10, builder=build,
                scheduled=True,
                meta=dict(grid=xs, depth=depth, q=q, fr=fr,
                          a4_mult=a4_mult, augmented=True, lead=lead))


def _mean_noise(plant):
    n = plant.n
    W = np.zeros((2 * n, 2 * n))
    xs = np.linspace(0.0, plant.plate.lp, 21)
    for x in xs:
        _, _, _, E, _ = plant.matrices(x_pos=x)
        W += E @ E.T
    return W / len(xs)


def _ro(ss):
    return series(ss, rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))


# ---------------------------------------------------------------------------
def band_proxy(plate, ss, f=None):
    """(hi, lo): max |W_Pau K S| over the truncated-mode band and the working
    band -- the cheap stand-in for the actuator block of the G1 judge."""
    from weights import weight_mag, W_PAU
    from plate_model import plant_frf
    if f is None:
        f = np.concatenate([np.linspace(*BAND_LO, 90),
                            np.linspace(1400.0, 2200.0, 30),
                            np.linspace(*BAND_HI, 140)])
    om = 2 * np.pi * f
    K = ss_frf(ss, om)
    Pu, _ = plant_frf(plate, f, C.N_MODES)
    S = 1.0 / (1.0 - Pu * K)
    w = weight_mag(W_PAU, f) * np.abs(K * S)
    hi = float(np.max(w[(f >= BAND_HI[0]) & (f <= BAND_HI[1])]))
    lo = float(np.max(w[(f >= BAND_LO[0]) & (f <= BAND_LO[1])]))
    return hi, lo


# ---------------------------------------------------------------------------
def scan():
    """Filter grid on the STORED PS-AC-R gains: which (depth, q, fr) kill the
    actuator excess without killing the loop.  Shortlist judged by full mu."""
    from plate_model import build_plate
    from plant_ss import ControlledPlant
    from eval2 import evaluate
    from musyn_sched import judge_point
    from robust_design import freq_grid

    fh = open(os.path.join(OUT, 'log_act_filter.txt'), 'w')

    def log(*a):
        line = ' '.join(str(x) for x in a)
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()

    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate, ap=C.AP_S)
    import pickle
    with open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb') as f_:
        store = pickle.load(f_)
    u = dict(store['ps_ac_r']['params'])
    f_judge = freq_grid()

    log('=' * 78)
    log('ACTUATOR FILTER SCAN on the stored PS-AC-R gains')
    log('=' * 78)
    base = ps_ac_rf(plant, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                    10 ** u['log_r'], 10 ** u['log_ratio'], u['a4_mult'],
                    1.0, 2.0, None)          # depth 1 = no filter
    hi0, lo0 = band_proxy(plate, base.at(0.5 * plant.plate.lp)[0])
    J0 = evaluate(plate, base)
    log(f'  unfiltered: proxy hi {hi0:8.3f}  lo {lo0:6.3f}   J = {J0:+.4f}')
    log('')
    log(f'  {"depth":>6} {"q":>5} {"fr":>6} | {"hi":>8} {"lo":>6} '
        f'{"J":>9}')

    rows = []
    for depth in (0.03, 0.1, 0.3):
        for q in (1.0, 2.0, 4.0):
            for fr in (None, 2500.0, 1800.0):
                c = ps_ac_rf(plant, 10 ** u['log_q_pos'],
                             10 ** u['log_q_vel'], 10 ** u['log_r'],
                             10 ** u['log_ratio'], u['a4_mult'],
                             depth, q, fr)
                hi, lo = band_proxy(plate, c.at(0.5 * plant.plate.lp)[0])
                J = evaluate(plate, c)
                rows.append((depth, q, 0.0 if fr is None else fr, hi, lo, J))
                log(f'  {depth:6.2f} {q:5.1f} {0 if fr is None else fr:6.0f}'
                    f' | {hi:8.3f} {lo:6.3f} {J:+9.4f}')

    # shortlist: the actuator excess killed, the loop alive
    rows.sort(key=lambda r: (r[3] > 0.55, -r[5]))
    log('')
    log('  full G1 judge (mu_RS point, actuator included) on the 3 best:')
    for depth, q, fr, hi, lo, J in rows[:3]:
        c = ps_ac_rf(plant, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                     10 ** u['log_r'], 10 ** u['log_ratio'], u['a4_mult'],
                     depth, q, fr if fr else None)
        t = time.time()
        vals = [judge_point(plate, frx * plate.lp,
                            c.at(frx * plate.lp)[0], f_judge)
                for frx in (0.0, 0.5, 1.0)]
        log(f'  depth {depth:.2f} q {q:.1f} fr {fr:.0f}: '
            f'RS(3 nodes) = ' + '  '.join(f'{v:6.3f}' for v in vals)
            + f'   sup {max(vals):6.3f}  [{time.time()-t:.0f}s]'
            + ('  <-- G1' if max(vals) < 1 else ''))

    log(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    scan() if 'scan' in sys.argv[1:] else scan()
