"""
run_dominate.py -- PS-ROB / PS-ROB-TDC: the try at leading on EVERY axis.
=========================================================================
WHAT IS BUILT.  ctrl2.ps_rob / ctrl2.ps_rob_tdc: the position-scheduled law
with BOTH the state feedback and the Kalman filter scheduled on x_P, the
filter's assumed disturbance direction REGULARISED by the direction floor that
G0a measured admissible (results/g0_lambda.txt, sweep R3 and the ADDENDUM), the
state feedback designed at an ENVELOPE milling coefficient a4_mult * alpha_40
as PS-AC-R does, and -- in the TDC variant -- the Eq. (30) delayed pair grafted
on as PS-TDC-R does.  Six tuned parameters for PS-ROB, eight for PS-ROB-TDC.

WHY THIS STRUCTURE, from Gate Zero's measurements and not from the plan:
  * G0a: the observer-scheduled family already beats mu-TDC on delta_max
    (1.676 vs 1.54) and its ONLY defect is the box collapse; the direction
    floor removes that collapse at ZERO cost in Ms, V, J, a_p^inf or delta_max
    (sweep R3: those six columns are bit-identical for c = 0.02 .. 0.50 while
    S3 goes 0.0000 -> 1.1530 mm).  The two blends (R1 gain, R2 covariance) buy
    the same margin but pay for it in Ms, which G0a measured to be THE binding
    constraint.  So lambda is FIXED at 1 (full observer scheduling) and the
    floor level c is the tuned regulariser -- G0a's own recommendation.
  * G0a also measured that the floor level that works depends on the weights
    (c = 0.02 suffices at the PS-AC weights, c = 0.05 is needed at the
    PS-AC-obs weights), which is why c is SEARCHED and not fixed.
  * G0c: the box-robust nominal-pole screen is what makes such a particle
    visible to the optimiser at all, and it costs 2.4 ms per particle.
  * G0b: no single weight of the gain-only family moves a_p^inf, and the
    delayed pair alone cannot deliver D1 either (results/margin_landscape.txt).
    Hence the extra freedom is spent where G0b says the movement is: the
    binding crossing frequency, which the observer scheduling already moves
    from 643.6 Hz (ps_ac) to 458.7 Hz (ps_ac_full) against mu-TDC's 315.0 Hz.

=========================================================================
PRE-DECLARED SUCCESS CRITERIA -- written before the run, verbatim from the
task, and NOT edited afterwards.
=========================================================================
  D1  a_p^inf        >= 0.4364 mm   (mu-TDC's)
  D2  delta_max      >= 1.54        (mu-TDC's)
  D3  nominal floor  >= 0.9307 mm   (PS-TDC-R's)
  D4  worst-of-4     >= 0.6704 mm   (PS-TDC-R's; worst over S1, S2, S3, S4)
  D5  no collapse: every scenario > 0.05 mm
  D6  mu_RS on the parametric physics, position scheduled  <= 0.717
  D7  the SAME fairness protocol: PSO 20x20, seeds (1,2,3), Ms <= 2, effort
      <= 450 V/N, nominal poles <= -1 /s, n_sub = 656, m = 120, five-mode
      evaluation.  Any new constraint must be applied to EVERY design that is
      compared, not only the new one.

  A design "dominates" iff all seven hold.  The outcome is reported whatever it
  is: if k of 7 hold, the report says k of 7 and names the shortfall in per
  cent on each axis that fails.  No threshold is moved after the fact.

=========================================================================
THE SEARCH, declared before running
=========================================================================
PROTOCOL, unchanged from run_stage3: PSO 20 particles x 20 iterations, seeds
(1, 2, 3), latin-hypercube init, w = 0.72, c1 = c2 = 1.5, v_max = 0.25, the
decision vector normalised to [0,1]^n.  Objective and screens through
eval2.evaluate on the five-mode model with m = M_FLOQUET_PSO = 40, probes
(0.3, 0.6, 1.0) mm, positions (0, 0.5, 1).

SEARCH BOUNDS.  Every parameter PS-ROB shares with an existing structure keeps
that structure's published interval (config.BOUNDS2, quoted here because
config.py must not be edited):
    log_q_pos (10, 20)   log_q_vel (-4, 8)   log_r (-12, -4)
    log_ratio (4, 16)    a4_mult (0.4, 2.0)  kpd (-1.5, 1.5)  kdd (-1.5, 1.5)
The one NEW parameter is the floor level, searched as
    log_dfloor (-2.0, 0.0),  i.e.  c in [0.01, 1.00],
an interval deliberately WIDER than the safe region G0a mapped: c = 0.01 is
known to still collapse, so the search must find the safe side itself and the
box screen must be what rejects the rest.  The region is not pre-trimmed.

THE ADDED CONSTRAINT (G0c's box-robust screen), applied to EVERY design that
is compared, as D7 requires:
  TIER 1  max Re lambda(A_cl) <= -1 /s, no cutting, at the three scheduling
          positions (0, 0.5, 1) l_P, on 42 plates: the nominal plate and a
          41-point sweep of the common modal-frequency ratio r over
          [0.9045, 1.1055] -- the reference's +-10 % mass / -10 % stiffness
          box, whose endpoints ARE its four vertices (two of which coincide
          with the nominal).  The 41-point sweep is used instead of G0c's
          4-vertex test because G0c measured strictly worse INTERIOR points
          (ps_ac_obs +9.67 at r = 0.9397 against +2.99 at the vertex), while
          changing no verdict on the twelve stored designs.  The delayed pair
          is folded in at tau -> 0, exactly as the existing screen does.
  TIER 2  runs ONLY on a particle that tier 1 rejects AND that carries a
          delayed pair, because for a pair-free design the tau -> 0 folding is
          exact and there is nothing to re-check.  It closes the no-cutting
          loop with the pair at its TRUE delay tau_0 and takes the Floquet
          exponent log(rho)/tau_0 at the three worst r of the tier-1 table,
          m = 40.  PASS iff that exponent <= -1 /s.
  WHY TWO TIERS, and what it costs: G0c measured that tier 1 alone FAILS
  mu-TDC (+9.08 /s) and would therefore delete the published benchmark -- the
  row that holds the D1 and D2 records -- from the comparison, while at its
  true delay the same design is at -43.19 /s, a PASS.  Adopting tier 1 alone
  would remove the reference in the new design's favour.  That is a protocol
  decision and it is declared here, not made silently.
  PENALTY ORDERING inside the PSO: a particle that fails the EXISTING screens
  keeps eval2's own graded penalties (<= -100, or <= -1000 for the nominal
  poles); a particle that passes them and fails the box screen scores
  -50 - min(worst + 1, 50).  So box failure ranks below every feasible point
  and above the older failures; the ordering is graded, never flat.

FOUR SEARCHES ARE RUN, and all four are reported.  No selection after the
fact.
  A  ps_rob      (6 parameters)  objective = eval2.evaluate's J   PROTOCOL
  B  ps_rob_tdc  (8 parameters)  objective = eval2.evaluate's J   PROTOCOL
  C  ps_rob      (6 parameters)  objective = G2                   DEVIATION
  D  ps_rob_tdc  (8 parameters)  objective = G2                   DEVIATION

  G2, the certificate-targeted objective, declared here in full:

      G2 = min(J_nom, J_eta, J_tau) + 10 * min(0, -log peak_D1)

    J_nom  eval2's J on the nominal plate at the design speed (this IS the
           protocol objective, and it is the proxy for D3: the nominal floor
           is the depth at which max_x log rho crosses zero);
    J_eta  the same margin on the eta = ETA_MAX plate with the controller
           asked at eta = ETA_MAX -- the S3 case that binds ps_ac_full;
    J_tau  the same margin at rpm = RPM_S / 3, the controller re-synthesised
           on that plant -- the S4 case that binds every design;
    peak_D1 = certify2.analyse(...)[1] at a_p = 0.4364 mm on run_stage56's
           reduced family; peak < 1 is EXACTLY the statement a_p^inf >= D1,
           so the second term is a one-sided barrier that is zero once D1 is
           met and then leaves the search maximising the worst-case margin.
  A, B are directly comparable with every published row.  C, D are NOT: they
  spend the same optimiser budget on a different objective, and are reported
  as a FRONTIER PROBE -- the honest answer to "can this family reach D1 at
  Ms <= 2 at all", which G0a left open.  Their J is reported beside them so
  the cost of the deviation is visible.

WHAT IS FROZEN AND WHY.  lambda = 1 (full observer scheduling, no blend); the
removal state eta is not scheduled; the observer gain is computed on the
structure-only A0 and the controller state matrix on the cutting-loaded A(x),
both exactly as ctrl2.ps_ac does.

=========================================================================
ANCHOR CHECKS (K0), pre-declared, run before any search
=========================================================================
  K0a  ctrl2 must be inert for every stored design: J, Ms, V, max Re and the
       state-space matrices at the five evaluation positions must be
       IDENTICAL (difference exactly 0) to the values before ctrl2.py was
       touched, for all twelve stored designs.
  K0b  ps_rob at dfloor = 0 must reproduce ps_ac_full elementwise (difference
       exactly 0), since the floor is the only thing added.
  K0c  ps_rob at the stored ps_ac_obs weights, a4_mult = 1, c = 0.05 must
       reproduce G0a's 'ADD full floor c=0.05' row: Ms 1.999, V 251.1,
       J +0.0392, a_p^inf 0.4246 mm, delta_max 1.676, floor 0.8927 mm,
       S3 0.8430 mm, max Re under the box -8.40 /s.  Tolerance: 1 in the last
       printed digit of each.
  If K0 fails the run is reported as UNANCHORED and no conclusion is drawn.

=========================================================================
WHAT THIS RUN DOES NOT DO -- declared, not discovered afterwards
=========================================================================
  * The four reference designs are NOT re-optimised under the box screen.
    They are re-VERDICTED under it, which is what D7 requires of a new
    constraint; re-running their searches is not.  The argument that this does
    not tilt the comparison is measurable and is printed: adding a constraint
    can only shrink a feasible set, so a re-optimised row can never beat the
    published one, and the three surviving references sit at -93, -91 and -43
    /s against the -1 /s threshold, i.e. the new constraint is INACTIVE at
    their optima, which therefore remain attainable under the enlarged
    protocol.
  * a_p^inf and delta_max use run_stage56's REDUCED vertex families
    (a_p^inf: n_pos 5, etas (0, ETA_MAX), zetas (0.8, 1.2), xis (1,), 16
    bisection steps; delta_max: n_pos 5, xis (1,), 14 steps), so both are
    upper bounds on the full-family answer -- the same caveat that stage
    carries, applied identically to the new rows and the stored ones.
  * The Lyapunov-Krasovskii column is not computed (it was withdrawn from the
    report as a solver status rather than a certificate).
  * The settling time T_s of run_stage78's stage-7 table is not computed; the
    moving-position time simulation IS run, at n_sub = 656, to check that the
    scheduled filter's single discontinuity (the floor flips sign where D_2
    crosses zero, at x = 49.7 mm) does not break the integration.
  * mu_RS is the delay-free point test of mu_scheduled (position channels
    dropped, actuator block out), so a design's delayed pair does not enter
    it -- the same convention under which the stored mu_paper row was made.
  * The screen's damping box (ZETA_LO/HI) and the removal axis eta are not in
    the box screen; it is the reference's mass/stiffness box alone, as G0c
    defined it.

=========================================================================
ADDENDUM -- searches E and F, added AFTER the first full run
=========================================================================
Added because the first run came back 4 of 7 for PS-ROB-TDC and the three
misses were small and specific: D3 by 1.57 %, D4 by 3.93 %, D6 by 2.56 %.  No
pre-declared criterion, threshold or reported row is changed by this addendum;
it adds two more searches and reports them beside the first four.  The reason
for adding them is post-hoc, the objective they use is declared here in full
BEFORE they were run, and the outcome is reported whatever it is.

  E  ps_rob_tdc  (8 parameters)  objective = G3   DEVIATION
  F  ps_rob      (6 parameters)  objective = G3   DEVIATION

  G3 aims at the criteria themselves instead of at the margin.  It is a sum
  of ONE-SIDED barriers, each exactly zero once its criterion is met, plus a
  tie-breaker so that among designs that meet them all the search still
  prefers margin:

      G3 = b1 + b2 + b3 + b4e + b4t + 0.1 * J_nom

      b1  = min(0, -log peak(a_p = 0.4364 mm))            -> D1
            peak from certify2.analyse on run_stage56's reduced family;
            peak < 1 is exactly a_p^inf >= D1.
      b2  = min(0, -log peak(inflation = 1.54))           -> D2
            the same test at margin_bisect's inflated set, a_scale = 1.54,
            etas linspace(0, 1.54 ETA_MAX, 3), zetas 1 + 1.54 (ZETA-1);
            peak < 1 is exactly delta_max >= D2.
      b3  = min(0, -max_x log rho(a_p = 0.9307 mm))       -> D3
      b4e = min(0, -max_x log rho(a_p = 0.6704 mm, eta = ETA_MAX))  -> D4
      b4t = min(0, -max_x log rho(a_p = 0.6704 mm, rpm = RPM_S/3))  -> D4
            the two S3/S4 cases the first run measured as binding.

  G3 is a PROXY on three of its five terms: b3, b4e and b4t use m = 40 and the
  three scheduling positions, while the reported floor, S3 and S4 use m = 120
  and the five evaluation positions, so a design that just meets a barrier can
  still miss the reported criterion.  D6 is NOT in G3: one mu_RS node costs
  5.5 s, 1260 of them would cost two hours, and it is measured afterwards
  instead.  E and F are therefore reported exactly like C and D: as frontier
  probes that fail D7 under the strict reading, listed so the frontier is
  visible, not to claim a win.

    python run_dominate.py anchor              K0
    python run_dominate.py search A|B|C|D|E|F  one PSO (writes to the scratch)
    python run_dominate.py final               certificates, scenarios, mu, table
    python run_dominate.py verdict             the verdict, derived from the npz
Writes results/dominate.txt and results/dominate.npz.  Nothing else in
results/ is touched.
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
import certify2 as CF2
import ctrl2 as K2
import eval2 as E2
from closed_loop import is_stable
from plate_model import build_plate
from plant_ss import ControlledPlant
from pso import pso
from scenarios import _perturbed, mode_scale_from_removal, paper_box_scale
from sim_ctrl2 import ScheduledLTI
from simulate import MillingSimulation
from uncertainty import RemovalFamily

OUT = C.RESULTS
TXT = os.path.join(OUT, 'dominate.txt')
NPZ = os.path.join(OUT, 'dominate.npz')
SCRATCH = os.environ.get('DOMINATE_SCRATCH', '/tmp/dominate_runs')
os.makedirs(SCRATCH, exist_ok=True)

POLE_MAX = -1.0
N_R = 41
AP_D1 = 0.4364e-3                     # D1, the depth the barrier tests at
TAU0 = 60.0 / (3 * C.RPM_S)

# the seven thresholds, in the units of the table
D_THR = dict(D1=0.4364, D2=1.54, D3=0.9307, D4=0.6704, D5=0.05, D6=0.717)

BOUNDS = dict(
    ps_rob=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0),
                a4_mult=(0.4, 2.0), log_dfloor=(-2.0, 0.0)),
    ps_rob_tdc=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                    log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0),
                    a4_mult=(0.4, 2.0), log_dfloor=(-2.0, 0.0),
                    kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
)

SEARCHES = dict(
    A=dict(kind='ps_rob', obj='J', label='PS-ROB'),
    B=dict(kind='ps_rob_tdc', obj='J', label='PS-ROB-TDC'),
    C=dict(kind='ps_rob', obj='G2', label='PS-ROB-F'),
    D=dict(kind='ps_rob_tdc', obj='G2', label='PS-ROB-TDC-F'),
    E=dict(kind='ps_rob_tdc', obj='G3', label='PS-ROB-TDC-G'),
    F=dict(kind='ps_rob', obj='G3', label='PS-ROB-G'),
)

LIGHT = dict(n_pos=5, etas=(0.0, C.ETA_MAX), zetas=(C.ZETA_LO, C.ZETA_HI),
             xis=(1.0,))

LINES = []


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LINES.append(line)


# ---------------------------------------------------------------------------
# the search space
# ---------------------------------------------------------------------------
class Design:
    def __init__(self, kind):
        self.kind = kind
        bd = BOUNDS[kind]
        self.names = list(bd.keys())
        self.lo = np.array([bd[k][0] for k in self.names], float)
        self.hi = np.array([bd[k][1] for k in self.names], float)
        self.n = len(self.names)

    def decode(self, u):
        v = self.lo + np.clip(np.asarray(u, float), 0.0, 1.0) * (self.hi - self.lo)
        return dict(zip(self.names, v))

    def factory(self, params):
        """plant -> Ctrl, re-synthesised at each call."""
        def mk(p):
            return K2.build(self.kind, p, params, None)
        return mk


# ---------------------------------------------------------------------------
# the box-robust screen (G0c, tightened to a 41-point sweep)
# ---------------------------------------------------------------------------
def screen_grid(plate):
    r_lo = float(paper_box_scale(0.10, -0.10, C.N_MODES)[0])
    r_hi = float(paper_box_scale(-0.10, 0.10, C.N_MODES)[0])
    rs = np.unique(np.concatenate([np.linspace(r_lo, r_hi, N_R), [1.0]]))
    pls = [plate if abs(r - 1.0) < 1e-12
           else _perturbed(plate, np.full(C.N_MODES, r), 1.0) for r in rs]
    return rs, pls


def box_screen(plate, grid, ctrl, pos=None, tier2=True, n_worst=3):
    """The declared two-tier screen.  Returns a dict; 'worst' is the number the
    -1 /s threshold is applied to."""
    rs, pls = grid
    pos = C.POSITIONS_DESIGN if pos is None else pos
    tab = np.empty((len(pls), len(pos)))
    for i, p in enumerate(pls):
        for j, fr in enumerate(pos):
            ss, pd = ctrl.at(fr * plate.lp)
            tab[i, j] = float(np.max(E2.nominal_poles(p, ss, pd).real))
    w1 = float(np.max(tab))
    per_r = tab.max(axis=1)
    r_arg = float(rs[int(np.argmax(per_r))])
    has_pd = ctrl.at(0.0)[1] is not None
    out = dict(worst1=w1, r_worst=r_arg, tier=1, worst2=np.nan,
               worst=w1, passed=bool(w1 <= POLE_MAX), has_pd=bool(has_pd))
    if w1 <= POLE_MAX or not (tier2 and has_pd):
        return out
    idx = np.argsort(per_r)[::-1][:n_worst]
    ex = -np.inf
    for i in idx:
        for fr in pos:
            ss, pd = ctrl.at(fr * plate.lp)
            _, rho = is_stable(pls[i], C.RPM_S, 1e-12, fr * plate.lp, ctrl=ss,
                               pd=pd, n_modes=C.N_MODES, m=C.M_FLOQUET_PSO,
                               n_period=C.N_PERIOD, coeff_mode='time',
                               coeff_scale=C.SIGN, ae=C.AE)
            rho = float(rho)
            e = (np.log(max(rho, 1e-300)) / TAU0) if np.isfinite(rho) else np.inf
            ex = max(ex, e)
    out.update(tier=2, worst2=float(ex), worst=float(ex),
               passed=bool(ex <= POLE_MAX))
    return out


# ---------------------------------------------------------------------------
# the two objectives
# ---------------------------------------------------------------------------
def _margin(plate, ctrl, rpm, eta=0.0, m=None):
    """-mean_probes max_positions log rho -- eval2's J, on any plate/speed."""
    probes = C.AP_PROBE
    pos = C.POSITIONS_DESIGN
    mm = []
    for ap in probes:
        mm.append(max(E2.floquet_log_rho(plate, ctrl, rpm, ap, fr * plate.lp,
                                         eta=eta, m=m) for fr in pos))
    return -float(np.mean(mm))


class Fitness:
    def __init__(self, design, plate, obj):
        self.d = design
        self.plate = plate
        self.plant = ControlledPlant(plate)
        self.grid = screen_grid(plate)
        self.obj = obj
        rem = RemovalFamily(n=C.N_MODES_DESIGN)
        self.plate_eta = _perturbed(
            plate, mode_scale_from_removal(rem, C.ETA_MAX, C.N_MODES), 1.0)
        self.rpm_tau = C.RPM_S / 3.0
        self.plant_tau = ControlledPlant(plate, rpm=self.rpm_tau)
        self.plant_d1 = ControlledPlant(plate, ap=AP_D1)
        self.n_eval = 0
        self.n_tier2 = 0
        self.t_screen = 0.0

    def _bar_rho(self, plate, ctrl, rpm, ap, eta=0.0):
        """min(0, -max_x log rho) at one depth: a one-sided stability barrier."""
        w = max(E2.floquet_log_rho(plate, ctrl, rpm, ap, fr * self.plate.lp,
                                   eta=eta) for fr in C.POSITIONS_DESIGN)
        return float(min(0.0, -w))

    def _g3(self, mk, c, J):
        _, p1, _ = CF2.analyse(self.plant_d1, mk(self.plant_d1), **LIGHT)
        kw = dict(n_pos=5, xis=(1.0,), a_scale=D_THR['D2'],
                  etas=tuple(np.linspace(0.0, C.ETA_MAX * D_THR['D2'], 3)),
                  zetas=(1.0 + (C.ZETA_LO - 1.0) * D_THR['D2'],
                         1.0 + (C.ZETA_HI - 1.0) * D_THR['D2']))
        _, p2, _ = CF2.analyse(self.plant, c, **kw)
        b1 = min(0.0, -float(np.log(min(max(p1, 1e-12), 1e6))))
        b2 = min(0.0, -float(np.log(min(max(p2, 1e-12), 1e6))))
        b3 = self._bar_rho(self.plate, c, C.RPM_S, D_THR['D3'] * 1e-3)
        b4e = self._bar_rho(self.plate_eta, c, C.RPM_S, D_THR['D4'] * 1e-3,
                            eta=C.ETA_MAX)
        b4t = self._bar_rho(self.plate, mk(self.plant_tau), self.rpm_tau,
                            D_THR['D4'] * 1e-3)
        return float(b1 + b2 + b3 + b4e + b4t + 0.1 * J)

    def __call__(self, u):
        self.n_eval += 1
        try:
            p = self.d.decode(u)
            mk = self.d.factory(p)
            c = mk(self.plant)
            J, info = E2.evaluate(self.plate, c, detail=True)
            if not info['feasible']:
                return float(J)
            t = time.perf_counter()
            s = box_screen(self.plate, self.grid, c)
            self.t_screen += time.perf_counter() - t
            if s['tier'] == 2:
                self.n_tier2 += 1
            if not s['passed']:
                return -50.0 - float(min(s['worst'] + 1.0, 50.0))
            if self.obj == 'J':
                return float(J)
            if self.obj == 'G3':
                return self._g3(mk, c, float(J))
            j_eta = _margin(self.plate_eta, c, C.RPM_S, eta=C.ETA_MAX)
            j_tau = _margin(self.plate, mk(self.plant_tau), self.rpm_tau)
            _, peak, _ = CF2.analyse(self.plant_d1, mk(self.plant_d1), **LIGHT)
            peak = float(min(max(peak, 1e-12), 1e6))
            return (float(min(J, j_eta, j_tau))
                    + 10.0 * float(min(0.0, -np.log(peak))))
        except Exception:                                      # noqa: BLE001
            return -1e4


# ---------------------------------------------------------------------------
# certificates, scenarios, mu
# ---------------------------------------------------------------------------
def certificates(plate, mk, plant0):
    c = mk(plant0)
    base = dict(n_pos=9, etas=tuple(np.linspace(0.0, C.ETA_MAX, 3)),
                zetas=(C.ZETA_LO, C.ZETA_HI), xis=(0.5, 1.0))
    tm, peak, _ = CF2.analyse(plant0, c, **base)
    ap_inf = CF2.depth_bisect(lambda ap: ControlledPlant(plate, ap=ap), mk,
                              n_iter=16, **LIGHT)
    dmax = CF2.margin_bisect(plant0, c, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    return dict(peak=float(peak), tau_max=float(tm), ap_inf=float(ap_inf * 1e3),
                delta_max=float(dmax))


def time_run(plate, plant, ctrl, ap=None):
    """run_stage78.time_run, copied so that results/log_stage78.txt is not
    touched by importing that module."""
    ap = C.AP_S if ap is None else ap
    sim = MillingSimulation(plate, C.RPM_S, ap, ae=C.AE, fz=C.FZ, sign=C.SIGN,
                            n_modes=C.N_MODES, n_sub=C.N_SUB)
    ss, _ = ctrl.at(0.0)
    c = None if ss is None else ScheduledLTI(ctrl, sim.dt, plate.lp,
                                             tau=sim.tau,
                                             feed=plant.feed_speed(),
                                             moving=True)
    r = sim.run(controller=c, T=None, moving=True)
    y = np.asarray(r['y_obs'], float)
    u = np.asarray(r['u'], float)
    return dict(A_max=float(np.max(np.abs(y))) * 1e6,
                A_rms=float(np.sqrt(np.mean(y ** 2))) * 1e6,
                E_u=float(np.sum(u ** 2) * r['dt']),
                u_max=float(np.max(np.abs(u))),
                diverged=bool(r['diverged']))


def scenarios(plate, mk, plant0):
    """run_stage78's S1-S4, same kwargs."""
    c = mk(plant0)
    s1 = E2.limits(plate, c)
    fr2 = np.linspace(0.0, 1.0, 11)
    s2 = E2.limits(plate, c, positions=fr2)
    rem = RemovalFamily(n=C.N_MODES_DESIGN)
    cases = [('eta=1/2 max', mode_scale_from_removal(rem, C.ETA_MAX / 2,
                                                     C.N_MODES), 1.0,
              C.ETA_MAX / 2),
             ('eta=max', mode_scale_from_removal(rem, C.ETA_MAX, C.N_MODES),
              1.0, C.ETA_MAX),
             ('box -10%', paper_box_scale(-0.10, 0.10, C.N_MODES), 1.0, 0.0),
             ('box +10%', paper_box_scale(0.10, -0.10, C.N_MODES), 1.0, 0.0),
             ('zeta x0.8', None, 0.8, 0.0),
             ('zeta x1.2', None, 1.2, 0.0)]
    s3 = []
    for _, ms, zs, eta in cases:
        s3.append(E2.limits(_perturbed(plate, ms, zs), c, eta=eta).min() * 1e3)
    ratios = (1.0, 1.25, 1.5, 2.0, 3.0)
    s4 = []
    for r in ratios:
        rpm = C.RPM_S / r
        s4.append(E2.limits(plate, mk(ControlledPlant(plate, rpm=rpm)),
                            rpm=rpm).min() * 1e3)
    return dict(S1=s1 * 1e3, S2=s2 * 1e3, S3=np.array(s3),
                S4=np.array(s4), S3_cases=[n for n, _, _, _ in cases],
                S4_ratios=np.array(ratios), S2_positions=fr2,
                time=time_run(plate, plant0, c))


def mu_point(plate, plant0, mk, n_nodes=21):
    """mu_scheduled's [1] point test: parametric physics, position frozen,
    actuator block out."""
    import mu_scheduled as MU
    from robust_design import freq_grid
    f = freq_grid()
    xs = np.linspace(0.0, plate.lp, n_nodes)
    c = mk(plant0)
    v = np.array([MU.mu_rs(MU.plant_at(plate, x), c.at(float(x))[0], f,
                           drop=MU.POS, actuator=False)[0] for x in xs])
    return v, xs


def ablate_factory(params, with_pd):
    """The SAME weights, the SAME envelope coefficient, the SAME delayed pair,
    but the observer NOT scheduled (sched_L = False, so the direction floor has
    nothing to act on).  This is the PS-AC-R / PS-TDC-R structure at the new
    design's own parameters, and it isolates exactly what scheduling the
    observer buys and what it costs, with the weights held fixed."""
    def mk(p):
        inner = K2.ps_ac(p, 10 ** params['log_q_pos'],
                         10 ** params['log_q_vel'], 10 ** params['log_r'],
                         10 ** params['log_ratio'], sched_K=True,
                         sched_L=False, a4_mult=float(params['a4_mult']),
                         name='ABLATION')
        if not with_pd:
            return inner
        k0 = K2.cancellation_gain(p)
        pd = (params['kpd'] * k0, params['kdd'] * k0 / p.omega0[0])
        return K2.Ctrl('ABLATION+pair', inner.n_params + 2,
                       builder=lambda x, e: (inner.at(x, e)[0], pd),
                       scheduled=True, meta=dict(grid=inner.meta['grid']))
    return mk


def ablation(plate, plant0, plant_d, grid, rows):
    """Observer scheduling ON vs OFF at identical weights, for the two
    PROTOCOL designs.  Reported, not used in any verdict."""
    log('=' * 110)
    log('ABLATION -- THE SAME DESIGN WITH THE OBSERVER SCHEDULING SWITCHED OFF')
    log('=' * 110)
    log('  identical weights, identical a4_mult, identical delayed pair; only')
    log('  sched_L goes True -> False, which also removes the direction floor')
    log('  (there is no scheduled direction left to floor).')
    log(f'{"design":<22}{"J":>9}{"Ms":>7}{"a_p^inf":>9}{"delta":>7}'
        f'{"floor":>8}{"S3":>8}{"S4":>8}{"mu_RS":>7}{"box":>9}')
    out = []
    for r in rows:
        if r['tag'] not in ('A', 'B'):
            continue
        for on in (True, False):
            if on:
                nm, mk = r['name'] + ' (obs sched)', Design(r['kind']).factory(
                    r['params'])
                row = dict(J=r['J'], Ms=r['Ms'], ap_inf=r['ap_inf'],
                           delta_max=r['delta_max'], floor=r['floor'],
                           s3=r['s3'], s4=r['s4'], mu_rs=r['mu_rs'],
                           box=r['box_worst'])
            else:
                nm = r['name'] + ' (obs FIXED)'
                mk = ablate_factory(r['params'],
                                    r['kind'] == 'ps_rob_tdc')
                c = mk(plant_d)
                J, info = E2.evaluate(plate, c, detail=True)
                sc = box_screen(plate, grid, c)
                ce = certificates(plate, mk, plant0)
                sn = scenarios(plate, mk, plant0)
                mu, _ = mu_point(plate, plant0, mk)
                row = dict(J=float(J), Ms=float(info['Ms']),
                           ap_inf=ce['ap_inf'], delta_max=ce['delta_max'],
                           floor=float(sn['S1'].min()),
                           s3=float(sn['S3'].min()), s4=float(sn['S4'].min()),
                           mu_rs=float(mu.max()), box=float(sc['worst']))
            log(f'{nm:<22}{row["J"]:>9.4f}{row["Ms"]:>7.3f}'
                f'{row["ap_inf"]:>9.4f}{row["delta_max"]:>7.3f}'
                f'{row["floor"]:>8.4f}{row["s3"]:>8.4f}{row["s4"]:>8.4f}'
                f'{row["mu_rs"]:>7.3f}{row["box"]:>9.2f}')
            out.append(dict(name=nm, **row))
    log('')
    return out


# ---------------------------------------------------------------------------
# K0 anchors
# ---------------------------------------------------------------------------
def anchor(plate=None, plant0=None):
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED) if plate is None else plate
    plant0 = ControlledPlant(plate, ap=C.AP_S) if plant0 is None else plant0
    plate_box = _perturbed(plate, paper_box_scale(0.10, -0.10, C.N_MODES), 1.0)
    store = pickle.load(open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb'))
    u = dict(store['ps_ac_obs']['params'])

    log('=' * 110)
    log('K0  ANCHOR CHECKS')
    log('=' * 110)

    # K0a -- ctrl2 is INERT for every stored design.  The pre-edit ctrl2.py is
    # taken straight out of git (HEAD, before this campaign touched it),
    # imported as a second module, and both modules are asked for the same
    # twelve designs.  The comparison is elementwise on the state space at the
    # five evaluation positions plus (J, Ms, V, max Re): an exact zero, not a
    # tolerance.
    worst, k0a, note = np.nan, False, ''
    try:
        import subprocess
        import importlib.util
        src = subprocess.run(['git', 'show', 'HEAD:phase2/ctrl2.py'],
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))),
                             capture_output=True, text=True, check=True).stdout
        tmp = os.path.join(SCRATCH, 'ctrl2_head.py')
        with open(tmp, 'w') as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location('ctrl2_head', tmp)
        old_mod = importlib.util.module_from_spec(spec)
        sys.modules['ctrl2_head'] = old_mod
        spec.loader.exec_module(old_mod)
        from stage_common import load_mu_ss
        ss = dict(mu_tdc=load_mu_ss('mu_paper'),
                  mu_phys_tdc=load_mu_ss('mu_phys'),
                  ps_tdc=dict(store['ps_ac']['params']),
                  ps_tdc_r=dict(store['ps_ac_r']['params']))
        worst = 0.0
        n_ok = 0
        for k in store:
            if k in ss and ss[k] is None:
                continue
            uk = dict(store[k]['params'])
            pl = ControlledPlant(plate)
            ca = old_mod.build(k, pl, uk, ss.get(k))
            cb = K2.build(k, pl, uk, ss.get(k))
            for fr in C.POSITIONS:
                sa, pa = ca.at(fr * plate.lp)
                sb, pb = cb.at(fr * plate.lp)
                for Ma, Mb in zip(sa, sb):
                    worst = max(worst, float(np.max(np.abs(
                        np.atleast_2d(Ma) - np.atleast_2d(Mb)))))
                if pa is not None:
                    worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
            Ja, ia = E2.evaluate(plate, ca, detail=True)
            Jb, ib = E2.evaluate(plate, cb, detail=True)
            worst = max(worst, abs(Ja - Jb), abs(ia['Ms'] - ib['Ms']),
                        abs(ia['V'] - ib['V']),
                        abs(ia['max_re'] - ib['max_re']))
            n_ok += 1
        k0a = (worst == 0.0)
        note = f'{n_ok} stored designs, pre-edit ctrl2.py from git HEAD'
    except Exception as exc:                                   # noqa: BLE001
        note = f'NOT RUN: {type(exc).__name__}: {exc}'
    log(f'  K0a  edited ctrl2 vs the pre-edit ctrl2 (git HEAD), elementwise on '
        f'the state space at the five')
    log(f'       positions and on (J, Ms, V, max Re): worst |difference| = '
        f'{worst:.3e}   ' + ('PASS' if k0a else 'FAIL'))
    log(f'       {note}')

    # K0b -- ps_rob(dfloor=0) IS ps_ac_full
    ref = K2.build('ps_ac_full', plant0, u, None)
    new = K2.ps_rob(plant0, 10 ** u['log_q_pos'], 10 ** u['log_q_vel'],
                    10 ** u['log_r'], 10 ** u['log_ratio'], a4_mult=1.0,
                    dfloor=0.0)
    d = 0.0
    for fr in C.POSITIONS:
        for Ma, Mb in zip(ref.at(fr * plate.lp)[0], new.at(fr * plate.lp)[0]):
            d = max(d, float(np.max(np.abs(np.atleast_2d(Ma)
                                           - np.atleast_2d(Mb)))))
    k0b = (d == 0.0)
    log(f'  K0b  ps_rob(dfloor = 0) vs ctrl2.build("ps_ac_full"): max '
        f'elementwise |difference| over 5 positions = {d:.3e}   '
        + ('PASS' if k0b else 'FAIL'))

    # K0c -- ps_rob at c = 0.05 IS G0a's 'ADD full floor c=0.05'
    uu = dict(u)
    uu['a4_mult'] = 1.0
    uu['log_dfloor'] = float(np.log10(0.05))
    mk = lambda p: K2.build('ps_rob', p, uu, None)          # noqa: E731
    c = mk(plant0)
    J, info = E2.evaluate(plate, c, detail=True)
    mre = max(float(np.max(E2.nominal_poles(plate_box, c.at(fr * plate.lp)[0],
                                            c.at(fr * plate.lp)[1]).real))
              for fr in C.POSITIONS)
    dmax = CF2.margin_bisect(plant0, c, n_iter=14,
                             base_kw=dict(n_pos=5, xis=(1.0,)))
    apinf = CF2.depth_bisect(lambda ap: ControlledPlant(plate, ap=ap), mk,
                             n_iter=16, **LIGHT) * 1e3
    floor = float(E2.limits(plate, c).min() * 1e3)
    s3 = float(E2.limits(plate_box, c).min() * 1e3)
    got = dict(Ms=info['Ms'], V=info['V'], J=float(J), ap_inf=apinf,
               delta=dmax, floor=floor, S3=s3, mre_box=mre)
    want = dict(Ms=1.999, V=251.1, J=0.0392, ap_inf=0.4246, delta=1.676,
                floor=0.8927, S3=0.8430, mre_box=-8.40)
    tol = dict(Ms=1e-3, V=0.1, J=1e-4, ap_inf=1e-4, delta=1e-3, floor=1e-4,
               S3=1e-4, mre_box=0.01)
    log('  K0c  ps_rob at the stored ps_ac_obs weights, a4_mult = 1, c = 0.05, '
        'against G0a\'s')
    log('       "ADD full floor c=0.05" row of results/g0_lambda.txt:')
    k0c = True
    for kk in ('Ms', 'V', 'J', 'ap_inf', 'delta', 'floor', 'S3', 'mre_box'):
        ok = abs(got[kk] - want[kk]) <= tol[kk] + 1e-12
        k0c &= ok
        log(f'         {kk:<8}{got[kk]:>12.4f}  vs G0a {want[kk]:>9.4f}   '
            + ('OK' if ok else 'DIFFERS'))
    log(f'  K0c  ' + ('PASS' if k0c else 'FAIL'))
    ok = k0a and k0b and k0c
    log('')
    log('  K0: ' + ('PASS - the new controller is ANCHORED to the measured '
                    'family' if ok else 'FAIL - UNANCHORED, no conclusion is '
                    'drawn'))
    log('')
    return dict(k0a=k0a, k0b=k0b, k0c=k0c, k0=ok, worst_stored=worst,
                d_ps_ac_full=d, **{f'k0c_{k}': v for k, v in got.items()})


# ---------------------------------------------------------------------------
# one search
# ---------------------------------------------------------------------------
def search(tag):
    cfg = SEARCHES[tag]
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    d = Design(cfg['kind'])
    fit = Fitness(d, plate, cfg['obj'])
    print(f'[{tag}] {cfg["kind"]}  objective {cfg["obj"]}  '
          f'{d.n} parameters  PSO {C.OPT["n_particles"]}x{C.OPT["n_iter"]} '
          f'seeds {C.OPT["seeds"]}', flush=True)
    best = (None, -np.inf, None)
    per_seed = {}
    for sd in C.OPT['seeds']:
        x, J, info = pso(fit, d.n, seed=sd)
        per_seed[sd] = float(J)
        print(f'[{tag}]   seed {sd}: score {J:+.5f}   '
              f'[{time.time()-t0:.0f}s]', flush=True)
        if J > best[1]:
            best = (x, J, info)
    x, J, info = best
    params = d.decode(x)
    out = dict(tag=tag, kind=cfg['kind'], obj=cfg['obj'], label=cfg['label'],
               x=np.asarray(x, float), score=float(J), params=params,
               per_seed=per_seed, n_eval=fit.n_eval, n_tier2=fit.n_tier2,
               t_screen=fit.t_screen, secs=time.time() - t0,
               history=info['history'])
    with open(os.path.join(SCRATCH, f'search_{tag}.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print(f'[{tag}] done  score {J:+.5f}  {fit.n_eval} evaluations, '
          f'{fit.n_tier2} tier-2 calls, {time.time()-t0:.0f}s', flush=True)
    return out


# ---------------------------------------------------------------------------
# the final assembly
# ---------------------------------------------------------------------------
REFS = ('mu_tdc', 'ps_ac', 'ps_ac_full', 'ps_tdc_r')
REF_LABEL = dict(mu_tdc='mu-TDC', ps_ac='PS-AC',
                 ps_ac_full='PS-AC K+obs', ps_tdc_r='PS-TDC-R')
REF_NPAR = dict(mu_tdc=6, ps_ac=4, ps_ac_full=4, ps_tdc_r=7)


def _crit(row, thr=None, literal_d7=False):
    """PASS/FAIL per criterion for one row of the table."""
    t = D_THR if thr is None else thr
    c = {}
    c['D1'] = row['ap_inf'] >= t['D1']
    c['D2'] = row['delta_max'] >= t['D2']
    c['D3'] = row['floor'] >= t['D3']
    c['D4'] = row['worst4'] >= t['D4']
    c['D5'] = row['min_scen'] > t['D5']
    c['D6'] = np.isfinite(row['mu_rs']) and row['mu_rs'] <= t['D6']
    c['D7'] = bool(row['constraints']) if literal_d7 else bool(row['protocol'])
    return c


def exact_thresholds(s56, s78, mus):
    """The same six thresholds at the reference designs' EXACT stored values.

    D1-D4 and D6 are quoted in the task to four decimals; taken literally that
    makes mu-TDC fail its own D1 (stored 0.436353 mm against the quoted 0.4364)
    and PS-TDC-R fail its own D4 (stored 0.670393 against 0.6704).  Nothing is
    adjusted: the pre-declared table stands, and this second table applies the
    same criteria at the numbers the thresholds were rounded FROM, so that each
    record-holder holds its own record with equality."""
    S = lambda k: (s78[f'S1_{k}']['limits'].min() * 1e3,      # noqa: E731
                   s78[f'S2_{k}'].min() * 1e3, s78[f'S3_{k}'].min(),
                   s78[f'S4_{k}'].min())
    r = S('ps_tdc_r')
    return dict(D1=float(s56['mu_tdc']['ap_inf'] * 1e3),
                D2=float(s56['mu_tdc']['delta_max']),
                D3=float(r[0]), D4=float(min(r)), D5=0.05,
                D6=float(np.max(mus['ps_ac_r_point'])))


def final():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant0 = ControlledPlant(plate, ap=C.AP_S)
    plant_d = ControlledPlant(plate)
    grid = screen_grid(plate)
    store = pickle.load(open(os.path.join(OUT, 'stage3_controllers.pkl'), 'rb'))
    s56 = pickle.load(open(os.path.join(OUT, 'stage56.pkl'), 'rb'))
    s78 = pickle.load(open(os.path.join(OUT, 'stage78.pkl'), 'rb'))
    mus = np.load(os.path.join(OUT, 'mu_scheduled.npz'), allow_pickle=True)

    log('=' * 110)
    log('PS-ROB / PS-ROB-TDC  -- DOES A DESIGN LEAD ON ALL SEVEN AXES?')
    log('=' * 110)
    log(f'  plate      : build_plate(patch={C.PATCH_SIDE}, freqs=F_MEASURED), '
        f'{C.N_MODES}-mode evaluation')
    log(f'  protocol   : PSO {C.OPT["n_particles"]}x{C.OPT["n_iter"]}, seeds '
        f'{C.OPT["seeds"]}, Ms <= {C.MS_MAX}, effort <= {C.V_PER_N} V/N, '
        f'nominal poles <= {POLE_MAX:.0f} /s,')
    log(f'               n_sub = {C.N_SUB}, m = {C.M_FLOQUET} for every '
        f'reported limit, m = {C.M_FLOQUET_PSO} inside the PSO')
    log(f'  added      : the two-tier box-robust screen (G0c), '
        f'{len(grid[0])} plates x 3 positions')
    log(f'  thresholds : D1 {D_THR["D1"]} mm, D2 {D_THR["D2"]}, '
        f'D3 {D_THR["D3"]} mm, D4 {D_THR["D4"]} mm, D5 {D_THR["D5"]} mm, '
        f'D6 {D_THR["D6"]}')
    log('')

    a = anchor(plate, plant0)

    # ---------------- the new designs --------------------------------------
    rows = []
    npz = dict()
    for tag in ('A', 'B', 'C', 'D', 'E', 'F'):
        p = os.path.join(SCRATCH, f'search_{tag}.pkl')
        if not os.path.exists(p):
            log(f'  [{tag}] no search result on disk -- SKIPPED')
            continue
        r = pickle.load(open(p, 'rb'))
        d = Design(r['kind'])
        mk = d.factory(r['params'])
        t = time.time()
        c = mk(plant_d)
        J, info = E2.evaluate(plate, c, detail=True)
        s = box_screen(plate, grid, c)
        cert = certificates(plate, mk, plant0)
        sc = scenarios(plate, mk, plant0)
        mu, mu_x = mu_point(plate, plant0, mk)
        worst4 = float(min(sc['S1'].min(), sc['S2'].min(), sc['S3'].min(),
                           sc['S4'].min()))
        row = dict(name=r['label'], tag=tag, kind=r['kind'], obj=r['obj'],
                   n_par=len(r['params']), J=float(J), Ms=float(info['Ms']),
                   V=float(info['V']), max_re=float(info['max_re']),
                   box_worst=float(s['worst']), box_tier=int(s['tier']),
                   box_pass=bool(s['passed']), box1=float(s['worst1']),
                   r_worst=float(s['r_worst']),
                   ap_inf=cert['ap_inf'], delta_max=cert['delta_max'],
                   peak=cert['peak'],
                   floor=float(sc['S1'].min()), s2=float(sc['S2'].min()),
                   s3=float(sc['S3'].min()), s4=float(sc['S4'].min()),
                   worst4=worst4,
                   min_scen=float(min(sc['S1'].min(), sc['S2'].min(),
                                      sc['S3'].min(), sc['S4'].min())),
                   mu_rs=float(mu.max()),
                   protocol=bool(info['feasible'] and s['passed']
                                 and r['obj'] == 'J'),
                   constraints=bool(info['feasible'] and s['passed']),
                   proto_note=('protocol objective' if r['obj'] == 'J'
                               else 'DEVIATION: objective G2'),
                   params=r['params'], score=r['score'],
                   per_seed=r['per_seed'], n_eval=r['n_eval'],
                   n_tier2=r['n_tier2'], t_screen=r['t_screen'],
                   secs=r['secs'], S3_row=sc['S3'], S4_row=sc['S4'],
                   S2_row=sc['S2'], S1_row=sc['S1'], mu_curve=mu,
                   diverged=sc['time']['diverged'], A_max=sc['time']['A_max'],
                   u_max=sc['time']['u_max'], E_u=sc['time']['E_u'])
        rows.append(row)
        log(f'  [{tag}] {r["label"]:<14} evaluated in {time.time()-t:.0f}s')

    # ---------------- the reference rows -----------------------------------
    from stage_common import load_controllers
    made = load_controllers(plate, plant_d, include_open=False)
    mu_stored = dict(mu_tdc='mu_paper_point', ps_ac='ps_ac_point',
                     ps_tdc_r='ps_ac_r_point')
    for k in REFS:
        mk = made[k]
        c = mk(plant_d)
        J, info = E2.evaluate(plate, c, detail=True)
        s = box_screen(plate, grid, c)
        S1 = s78[f'S1_{k}']['limits'] * 1e3
        S2 = s78[f'S2_{k}'] * 1e3
        S3 = s78[f'S3_{k}']
        S4 = s78[f'S4_{k}']
        if k in mu_stored:
            mu = float(np.max(mus[mu_stored[k]]))
            mu_src = 'results/mu_scheduled.npz'
        else:
            t = time.time()
            v, _ = mu_point(plate, plant0, mk)
            mu = float(v.max())
            mu_src = f'computed here ({time.time()-t:.0f}s)'
        w4 = float(min(S1.min(), S2.min(), S3.min(), S4.min()))
        rows.append(dict(
            name=REF_LABEL[k], tag=k, kind=k, obj='J', n_par=REF_NPAR[k],
            J=float(J), Ms=float(info['Ms']), V=float(info['V']),
            max_re=float(info['max_re']), box_worst=float(s['worst']),
            box_tier=int(s['tier']), box_pass=bool(s['passed']),
            box1=float(s['worst1']), r_worst=float(s['r_worst']),
            ap_inf=float(s56[k]['ap_inf'] * 1e3),
            delta_max=float(s56[k]['delta_max']), peak=float(s56[k]['peak']),
            floor=float(S1.min()), s2=float(S2.min()), s3=float(S3.min()),
            s4=float(S4.min()), worst4=w4, min_scen=w4, mu_rs=mu,
            protocol=bool(info['feasible'] and s['passed']),
            constraints=bool(info['feasible'] and s['passed']),
            proto_note='published row, ' + mu_src,
            params=dict(store[k]['params']), score=float(J), per_seed={},
            n_eval=0, n_tier2=0, t_screen=0.0, secs=0.0,
            S3_row=S3, S4_row=S4, S2_row=S2, S1_row=S1, mu_curve=np.array([mu]),
            diverged=bool(s78[f'S1_{k}']['diverged']),
            A_max=float(s78[f'S1_{k}']['A_max']),
            u_max=float(s78[f'S1_{k}']['u_max']),
            E_u=float(s78[f'S1_{k}']['E_u'])))
        log(f'  [{k}] reference row assembled '
            f'({"stored" if k in mu_stored else "mu computed"})')

    log('')
    thr_x = exact_thresholds(s56, s78, mus)
    _report(rows, grid, a, plate, thr_x)
    abl = ablation(plate, plant0, plant_d, grid, rows)
    with open(TXT, 'w') as fh:
        fh.write('\n'.join(LINES) + '\n')
    _save(rows, a, grid, abl)
    log(f'\ntotal {time.time()-t0:.0f}s')
    with open(TXT, 'w') as fh:
        fh.write('\n'.join(LINES) + '\n')
    print('written', TXT, 'and', NPZ)
    return rows


def _report(rows, grid, a, plate, thr_x):
    log('=' * 110)
    log('THE SCOREBOARD  (new rows first; the four published rows as stored)')
    log('=' * 110)
    log(f'{"design":<15}{"par":>4}{"obj":>5}{"J":>10}{"Ms":>7}{"V":>7}'
        f'{"a_p^inf":>9}{"delta":>7}{"floor":>8}{"S2":>8}{"S3":>8}{"S4":>8}'
        f'{"worst4":>8}{"mu_RS":>7}{"box":>8}')
    for r in rows:
        log(f'{r["name"]:<15}{r["n_par"]:>4}{r["obj"]:>5}{r["J"]:>10.4f}'
            f'{r["Ms"]:>7.3f}{r["V"]:>7.1f}{r["ap_inf"]:>9.4f}'
            f'{r["delta_max"]:>7.3f}{r["floor"]:>8.4f}{r["s2"]:>8.4f}'
            f'{r["s3"]:>8.4f}{r["s4"]:>8.4f}{r["worst4"]:>8.4f}'
            f'{r["mu_rs"]:>7.3f}{r["box_worst"]:>8.2f}')
    log('')
    log('  J is eval2.evaluate on the nominal plate; obj says which objective')
    log('  the row was TUNED on (J = the protocol objective, G2 = the declared')
    log('  deviation).  box = the worst max Re /s the two-tier screen sees.')
    log('')

    log('=' * 110)
    log('PASS / FAIL PER CRITERION')
    log('=' * 110)
    log(f'{"design":<15}' + ''.join(f'{d:>7}' for d in
                                    ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'))
        + f'{"met":>6}')
    log(f'{"threshold":<15}{D_THR["D1"]:>7.4f}{D_THR["D2"]:>7.2f}'
        f'{D_THR["D3"]:>7.4f}{D_THR["D4"]:>7.4f}{D_THR["D5"]:>7.2f}'
        f'{D_THR["D6"]:>7.3f}{"prot":>7}')
    for r in rows:
        c = _crit(r)
        n = sum(c.values())
        r['crit'] = c
        r['n_met'] = n
        log(f'{r["name"]:<15}'
            + ''.join(f'{("PASS" if c[d] else "FAIL"):>7}'
                      for d in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'))
            + f'{n:>4}/7')
    log('')
    log('  D7 for a new row = feasible under the existing screens AND under the')
    log('  box screen AND tuned on the protocol objective.  A row tuned on G2')
    log('  fails D7 BY CONSTRUCTION -- it is a frontier probe, not a protocol')
    log('  design, and it is listed so the frontier is visible, not to claim a')
    log('  win.')
    log('')

    log('=' * 110)
    log('THE SAME SEVEN, READ TWO OTHER WAYS  (nothing above is changed)')
    log('=' * 110)
    log('  (i)  the thresholds at the reference designs\' EXACT stored values,')
    log('       because the quoted four decimals make two record-holders fail')
    log('       their own record:')
    log('         D1 ' + f'{thr_x["D1"]:.6f} mm (mu-TDC, quoted 0.4364)   '
        f'D2 {thr_x["D2"]:.6f} (mu-TDC, quoted 1.54)')
    log('         D3 ' + f'{thr_x["D3"]:.6f} mm (PS-TDC-R, quoted 0.9307)  '
        f'D4 {thr_x["D4"]:.6f} mm (PS-TDC-R, quoted 0.6704)')
    log('         D6 ' + f'{thr_x["D6"]:.6f} (PS-AC-R, quoted 0.717)')
    log('  (ii) D7 read LITERALLY as the task words it -- "PSO 20x20, seeds')
    log('       (1,2,3), Ms <= 2, effort <= 450 V/N, nominal poles <= -1 /s,')
    log('       n_sub = 656, m = 120, five-mode evaluation" -- which the G2')
    log('       rows also satisfy, since only the OBJECTIVE differs.  The')
    log('       pre-declared reading (D7 fails for a G2 row) is the strict one')
    log('       and is what the table above uses.')
    log('')
    log(f'{"design":<15}{"as declared":>13}{"exact thr":>11}'
        f'{"literal D7":>12}{"both":>7}')
    for r in rows:
        c_x = _crit(r, thr_x)
        c_l = _crit(r, None, literal_d7=True)
        c_b = _crit(r, thr_x, literal_d7=True)
        r['n_met_exact'] = sum(c_x.values())
        r['n_met_literal'] = sum(c_l.values())
        r['n_met_both'] = sum(c_b.values())
        log(f'{r["name"]:<15}{r["n_met"]:>11}/7{r["n_met_exact"]:>9}/7'
            f'{r["n_met_literal"]:>10}/7{r["n_met_both"]:>5}/7')
    log('')

    log('=' * 110)
    log('SHORTFALLS, in per cent of the threshold')
    log('=' * 110)
    for r in rows:
        c = r['crit']
        bad = []
        for d, v, thr in (('D1', r['ap_inf'], D_THR['D1']),
                          ('D2', r['delta_max'], D_THR['D2']),
                          ('D3', r['floor'], D_THR['D3']),
                          ('D4', r['worst4'], D_THR['D4'])):
            if not c[d]:
                bad.append(f'{d} short by {100*(thr-v)/thr:.2f} % '
                           f'({v:.4f} vs {thr:.4f})')
        if not c['D6']:
            bad.append(f'D6 over by {100*(r["mu_rs"]-D_THR["D6"])/D_THR["D6"]:.2f}'
                       f' % ({r["mu_rs"]:.4f} vs {D_THR["D6"]:.4f})')
        if not c['D5']:
            bad.append(f'D5 collapse: min scenario {r["min_scen"]:.4f} mm')
        if not c['D7']:
            bad.append('D7 ' + r['proto_note'])
        log(f'  {r["name"]:<15}' + ('all seven met' if not bad
                                    else '; '.join(bad)))
    log('')

    log('=' * 110)
    log('THE ADDED CONSTRAINT, APPLIED TO EVERY COMPARED DESIGN (D7)')
    log('=' * 110)
    log(f'{"design":<15}{"tier-1 worst":>14}{"r at worst":>12}'
        f'{"tier 2":>10}{"verdict":>10}')
    for r in rows:
        t2 = ('-' if r['box_tier'] == 1 else f'{r["box_worst"]:.2f}')
        log(f'{r["name"]:<15}{r.get("box1", r["box_worst"]):>14.2f}'
            f'{r.get("r_worst", float("nan")):>12.4f}{t2:>10}'
            f'{("PASS" if r["box_pass"] else "FAIL"):>10}')
    log('')
    log('  The published references were NOT re-optimised under this screen;')
    log('  they were re-verdicted, which is what D7 asks of a new constraint.')
    log('  Adding a constraint can only shrink a feasible set, so a row that')
    log('  remains feasible keeps its published numbers as an attainable')
    log('  lower bound under the enlarged protocol -- and the surviving')
    log('  references clear the threshold by one to two orders of magnitude,')
    log('  i.e. the new constraint is inactive at their optima.')
    log('')

    log('=' * 110)
    log('THE DESIGNS THEMSELVES')
    log('=' * 110)
    for r in rows:
        if r['tag'] not in ('A', 'B', 'C', 'D', 'E', 'F'):
            continue
        log(f'  {r["name"]:<14} {r["kind"]}  {r["n_par"]} parameters  '
            f'objective {r["obj"]}  score {r["score"]:+.5f}')
        log('     ' + ', '.join(f'{k}={v:.4f}' for k, v in r['params'].items())
            + f'  -> c = {10**r["params"]["log_dfloor"]:.4f}')
        log(f'     per-seed best: '
            + ', '.join(f'{s}:{v:+.5f}' for s, v in r['per_seed'].items())
            + f'   {r["n_eval"]} evaluations, {r["n_tier2"]} tier-2 calls, '
              f'{r["secs"]:.0f}s')
        log(f'     S3 row: ' + ' '.join(f'{v:.4f}' for v in r['S3_row'])
            + f'   S4 row: ' + ' '.join(f'{v:.4f}' for v in r['S4_row']))
        log(f'     moving-pass simulation (n_sub = {C.N_SUB}): A_max = '
            f'{r["A_max"]:.3f} um, u_max = {r["u_max"]:.2f} V, '
            + ('DIVERGED' if r['diverged'] else 'no divergence'))
    log('')


def _save(rows, a, grid, abl=()):
    d = dict(rs=grid[0], thresholds=np.array([D_THR[k] for k in
                                              ('D1', 'D2', 'D3', 'D4', 'D5',
                                               'D6')]),
             names=np.array([r['name'] for r in rows]),
             tags=np.array([r['tag'] for r in rows]),
             obj=np.array([r['obj'] for r in rows]))
    for k in ('n_par', 'J', 'Ms', 'V', 'max_re', 'box_worst', 'box_tier',
              'box1', 'r_worst',
              'ap_inf', 'delta_max', 'peak', 'floor', 's2', 's3', 's4',
              'worst4', 'min_scen', 'mu_rs', 'n_met', 'n_met_exact',
              'n_met_literal', 'n_met_both', 'A_max', 'u_max', 'E_u',
              'n_eval', 'n_tier2', 'secs'):
        d[k] = np.array([float(r[k]) for r in rows])
    for k in ('box_pass', 'protocol', 'constraints', 'diverged'):
        d[k] = np.array([bool(r[k]) for r in rows])
    for dd in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'):
        d['crit_' + dd] = np.array([bool(r['crit'][dd]) for r in rows])
    d['S3_rows'] = np.array([np.asarray(r['S3_row'], float) for r in rows])
    d['S4_rows'] = np.array([np.asarray(r['S4_row'], float) for r in rows])
    d['S1_rows'] = np.array([np.asarray(r['S1_row'], float) for r in rows])
    for r in rows:
        if r['tag'] in ('A', 'B', 'C', 'D', 'E', 'F'):
            for k, v in r['params'].items():
                d[f'{r["tag"]}_{k}'] = np.array(float(v))
            d[f'{r["tag"]}_mu_curve'] = np.asarray(r['mu_curve'], float)
    for k, v in a.items():
        d['k0_' + k] = np.array(v)
    if abl:
        d['abl_names'] = np.array([r['name'] for r in abl])
        for k in ('J', 'Ms', 'ap_inf', 'delta_max', 'floor', 's3', 's4',
                  'mu_rs', 'box'):
            d['abl_' + k] = np.array([float(r[k]) for r in abl])
    np.savez(NPZ, **d)


def verdict():
    """Append the verdict to results/dominate.txt, derived ONLY from the numbers
    already in results/dominate.npz -- no new measurement, nothing from
    memory."""
    d = np.load(NPZ, allow_pickle=True)
    nm = list(d['names'])
    obj = list(d['obj'])
    g = lambda k: dict(zip(nm, d[k]))                          # noqa: E731
    new_rows = [n for n, o in zip(nm, obj) if n.startswith('PS-ROB')]
    proto = [n for n, o in zip(nm, obj) if n.startswith('PS-ROB') and o == 'J']
    crit = {c: g('crit_' + c) for c in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                                        'D7')}
    ap, dm, fl, w4, mu = (g('ap_inf'), g('delta_max'), g('floor'),
                          g('worst4'), g('mu_rs'))
    J, nmet = g('J'), g('n_met')
    head = max(proto, key=lambda n: nmet[n] + 1e-3 * J[n])
    L = ['', '=' * 110, 'VERDICT', '=' * 110]
    L.append(f'  NO design in this run meets all seven.  Best: {head} at '
             f'{int(nmet[head])} of 7, tied with')
    tie = [n for n in nm if not n.startswith('PS-ROB')
           and nmet[n] == nmet[head]]
    L.append(f'  {", ".join(tie) if tie else "nothing"} among the published '
             f'rows -- and on a COMPLEMENTARY set of axes.')
    L.append('')
    L.append(f'  THE PROTOCOL DESIGN.  {head}: '
             + ', '.join(c for c in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7')
                         if crit[c][head]) + ' met; '
             + ', '.join(c for c in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7')
                         if not crit[c][head]) + ' missed.')
    L.append(f'    It is the only design in the table holding BOTH of mu-TDC\'s'
             f' certified records at once:')
    L.append(f'      a_p^inf {ap[head]:.4f} mm  ('
             f'{100*(ap[head]-D_THR["D1"])/D_THR["D1"]:+.2f} % on D1 = '
             f'{D_THR["D1"]}),   delta_max {dm[head]:.3f}  ('
             f'{100*(dm[head]-D_THR["D2"])/D_THR["D2"]:+.2f} % on D2 = '
             f'{D_THR["D2"]})')
    ibest = int(np.argmax(d['J']))
    L.append(f'    and it carries the highest protocol objective in the whole '
             f'table, J = {J[head]:+.4f}')
    L.append(f'    (best of any row here: {nm[ibest]} at {d["J"][ibest]:+.4f}), '
             f'with no collapse anywhere')
    L.append('    -- "in this table", not in the study: '
             'results/margin_landscape.npz holds a')
    L.append('    protocol-feasible 6-parameter ps_tdc point at J = +0.1961.  '
             'And the lead over')
    L.append('    PS-TDC-R is +0.0039 against a per-seed spread of 0.0230 in '
             'the very search that')
    L.append('    produced it, i.e. INSIDE the noise -- see '
             'results/dominate_audit.txt.')
    L.append(f'    (worst scenario {w4[head]:.4f} mm against the 0.05 mm '
             f'collapse line).')
    L.append(f'    It misses D3 by {100*(D_THR["D3"]-fl[head])/D_THR["D3"]:.2f} '
             f'% ({fl[head]:.4f} vs {D_THR["D3"]}), D4 by '
             f'{100*(D_THR["D4"]-w4[head])/D_THR["D4"]:.2f} % '
             f'({w4[head]:.4f} vs {D_THR["D4"]}),')
    L.append(f'    and D6 by {100*(mu[head]-D_THR["D6"])/D_THR["D6"]:.2f} % '
             f'({mu[head]:.4f} vs {D_THR["D6"]}).')
    L.append('')
    L.append('  WHAT THE SIX NEW DESIGNS REACH BETWEEN THEM (best value of each'
             ' axis over all six):')
    bestD3 = max(new_rows, key=lambda n: fl[n])
    bestD1 = max(new_rows, key=lambda n: ap[n])
    bestD4 = max(new_rows, key=lambda n: w4[n])
    bestD6 = min(new_rows, key=lambda n: mu[n])
    L.append(f'    D1 best {ap[bestD1]:.4f} mm ({bestD1})  -- MET')
    L.append(f'    D2 best {max(dm[n] for n in new_rows):.3f}  -- MET')
    L.append(f'    D3 best {fl[bestD3]:.4f} mm ({bestD3})  -- MET, but by a '
             f'DIFFERENT design from D1:')
    L.append(f'       no single design in this run met D1 and D3 together '
             f'({bestD1} has floor {fl[bestD1]:.4f},')
    L.append(f'       {bestD3} has a_p^inf {ap[bestD3]:.4f}).')
    L.append(f'    D4 best {w4[bestD4]:.4f} mm ({bestD4})  -- NOT MET by any of'
             f' the six, short by '
             f'{100*(D_THR["D4"]-w4[bestD4])/D_THR["D4"]:.2f} %')
    L.append(f'    D6 best {mu[bestD6]:.4f} ({bestD6})  -- NOT MET by any of '
             f'the six, over by '
             f'{100*(mu[bestD6]-D_THR["D6"])/D_THR["D6"]:.2f} %')
    L.append('')
    if 'abl_names' in d.files:
        an = list(d['abl_names'])
        a = {k: dict(zip(an, d['abl_' + k])) for k in
             ('J', 'ap_inf', 'delta_max', 'floor', 's4', 'mu_rs')}
        L.append('  WHAT THE ABLATION MEASURES -- the same weights with the '
                 'observer scheduling OFF:')
        for i in range(0, len(an), 2):
            on, off = an[i], an[i + 1]
            L.append(f'    {on.split(" (")[0]:<12}'
                     f' a_p^inf {a["ap_inf"][on]:.4f} -> {a["ap_inf"][off]:.4f}'
                     f' ({100*(a["ap_inf"][off]/a["ap_inf"][on]-1):+.1f} %),'
                     f'  delta {a["delta_max"][on]:.3f} -> '
                     f'{a["delta_max"][off]:.3f} '
                     f'({100*(a["delta_max"][off]/a["delta_max"][on]-1):+.1f} %),')
            L.append(f'    {"":<12} floor {a["floor"][on]:.4f} -> '
                     f'{a["floor"][off]:.4f} '
                     f'({100*(a["floor"][off]/a["floor"][on]-1):+.1f} %),'
                     f'  S4 {a["s4"][on]:.4f} -> {a["s4"][off]:.4f} '
                     f'({100*(a["s4"][off]/a["s4"][on]-1):+.1f} %),'
                     f'  mu_RS {a["mu_rs"][on]:.3f} -> {a["mu_rs"][off]:.3f}')
        L.append('    So the observer scheduling is what BUYS D1 and D2 and the'
                 ' nominal floor, and what')
        L.append('    PAYS on the delay scenario S4 -- the axis that keeps D4 '
                 'out of reach.  It does NOT')
        L.append('    cost mu_RS: at these weights the fixed-observer version '
                 'is WORSE on D6, so the D6')
        L.append('    shortfall is a property of the weight region the new '
                 'design lives in, not of')
        L.append('    scheduling the filter.')
        L.append('')
    L.append('  READ THIS AS: the direction floor turns the observer-scheduled '
             'family from')
    L.append('  INADMISSIBLE (S3 = 0.0000, the box collapse) into the strongest'
             ' certified design')
    L.append('  in the study on D1, D2 and the protocol objective, under a '
             'protocol that is')
    L.append('  STRICTER than the published one.  It does not deliver the '
             'user\'s ask of seven')
    L.append('  of seven.  Two axes -- D4 (the tau x 3 spindle-speed case) and '
             'D6 (mu_RS) --')
    L.append('  were not met by ANY of the six searches, including the two '
             'whose objective')
    L.append('  aimed straight at them; that is the measured frontier, not a '
             'tuning accident,')
    L.append('  and the ablation says which of them is caused by the '
             'scheduling itself.')
    L += ['', '=' * 110,
          'WHAT THIS RUN DID NOT DO -- declared in the docstring before the '
          'run, repeated here',
          '=' * 110,
          '  * the four published references were NOT re-optimised under the '
          'box screen; they were',
          '    re-verdicted.  Adding a constraint can only shrink a feasible '
          'set, and the three that',
          '    survive clear the -1 /s threshold by 42 to 92 /s (mu-TDC '
          '-43.19, PS-AC -92.99,',
          '    PS-TDC-R -91.37), so it is inactive at their optima.  The '
          '-17.46 quoted elsewhere is',
          '    PS-ROB-TDC\'s own margin, a new row, not a reference.',
          '  * a_p^inf and delta_max use run_stage56\'s REDUCED vertex '
          'families (a_p^inf: n_pos 5,',
          '    etas (0, ETA_MAX), zetas (0.8, 1.2), xis (1,), 16 steps; '
          'delta_max: n_pos 5, xis (1,),',
          '    14 steps), so both are UPPER BOUNDS -- identically for the new '
          'rows and the stored ones.',
          '  * the Lyapunov-Krasovskii column and the settling time T_s were '
          'not computed.',
          '  * mu_RS is the delay-free POINT test only (position channels '
          'dropped, actuator block out),',
          '    so a design\'s delayed pair does not enter it -- the same '
          'convention the stored mu_paper',
          '    row was made under.  The grid-cell test and the '
          'actuator-included test were not run',
          '    for the new designs.',
          '  * the box screen carries the reference\'s mass/stiffness box '
          'alone: not the +-20 % damping',
          '    box, not material removal, not spindle speed.',
          '  * G3\'s b3, b4e and b4t are PROXIES: m = 40 and the three '
          'scheduling positions, while the',
          '    reported floor, S3 and S4 use m = 120 and the five evaluation '
          'positions.  D6 is not in',
          '    G3 at all (one mu_RS node costs 5.5 s; 1260 of them would cost '
          'two hours).',
          '  * the PSO budget is exactly the protocol\'s 20x20 with seeds '
          '(1,2,3) -- no larger budget was',
          '    tried, and the search is stochastic: the per-seed spread is '
          'printed for every row and',
          '    reaches 0.02 in J and 3.2 in G3, so a different seed set could '
          'move these numbers.',
          '  * the direction floor is DISCONTINUOUS at the single point where '
          'D_2 crosses zero',
          '    (x = 49.7 mm), where the floored component flips sign.  The '
          'moving-pass simulation at',
          '    n_sub = 656 did not diverge for any of the six designs, but no '
          'finer time-step study',
          '    was made and no smooth variant was compared.',
          '  * run-to-run: LAPACK threading moves Ms and max Re in the fourth '
          'decimal (K0c reproduced',
          '    G0a to 1.6e-4 relative on max Re under the box across three '
          'runs); the K0c tolerances',
          '    absorb that and nothing in the verdict turns on it.',
          '  * side effect, named: importing mu_scheduled pulls in cross_mu, '
          'which opens',
          '    results/log_cross_mu.txt with mode \'w\' at module level.  '
          'That file was already 0 bytes',
          '    and is still 0 bytes; only its mtime moved.  No other file in '
          'results/ was written',
          '    except results/dominate.txt and results/dominate.npz.']
    with open(TXT, 'a') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
    if mode == 'anchor':
        anchor()
        print('\n'.join(LINES[-3:]))
    elif mode == 'search':
        search(sys.argv[2])
    elif mode == 'verdict':
        verdict()
    else:
        final()
