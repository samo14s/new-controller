"""Rigorous check of the frequency condition of Proposition 1 (circle criterion) on the
continuous frequency axis, by interval arithmetic.

Purpose.  circle_criterion.py evaluates mu(a) = inf_w Re[1 + L(i w; a)] on a frequency grid
(0.002 Hz near the structural mode) with local refinement, and a_circ(n, g) is the depth found
by bisection on that grid value.  The referee asked for a conservative bound instead of a grid.
This script proves, with outward-rounded interval arithmetic, that
    Re[1 + L(i w; a)] > 0   for every real w and every depth a in [0, a_v],
    a_v = (1 - 1e-3) a_circ,
at each of the 251 speeds at the scheduled gain g_best (circle_criterion.json, g_best rows),
and gives a certified lower bound on mu over that set.  As a sanity check it also shows, with a
rigorous upper bound, that the condition fails just above a_circ.

Protocol.
  model     the averaged zero-order loop of circle_criterion.L_of, unchanged:
                L(s) = g s Ga(s) G(s) / (1 - KAPPA a (1 - e^{-sT}) G(s)),
            G = 1/(M s^2 + C s + K) (c4_robust.G_struct, one mode), Ga = K_a s^2/(s^2 + a1 s + a0)
            (structure.Ga), aHbar = KAPPA a with KAPPA = c4_robust.KAPPA = N K_tc alpha_yy/(4 pi)
            (negative), T = 60/(N n), damping of results/gainmap_main.npz (1.34 %), set by
            params.set_damping in every process before circle_criterion is imported.
  algebra   with s = i w (exact rewriting; checked against L_of at start-up, see 'selftest'):
                L = c(w) / B(w, a),   c = -i g K_a w^3 / A,   A = a0 - w^2 + i a1 w,
                B = K - M w^2 + i C w - KAPPA a (1 - cos wT + i sin wT)   (= 1/G - aHbar E),
                Re[1 + L] = N / |B|^2,   N = |B|^2 + Re(c conj B) = |B + c/2|^2 - |c|^2 / 4.
            N > 0 on a box proves Re[1 + L] > 0 there and B != 0 (B = 0 would give N = 0).
  intervals real intervals [lo, hi] in float64 (numpy arrays, vectorised over boxes); after every
            +, -, *, / (rounded to nearest by IEEE 754) the bounds are moved one ulp outward
            (np.nextafter towards -inf / +inf); squares are exact in sign (0 inside a straddling
            interval); division only by intervals bounded away from 0 (else (-inf, inf)).
            Complex intervals are pairs of real intervals.  cos and sin of [t1, t2] (t1 >= 0): the
            end-point values (glibc libm, documented maximum error 1 ulp on x86-64; numpy uses it
            here, checked at start-up against math.cos/math.sin), widened by TRIG_PAD = 2^-50 =
            4 ulp(1) in absolute value; the bound is set to +1 / -1 wherever a multiple j pi/2 of
            the matching residue j mod 4 may lie in [t1, t2] (j from ceil(t1/(pi/2) - 1e-9) to
            floor(t2/(pi/2) + 1e-9), a superset); between such points cos and sin are monotone.
            The model constants M, C,
            K, K_a, a1, a0, KAPPA enter as intervals of relative half-width REL_PAR = 1e-12
            around the double values the code uses (this also covers the rounding of the decimal
            inputs and of the derived constants); T = 60/(N n) is an outward-rounded quotient;
            g and the box limits are exact doubles.
  enclosure Re[1 + L] over a box [w1, w2] x [a1, a2] is enclosed three ways, N/|B|^2 with N in
            the expanded form |B|^2 + c_r B_r + c_i B_i, N/|B|^2 with N in the circle form above,
            and 1 + (c_r B_r + c_i B_i)/|B|^2; the intersection is kept, and where N > 0 the
            lower bound N_lo/|B|^2_hi also holds if |B|^2 is not bounded away from 0.
  domain    w in [0, W_MAX], W_MAX = 2 pi 5000 rad/s; a in [0, a_v], a_v = (1 - MARGIN) a_circ,
            MARGIN = 1e-3, a_circ = circle_criterion.json g_best a_circ_mm (grid bisection).
  phase 1   initial boxes: w in steps of at most DTHETA0 = 0.25 rad of w T, each with the whole
            depth range.  A box whose lower bound is <= 0 is bisected in w or in a, whichever
            leaves the wider enclosure when the other variable is fixed at its midpoint (two
            extra evaluations per split, counted).  Until every box has a lower bound > 0
            (verified) or BUDGET1 box evaluations are spent; the boxes left are 'undecided'.  A
            box with an upper bound < 0 is a proven violation (the condition fails at every
            point of it); boxes above the lowest violation are pruned.  a_verified = the
            smallest lower depth of an undecided or violating box (a_v if none): every depth
            below a_verified (all of [0, a_v] if verified) is proved at every frequency.
  phase 2   (margin, only if verified) a proved box is bisected further while its lower bound
            is below min(TIGHT mu_grid(a_v), TIGHT f(centre)), TIGHT = 0.99, f(centre) = the
            float Re[1 + L_of] at its centre, mu_grid(a_v) = circle_criterion.mu(n, a_v, g)
            (the grid value with local refinement); BUDGET2 evaluations.  A child keeps the
            larger of its own and its parent's bound.  min_lb = the smallest bound over the
            final boxes: a rigorous lower bound on Re[1 + L] over [0, W_MAX] x [0, a_v].
  tail      for w >= W_MAX and a in [0, a_v]: |A| >= w^2 - a0, |B| >= |Re B| >= M w^2 - K -
            2 |KAPPA| a_v, so |L| <= g K_a w^3 / ((w^2 - a0)(M w^2 - K - 2 |KAPPA| a_v)); with
            both factors positive at W_MAX this bound decreases in w (d log/dw <= (3-2-2)/w < 0),
            so its value at W_MAX (interval upper end) bounds |L| on [W_MAX, inf); it must be
            < 1, then Re[1 + L] >= 1 - |L| > 0 and B != 0 there.  Negative frequencies:
            L(-iw) = conj L(iw).
  conclusion  verified (phase 1 complete and tail < 1) means mu(a) >= min(min_lb, 1 - tail) > 0
            for every a in [0, a_v] and B(iw, a) != 0 for every real w: no root of the
            uncontrolled cut's characteristic equation (with the actuator) reaches the imaginary
            axis for a in [0, a_v]; stable at a = 0 and with roots moving continuously (retarded
            equation), the linear part L is exponentially stable there, so a_v < a_lim(0) is
            proved as well.  The other hypothesis of Proposition 1, |b(t)| <= Vmax (the forced
            orbit unclipped), is not part of this check (circle_criterion.py:
            first_clipped_depth_mm, a_circ_eff_mm, copied into each row).
  bracket   mu_upper = the upper end of the point enclosure at (w_grid, a_v), w_grid the
            frequency of mu_grid(a_v): mu(a_v) lies in [min_lb, mu_upper].
  fail check (sanity) at a_f = min((1 + FAIL_REL) a_circ, (a_circ + a_lim(0))/2), FAIL_REL =
            1e-3 (below a_lim(0), where L is stable), w_f = the grid minimiser of
            circle_criterion.mu(n, a_f, g): the point enclosure at (w_f, a_f) must have an upper
            bound < 0, i.e. the condition provably fails at a_f, so the largest depth up to
            which it holds lies in [a_v, a_f].  Not applicable where the grid search found no
            failure below a_lim(0) (reached_a0: 2990 and 3000 rpm).

Run (from the repository root ozsoy2026/; resumable):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/circle_interval.py --workers 2
  -> verification/paper/circle_interval.json, cache verification/paper/circle_interval_cache.jsonl
  (one line per finished speed, keyed by the protocol hash, speed, gain and a_circ; a restart
  skips the speeds already there).  Measured (smoke test and 12 hard speeds, one worker): 0.05-0.11 s
  of CPU per speed (1200-26000 box evaluations), 0.5 s of start-up (imports, self-test), so about
  30 s of CPU for the 251 speeds; a speed that exhausts both budgets takes about 4 s.
  Smoke test (writes circle_interval_smoke.json / circle_interval_smoke_cache.jsonl):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/circle_interval.py \\
        --smoke --speeds 1090,2500 --workers 1
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "..", "results")
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
STORED = os.path.join(HERE, "circle_criterion.json")

W_MAX = 2 * np.pi * 5000.0                    # rad/s
MARGIN = 1e-3                                 # a_v = (1 - MARGIN) a_circ
FAIL_REL = 1e-3                               # a_f = min((1 + FAIL_REL) a_circ, (a_circ + a0)/2)
REL_PAR = 1e-12                               # relative half-width of the constant intervals
TRIG_PAD = 2.0 ** -50                         # absolute padding of libm cos/sin (4 ulp(1))
J_SLACK = 1e-9                                # slack of the j pi/2 candidate range
DTHETA0 = 0.25                                # rad of w T per initial box
BUDGET1 = 2_000_000                           # box evaluations, phase 1
BUDGET2 = 2_000_000                           # box evaluations, phase 2
TIGHT = 0.99
BATCH = 1 << 15
N_KEEP = 20                                   # undecided / violating boxes listed per speed
VERSION = "v1"

NINF, PINF = -np.inf, np.inf


def proto_hash(budget1, budget2, tight):
    return hashlib.sha1(repr((ZETA, W_MAX, MARGIN, FAIL_REL, REL_PAR, TRIG_PAD, J_SLACK, DTHETA0, budget1,
                              budget2, tight, VERSION)).encode()).hexdigest()[:10]


# ------------------------------------------------------------ set-up ----
_MOD = {}


def _init_worker():
    """Pool initializer: the damping is set in the worker before anything that caches it is imported
    (with the fork start method the modules come from the parent, where the same order holds)."""
    import params as P
    P.set_damping(ZETA)
    _setup()


def _setup():
    """Damping of every stored result, then the modules that read it (once per process)."""
    if not _MOD:
        import params as P
        P.set_damping(ZETA)
        import circle_criterion as CC           # sets the same damping at import, builds its grid
        _MOD.update(P=P, CC=CC, R=CC.R)
    return _MOD["P"], _MOD["CC"], _MOD["R"]


class JL:
    """Append-only JSON-lines cache: key (a JSON-serialisable list) -> value."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            good = 0
            with open(path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:              # a line cut short by an interruption
                        break
                    self.d[json.dumps(rec["k"])] = rec["v"]
                    good += len(line.encode())
            if os.path.getsize(path) > good:
                with open(path, "r+") as fh:
                    fh.truncate(good)

    def get(self, k):
        return self.d.get(json.dumps(k))

    def put(self, k, v):
        self.d[json.dumps(k)] = v
        with open(self.path, "a") as fh:
            fh.write(json.dumps(dict(k=k, v=v), default=float) + "\n")
            fh.flush()
        return v


# ------------------------------------------------- interval arithmetic ----
def dn(x):
    return np.nextafter(x, NINF)


def up(x):
    return np.nextafter(x, PINF)


def ival(x, rel=0.0):
    """Interval around a double: exact (rel = 0) or of relative half-width rel."""
    x = float(x)
    if rel == 0.0:
        return (x, x)
    d = abs(x) * rel
    return (float(dn(x - d)), float(up(x + d)))


def iadd(x, y):
    return dn(x[0] + y[0]), up(x[1] + y[1])


def isub(x, y):
    return dn(x[0] - y[1]), up(x[1] - y[0])


def ineg(x):
    return -x[1], -x[0]


def imul(x, y):
    p1, p2, p3, p4 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
    return (dn(np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))),
            up(np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))))


def isqr(x):
    l2, h2 = x[0] * x[0], x[1] * x[1]
    mn = np.where((x[0] <= 0) & (x[1] >= 0), 0.0, np.minimum(l2, h2))
    return np.maximum(dn(mn), 0.0), up(np.maximum(l2, h2))


def idiv_pos(x, y):
    """x / y where y > 0; (-inf, inf) where the lower end of y is not positive."""
    ok = np.asarray(y[0]) > 0
    yl = np.where(ok, y[0], 1.0)
    q1, q2, q3, q4 = x[0] / yl, x[0] / y[1], x[1] / yl, x[1] / y[1]
    lo = dn(np.minimum(np.minimum(q1, q2), np.minimum(q3, q4)))
    hi = up(np.maximum(np.maximum(q1, q2), np.maximum(q3, q4)))
    return np.where(ok, lo, NINF), np.where(ok, hi, PINF)


def ihalf(x):
    return x[0] * 0.5, x[1] * 0.5                 # exact (no underflow at these magnitudes)


HALF_PI = np.pi / 2


def itrig(tl, th):
    """Enclosures of cos and sin over [tl, th] (arrays, 0 <= tl <= th up to rounding)."""
    cl, ch, sl, sh = np.cos(tl), np.cos(th), np.sin(tl), np.sin(th)
    clo, chi = dn(np.minimum(cl, ch) - TRIG_PAD), up(np.maximum(cl, ch) + TRIG_PAD)
    slo, shi = dn(np.minimum(sl, sh) - TRIG_PAD), up(np.maximum(sl, sh) + TRIG_PAD)
    jmin = np.ceil(tl / HALF_PI - J_SLACK)        # every j with j pi/2 in [tl, th] is in [jmin, jmax]
    jmax = np.floor(th / HALF_PI + J_SLACK)
    span = jmax - jmin

    def has(r):                                   # some j in [jmin, jmax] with j = r (mod 4)
        return (span >= 3) | ((span >= 0) & (np.mod(r - jmin, 4.0) <= span))
    chi = np.where(has(0), 1.0, chi)              # cos = 1 at 2 k pi
    shi = np.where(has(1), 1.0, shi)              # sin = 1 at pi/2 + 2 k pi
    clo = np.where(has(2), -1.0, clo)             # cos = -1 at pi + 2 k pi
    slo = np.where(has(3), -1.0, slo)             # sin = -1 at 3 pi/2 + 2 k pi
    return (np.maximum(clo, -1.0), np.minimum(chi, 1.0)), (np.maximum(slo, -1.0), np.minimum(shi, 1.0))


class Model:
    """Interval enclosure of Re[1 + L(i w; a)] for one speed and gain."""

    def __init__(self, rpm, g):
        P, CC, R = _setup()
        self.rpm, self.g = float(rpm), float(g)
        self.M = ival(P.M, REL_PAR)
        self.C = ival(P.C, REL_PAR)
        self.K = ival(P.K, REL_PAR)
        self.A0 = ival(P.GA_A0, REL_PAR)
        self.A1 = ival(P.GA_A1, REL_PAR)
        self.KAP = ival(R.KAPPA, REL_PAR)
        self.gKA = imul(ival(g), ival(P.GA_GAIN, REL_PAR))
        T = idiv_pos((60.0, 60.0), imul(ival(rpm), ival(P.N_TEETH)))
        self.T = (float(T[0]), float(T[1]))

    def enclose(self, wl, wh, al, ah):
        """(lo, hi) of Re[1 + L] over the boxes [wl, wh] x [al, ah] (wl >= 0, al >= 0)."""
        W, A = (wl, wh), (al, ah)
        W2 = isqr(W)
        W3 = imul(W2, W)
        COS, SIN = itrig(dn(wl * self.T[0]), up(wh * self.T[1]))
        Er = isub((1.0, 1.0), COS)                 # E = 1 - e^{-i w T} = (1 - cos) + i sin
        KA = imul(self.KAP, A)                     # aHbar = KAPPA a
        Br = isub(isub(self.K, imul(self.M, W2)), imul(KA, Er))
        Bi = isub(imul(self.C, W), imul(KA, SIN))
        Ar = isub(self.A0, W2)
        Ai = imul(self.A1, W)
        H = imul(self.gKA, W3)
        AA = iadd(isqr(Ar), isqr(Ai))
        cr = ineg(idiv_pos(imul(H, Ai), AA))       # c = -i H / A = -H (A_i + i A_r) / |A|^2
        ci = ineg(idiv_pos(imul(H, Ar), AA))
        BB = iadd(isqr(Br), isqr(Bi))
        X = iadd(imul(cr, Br), imul(ci, Bi))       # Re(c conj B)
        N1 = iadd(BB, X)
        N2 = isub(iadd(isqr(iadd(Br, ihalf(cr))), isqr(iadd(Bi, ihalf(ci)))),
                  ihalf(ihalf(idiv_pos(isqr(H), AA))))
        NB = (np.maximum(N1[0], N2[0]), np.minimum(N1[1], N2[1]))
        RE3 = iadd((1.0, 1.0), idiv_pos(X, BB))
        RE1 = idiv_pos(NB, BB)
        lo = np.maximum(RE3[0], RE1[0])
        hi = np.minimum(RE3[1], RE1[1])
        pos = NB[0] > 0                            # N > 0: Re >= N_lo / |B|^2_hi even if |B|^2_lo = 0
        lo = np.where(pos, np.maximum(lo, dn(NB[0] / np.where(pos, BB[1], 1.0))), lo)
        return lo, hi

    def point(self, w, a):
        lo, hi = self.enclose(np.array([float(w)]), np.array([float(w)]), np.array([float(a)]), np.array([float(a)]))
        return float(lo[0]), float(hi[0])

    def tail(self, a_v):
        """Upper bound on |L(i w)| for all w >= W_MAX and a in [0, a_v]; None if the bound does not apply."""
        w = (W_MAX, W_MAX)
        w2 = isqr(w)
        d1 = isub(w2, self.A0)
        absk = ineg(self.KAP)                      # |KAPPA| (KAPPA < 0)
        d2 = isub(isub(imul(self.M, w2), self.K), imul((2.0, 2.0), imul(absk, ival(a_v))))
        if not (absk[0] > 0 and d1[0] > 0 and d2[0] > 0):
            return None
        num = imul(self.gKA, imul(w2, w))
        return float(idiv_pos(num, imul(d1, d2))[1])


# ------------------------------------------------------------ bisection ----
def _split(m, wl, wh, al, ah, lbp):
    """Bisect each box in w or a: the variable whose freezing at its midpoint narrows the enclosure less."""
    wm, am = 0.5 * (wl + wh), 0.5 * (al + ah)
    lw, hw = m.enclose(wm, wm, al, ah)              # width left with w frozen (due to a)
    la, ha = m.enclose(wl, wh, am, am)              # width left with a frozen (due to w)
    ww, wa = ha - la, hw - lw
    fw, fa = np.isfinite(ww), np.isfinite(wa)
    rel = (wh - wl) / W_MAX >= (ah - al) / np.maximum(ah, 1e-300)      # fallback: the relatively wider side
    sw = np.where(fw & fa, ww >= wa, np.where(fw == fa, rel, ~fw))
    sw |= ah <= al                                  # degenerate depth range: split w
    n = ~sw
    ch = [np.concatenate(z) for z in ((wl[sw], wm[sw], wl[n], wl[n]), (wm[sw], wh[sw], wh[n], wh[n]),
                                      (al[sw], al[sw], al[n], am[n]), (ah[sw], ah[sw], am[n], ah[n]),
                                      (lbp[sw], lbp[sw], lbp[n], lbp[n]))]
    return tuple(ch), 2 * wl.size


def refine(m, boxes, level_fn, budget, prune):
    """Bisect until each box's lower bound exceeds level_fn(box) (phase 1: 0) or the budget is spent.

    boxes = (wl, wh, al, ah, lb_parent).  Returns the evaluation count, the accepted boxes and
    bounds, the boxes left (with their bounds) and the proven violations (upper bound < 0)."""
    stack, n_eval = [boxes], 0
    acc, pend, viol = [], [], []
    a_cut = np.inf
    while stack:
        b = stack.pop()
        if b[0].size > BATCH:
            stack.append(tuple(z[BATCH:] for z in b))
            b = tuple(z[:BATCH] for z in b)
        if prune and np.isfinite(a_cut):
            keep = b[2] < a_cut
            b = tuple(z[keep] for z in b)
            if not b[0].size:
                continue
        wl, wh, al, ah, lbp = b
        lo, hi = m.enclose(wl, wh, al, ah)
        n_eval += wl.size
        lo = np.maximum(lo, lbp)
        ok = lo > level_fn(wl, wh, al, ah)
        if ok.any():
            acc.append((wl[ok], wh[ok], al[ok], ah[ok], lo[ok]))
        bad = ~ok
        if not bad.any():
            continue
        if prune:
            v = bad & (hi < 0)
            if v.any():
                viol.append((wl[v], wh[v], al[v], ah[v], hi[v]))
                a_cut = min(a_cut, float(al[v].min()))
                bad &= ~v
                if not bad.any():
                    continue
        rest = (wl[bad], wh[bad], al[bad], ah[bad], lo[bad])
        if n_eval + 3 * rest[0].size > budget:
            pend.append(rest)
            continue
        ch, ne = _split(m, *rest)
        n_eval += ne
        stack.append(ch)
    cat = lambda L: tuple(np.concatenate([x[i] for x in L]) for i in range(5)) if L else tuple(np.zeros(0) for _ in range(5))
    pend = cat(pend)
    if prune and np.isfinite(a_cut):                  # pending boxes above a violation do not matter
        keep = pend[2] < a_cut
        pend = tuple(z[keep] for z in pend)
    return n_eval, cat(acc), pend, cat(viol)


def _boxes_json(b, n=N_KEEP):
    """The n lowest-depth boxes as [f_lo_hz, f_hi_hz, a_lo_mm, a_hi_mm, bound]."""
    k = np.argsort(b[2])[:n]
    return [[float(b[0][i] / (2 * np.pi)), float(b[1][i] / (2 * np.pi)), float(b[2][i] * 1e3), float(b[3][i] * 1e3),
             float(b[4][i])] for i in k]


# ------------------------------------------------------------ one speed ----
def speed_job(job):
    """Interval verification at one speed (job: dict from the stored g_best row)."""
    P, CC, R = _setup()
    t0 = time.time()
    rpm, g = job["rpm"], job["g"]
    a0, a_circ = job["a0_mm"] * 1e-3, job["a_circ_mm"] * 1e-3
    a_v = (1.0 - MARGIN) * a_circ
    budget1, budget2, tight = job["budget1"], job["budget2"], job["tight"]
    m = Model(rpm, g)

    # grid value at a_v (circle_criterion.mu, refined) and the rigorous point bound at its frequency
    mu_g, f_g = CC.mu(rpm, a_v, g)
    mu_up = m.point(2 * np.pi * f_g, a_v)[1]

    # phase 1: positivity on [0, W_MAX] x [0, a_v]
    T = 60.0 / (rpm * P.N_TEETH)
    n0 = int(np.ceil(W_MAX * T / DTHETA0))
    e = np.linspace(0.0, W_MAX, n0 + 1)
    e[0], e[-1] = 0.0, W_MAX
    boxes = (e[:-1].copy(), e[1:].copy(), np.zeros(n0), np.full(n0, a_v), np.full(n0, NINF))
    n1, acc, pend, viol = refine(m, boxes, lambda *b: 0.0, budget1, prune=True)
    t1 = time.time() - t0
    lb1 = float(acc[4].min()) if acc[4].size else None
    a_ver = a_v
    if pend[0].size:
        a_ver = min(a_ver, float(pend[2].min()))
    if viol[0].size:
        a_ver = min(a_ver, float(viol[2].min()))
    tail = m.tail(a_v)
    tail_ok = tail is not None and tail < 1.0
    complete = not pend[0].size and not viol[0].size
    verified = bool(complete and tail_ok)

    # phase 2: certified margin (only on a complete proof)
    n2, n_left2 = 0, 0
    final = acc
    if complete and mu_g > 0:
        lvl_g = tight * mu_g

        def level(wl, wh, al, ah):
            fc = np.real(1.0 + CC.L_of(0.5 * (wl + wh), rpm, 0.5 * (al + ah), g))
            return np.minimum(lvl_g, tight * fc)
        need = acc[4] < level(*acc[:4])
        todo = tuple(z[need] for z in acc)
        done = tuple(z[~need] for z in acc)
        n2, acc2, pend2, _ = refine(m, todo, level, budget2, prune=False)
        n_left2 = int(pend2[0].size)
        final = tuple(np.concatenate([done[i], acc2[i], pend2[i]]) for i in range(5))
    if not complete:                                # bound over the proved depths [0, a_verified] only
        keep = final[2] < a_ver
        final = tuple(z[keep] for z in final)
    k = int(np.argmin(final[4])) if final[4].size else None
    min_lb = float(final[4][k]) if k is not None else None
    dw = final[1] - final[0]
    da = final[3] - final[2]

    # fail check just above a_circ
    fail = dict(applicable=not job["reached_a0"])
    if fail["applicable"]:
        a_f = min((1.0 + FAIL_REL) * a_circ, 0.5 * (a_circ + a0))
        mu_f, f_f = CC.mu(rpm, a_f, g)
        lo_f, hi_f = m.point(2 * np.pi * f_f, a_f)
        fail.update(a_mm=a_f * 1e3, rel_above_a_circ=a_f / a_circ - 1.0, f_hz=f_f, mu_grid=mu_f,
                    lower=lo_f, upper=hi_f, confirmed=bool(hi_f < 0),
                    f_stored_hz=job["f_fail_hz"], df_vs_stored_hz=f_f - job["f_fail_hz"])

    a_eff = job["a_circ_eff_mm"] * 1e-3
    rec = dict(rpm=rpm, g=g, a0_mm=a0 * 1e3, a_circ_mm=a_circ * 1e3, a_v_mm=a_v * 1e3, ratio_v=a_v / a0,
               verified=verified, complete_phase1=complete, a_verified_mm=a_ver * 1e3, ratio_verified=a_ver / a0,
               a_verified_eff_mm=min(a_ver, a_eff) * 1e3, ratio_verified_eff=min(a_ver, a_eff) / a0,
               n_boxes=int(n1 + n2), n_boxes_phase1=int(n1), n_boxes_phase2=int(n2), n_final_boxes=int(final[0].size),
               n_undecided=int(pend[0].size), undecided=_boxes_json(pend), n_violations=int(viol[0].size),
               violations=_boxes_json(viol), n_left_phase2=n_left2,
               min_lb=min_lb, min_lb_phase1=lb1,
               min_lb_box=None if k is None else dict(f_lo_hz=float(final[0][k] / (2 * np.pi)),
                                                      f_hi_hz=float(final[1][k] / (2 * np.pi)),
                                                      a_lo_mm=float(final[2][k] * 1e3), a_hi_mm=float(final[3][k] * 1e3)),
               min_box_dw_rad_s=float(dw.min()) if dw.size else None,
               min_box_da_rel=float((da / a_v).min()) if da.size else None,
               mu_grid_at_a_v=float(mu_g), f_grid_hz=float(f_g), mu_upper_at_a_v=mu_up,
               lb_over_mu_grid=None if min_lb is None or mu_g <= 0 else min_lb / mu_g,
               tail_bound=tail, tail_ok=bool(tail_ok),
               mu_lower=None if (min_lb is None or tail is None) else min(min_lb, 1.0 - tail),
               fail_check=fail,
               stored=dict(mu_at_a_circ=job["mu_at_a_circ"], mu_just_above=job["mu_just_above"],
                           f_fail_hz=job["f_fail_hz"], reached_a0=job["reached_a0"], ratio=job["ratio"],
                           first_clipped_depth_mm=job["first_clipped_depth_mm"], a_circ_eff_mm=job["a_circ_eff_mm"],
                           ratio_eff=job["ratio_eff"]),
               t_phase1_s=round(t1, 3), runtime_s=round(time.time() - t0, 3))
    return rec


# ------------------------------------------------------------ self-test ----
def selftest(n_pts=4000, n_box=2000, seed=0):
    """The enclosure against circle_criterion.L_of (float): point enclosures must contain (to 1e-9
    relative, the float's own rounding) the float value, and boxes the float values at points inside."""
    P, CC, R = _setup()
    rng = np.random.default_rng(seed)
    out = dict(cases=[])
    worst_pt, worst_box = -np.inf, -np.inf
    for rpm, g, a in ((1090.0, 1500.0, 1.25e-3), (650.0, 1500.0, 0.73e-3), (1940.0, 134.9, 6.0e-3),
                      (2970.0, 1500.0, 1.3e-3)):
        m = Model(rpm, g)
        w = rng.uniform(0.0, W_MAX, n_pts)
        aa = rng.uniform(0.0, a, n_pts)
        lo, hi = m.enclose(w, w, aa, aa)
        f = np.real(1.0 + CC.L_of(w, rpm, aa, g))
        sc = np.maximum(1.0, np.abs(f))
        ex_pt = float(np.max(np.maximum(lo - f, f - hi) / sc))
        wl = rng.uniform(0.0, W_MAX, n_box)
        wh = wl + rng.uniform(0.0, 3.0, n_box)
        al = rng.uniform(0.0, a, n_box)
        ah = np.minimum(al + rng.uniform(0.0, 0.05 * a, n_box), a)
        lo, hi = m.enclose(wl, wh, al, ah)
        ex_box = -np.inf
        for _ in range(10):
            ww = wl + rng.uniform(size=n_box) * (wh - wl)
            a2 = al + rng.uniform(size=n_box) * (ah - al)
            f = np.real(1.0 + CC.L_of(ww, rpm, a2, g))
            ex_box = max(ex_box, float(np.max(np.maximum(lo - f, f - hi) / np.maximum(1.0, np.abs(f)))))
        out["cases"].append(dict(rpm=rpm, g=g, a_max_mm=a * 1e3, point_max_excess=ex_pt, box_max_excess=ex_box))
        worst_pt, worst_box = max(worst_pt, ex_pt), max(worst_box, ex_box)
    x = rng.uniform(0.0, 1000.0, 20000)
    out["numpy_vs_libm_cos_sin_max_abs_diff"] = float(max(np.max(np.abs(np.cos(x) - np.array([math.cos(v) for v in x]))),
                                                          np.max(np.abs(np.sin(x) - np.array([math.sin(v) for v in x])))))
    out.update(point_max_excess=worst_pt, box_max_excess=worst_box,
               passed=bool(worst_pt < 1e-9 and worst_box < 1e-9 and out["numpy_vs_libm_cos_sin_max_abs_diff"] <= TRIG_PAD),
               note="excess = largest (lower bound - float value, float value - upper bound) / max(1, |value|); "
                    "negative or ~1e-16 means the float value of L_of is inside the enclosure")
    return out


# ------------------------------------------------------------- summary ----
def summarise(rows):
    rpm = np.array([r["rpm"] for r in rows])
    rv = np.array([r["ratio_verified"] for r in rows])
    re_ = np.array([r["ratio_verified_eff"] for r in rows])
    ver = np.array([r["verified"] for r in rows])
    lb = np.array([np.nan if r["min_lb"] is None else r["min_lb"] for r in rows])
    mg = np.array([r["mu_grid_at_a_v"] for r in rows])
    tl = np.array([np.inf if r["tail_bound"] is None else r["tail_bound"] for r in rows])
    nb = np.array([r["n_boxes"] for r in rows])
    rt = np.array([r["runtime_s"] for r in rows])
    fa = [r for r in rows if r["fail_check"]["applicable"]]
    fc = [r for r in fa if r["fail_check"]["confirmed"]]
    tight = np.array([np.nan if r["lb_over_mu_grid"] is None else r["lb_over_mu_grid"] for r in rows])
    i_lb = int(np.nanargmin(lb)) if np.isfinite(lb).any() else None
    return dict(
        n_speeds=len(rows), n_verified=int(ver.sum()), rpm_not_verified=[float(x) for x in rpm[~ver]],
        n_undecided_boxes=int(sum(r["n_undecided"] for r in rows)),
        n_violation_boxes=int(sum(r["n_violations"] for r in rows)),
        ratio_verified_min=float(rv.min()), rpm_ratio_verified_min=float(rpm[int(np.argmin(rv))]),
        ratio_verified_median=float(np.median(rv)), ratio_verified_max=float(rv.max()),
        ratio_verified_eff_min=float(re_.min()), ratio_verified_eff_median=float(np.median(re_)),
        n_ratio_verified_eff_below_0p9=int(np.sum(re_ < 0.9)),
        min_lb_min=None if i_lb is None else float(lb[i_lb]), rpm_min_lb_min=None if i_lb is None else float(rpm[i_lb]),
        min_lb_median=float(np.nanmedian(lb)) if np.isfinite(lb).any() else None,
        lb_over_mu_grid_min=float(np.nanmin(tight)) if np.isfinite(tight).any() else None,
        lb_le_mu_grid_all=bool(np.all(~np.isfinite(lb) | (lb <= mg))),
        lb_le_mu_upper_all=bool(all(r["min_lb"] is None or r["min_lb"] <= r["mu_upper_at_a_v"] for r in rows)),
        tail_bound_max=float(tl.max()), mu_lower_min=float(min(r["mu_lower"] for r in rows if r["mu_lower"] is not None))
        if any(r["mu_lower"] is not None for r in rows) else None,
        fail_check_applicable=len(fa), fail_check_confirmed=len(fc),
        rpm_fail_not_confirmed=[r["rpm"] for r in fa if not r["fail_check"]["confirmed"]],
        fail_rel_max=float(max(r["fail_check"]["rel_above_a_circ"] for r in fa)) if fa else None,
        fail_df_vs_stored_max_abs_hz=float(max(abs(r["fail_check"]["df_vs_stored_hz"]) for r in fa)) if fa else None,
        n_boxes_total=int(nb.sum()), n_boxes_median=float(np.median(nb)), n_boxes_max=int(nb.max()),
        rpm_n_boxes_max=float(rpm[int(np.argmax(nb))]),
        runtime_total_s=float(rt.sum()), runtime_median_s=float(np.median(rt)), runtime_max_s=float(rt.max()))


def _g(x, fmt):
    return "None" if x is None else format(x, fmt)


def progress_line(r):
    fc = r["fail_check"]
    if not fc["applicable"]:
        fs = "n/a (reached a_lim(0))"
    else:
        fs = "%s at +%.1e (upper %.3g)" % ("confirmed" if fc["confirmed"] else "NOT confirmed", fc["rel_above_a_circ"],
                                          fc["upper"])
    return (f"{r['rpm']:6.0f} rpm g {r['g']:7.1f}: verified {r['verified']}  a_ver/a0 {r['ratio_verified']:.4f}"
            f"  boxes {r['n_boxes']} (undecided {r['n_undecided']}, violating {r['n_violations']})"
            f"  min_lb {_g(r['min_lb'], '.4g')} (mu_grid {r['mu_grid_at_a_v']:.4g}, upper {r['mu_upper_at_a_v']:.4g})"
            f"  tail {_g(r['tail_bound'], '.2e')}  fail check {fs}  [{r['runtime_s']:.2f} s]")


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--speeds", default="", help="comma-separated subset of the 251 speeds (default: all)")
    ap.add_argument("--smoke", action="store_true", help="write circle_interval_smoke.json / _smoke_cache.jsonl")
    ap.add_argument("--out", default="", help="output json (default circle_interval.json; cache <out>_cache.jsonl)")
    ap.add_argument("--budget1", type=int, default=BUDGET1)
    ap.add_argument("--budget2", type=int, default=BUDGET2)
    ap.add_argument("--tight", type=float, default=TIGHT)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "circle_interval_smoke.json" if a.smoke else "circle_interval.json")
    cache_path = os.path.splitext(out)[0] + "_cache.jsonl"
    proto = proto_hash(a.budget1, a.budget2, a.tight)
    t0 = time.time()

    with open(STORED) as fh:
        stored = json.load(fh)
    rows_st = stored["g_best"]["rows"]
    sel = {float(x) for x in a.speeds.split(",") if x.strip()} if a.speeds else None
    jobs = []
    for q in rows_st:
        if sel is not None and q["rpm"] not in sel:
            continue
        jobs.append(dict(rpm=q["rpm"], g=q["g"], a0_mm=q["a0_mm"], a_circ_mm=q["a_circ_mm"], ratio=q["ratio"],
                         reached_a0=q["reached_a0"], f_fail_hz=q["f_fail_hz"], mu_at_a_circ=q["mu_at_a_circ"],
                         mu_just_above=q["mu_just_above"], first_clipped_depth_mm=q["first_clipped_depth_mm"],
                         a_circ_eff_mm=q["a_circ_eff_mm"], ratio_eff=q["ratio_eff"],
                         budget1=a.budget1, budget2=a.budget2, tight=a.tight))
    if sel is not None and len(jobs) != len(sel):
        raise SystemExit(f"speeds not in circle_criterion.json g_best rows: {sorted(sel - {j['rpm'] for j in jobs})}")

    st = selftest()
    print("selftest: point excess %.2e, box excess %.2e, numpy vs libm cos/sin %.1e -> %s" % (
        st["point_max_excess"], st["box_max_excess"], st["numpy_vs_libm_cos_sin_max_abs_diff"],
        "passed" if st["passed"] else "FAILED"), flush=True)
    if not st["passed"]:
        raise SystemExit("self-test failed: the enclosure does not match circle_criterion.L_of")

    cache = JL(cache_path)
    key = lambda j: [proto, j["rpm"], j["g"], j["a_circ_mm"]]
    todo = [j for j in jobs if cache.get(key(j)) is None]
    todo.sort(key=lambda j: j["rpm"])
    print(f"{len(jobs)} speeds, {len(jobs) - len(todo)} in the cache ({os.path.basename(cache_path)}, protocol {proto}), "
          f"{len(todo)} to run on {a.workers} worker(s)", flush=True)
    if todo:
        with Pool(a.workers, initializer=_init_worker) as pool:
            for r in pool.imap_unordered(speed_job, todo):
                cache.put(key(r), r)
                print(progress_line(r), flush=True)

    rows = sorted((cache.get(key(j)) for j in jobs), key=lambda r: r["rpm"])
    P, CC, R = _setup()
    res = dict(
        note=("Rigorous check (interval arithmetic, outward rounding) of the frequency condition of Proposition 1, "
              "inf_w Re[1 + L(i w; a)] > 0, on the continuous frequency axis for every depth a in "
              "[0, (1 - 1e-3) a_circ] at the scheduled gain g_best (circle_criterion.json), averaged zero-order "
              "model with a linear force law.  'verified' also proves that no characteristic root of the uncontrolled "
              "cut with the actuator reaches the imaginary axis on that depth range (L stable, a_v < a_lim(0)).  The "
              "hypothesis |b(t)| <= Vmax is not checked here (stored first_clipped_depth_mm / a_circ_eff_mm)."
              + ("" if sel is None else "  SUBSET RUN: %d of 251 speeds." % len(jobs))),
        protocol=dict(
            hash=proto, version=VERSION, source="verification/paper/circle_criterion.json g_best rows",
            L="g s Ga(s) G(s) / (1 - KAPPA a (1 - exp(-s T)) G(s)) (circle_criterion.L_of)",
            form="Re[1+L] = N/|B|^2, N = |B|^2 + Re(c conj B) = |B + c/2|^2 - |c|^2/4, c = -i g K_a w^3/A, "
                 "A = a0 - w^2 + i a1 w, B = K - M w^2 + i C w - KAPPA a (1 - cos wT + i sin wT)",
            w_max_rad_s=W_MAX, margin_rel=MARGIN, fail_rel=FAIL_REL, rel_par=REL_PAR, trig_pad=TRIG_PAD,
            j_slack=J_SLACK, dtheta0_rad=DTHETA0, budget1=a.budget1, budget2=a.budget2, tight=a.tight,
            zeta=ZETA, constants=dict(M=P.M, C=P.C, K=P.K, K_a=P.GA_GAIN, a1=P.GA_A1, a0=P.GA_A0, KAPPA=R.KAPPA,
                                      N_teeth=P.N_TEETH),
            tail="|L| <= g K_a w^3 / ((w^2 - a0)(M w^2 - K - 2 |KAPPA| a_v)) for w >= w_max, decreasing in w",
            split="bisect w or a, whichever leaves the wider enclosure with the other frozen at its midpoint",
            numpy=np.__version__, libc=" ".join(platform.libc_ver()), python=platform.python_version()),
        selftest=st,
        rows=rows,
        summary=summarise(rows),
        runtime_s=round(time.time() - t0, 2))
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    os.replace(out + ".tmp", out)
    s = res["summary"]
    print(json.dumps(s, indent=1, default=float), flush=True)
    print(f"wrote {out} ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
