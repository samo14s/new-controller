"""A well-posed actuator-feasibility condition (proposed replacement for Eq. 2).

Why the paper's check is ill-posed.  Eq. (2) simulates the cut at exactly
a_lim, where the chatter mode is marginal: it neither grows nor decays, so the
peak demand is set by how hard tool entry excites it.  The demand then jumps
around in g, the feasible set fragments, and a gain that saturates the
actuator only for a moment at entry is rejected even though the cut is stable
(the clipped-DVF reversal of Table 3).

The condition here asks what an operator needs from a per-speed gain: once
the cut has settled, the actuator must not saturate at ANY depth the gain
permits, i.e. for every a in (0, kappa * a_lim(Omega, g)], with kappa < 1 a
stability margin.  Below a_lim the chatter mode decays, so the settled motion
is the tooth-periodic forced response, and that response has a closed form:

* at the tooth-passing harmonics w_m = m N Omega the regenerative factor
  1 - e^{-i w_m T} vanishes, so the forced response is Gcl(i w_m) F_m;
* the static chip-load force of one axial slice, w(theta), is the same for
  every slice up to the helix lag z tan(beta)/R, so integrating over the depth
  gives  F_m(a) = W_m (1 - e^{-i m N a tan(beta)/R}) / (i m N tan(beta)/R).
  The factor vanishes when a is a multiple of the axial pitch / m: the helix
  cancellation that makes the demand at a single depth jump around.

The demand max_t K_a g |xdot(t)| over all depths up to kappa*a_lim increases
with g in practice (checked in the tests), so the feasible set is one
interval [0, g_b] and the optimum is its upper edge.  The entry transient is
left to a separate time-domain certificate with the real, clipped actuator
(certify()).
"""
import numpy as np
import params as P
from structure import Gcl
from zoa import a_lim

KAPPA = 0.9         # operating margin below the stability limit
GUARD = 1.0         # fraction of F_max allowed (the paper suggests 0.8-0.9 as a guard band)
N_HARM = 40         # tooth-passing harmonics kept
N_THETA = 4096      # samples of one tooth period for the slice force
N_T = 256           # time samples per tooth period for the peak search
N_DEPTH = 80        # depths sampled in (0, kappa * a_lim]

_PHI_ST, _PHI_EX = P.immersion_angles()


def slice_force_harmonics(n_theta=N_THETA, n_harm=N_HARM):
    """W_m: harmonics of the static force per unit depth (N/m), m = 0..n_harm.

    w(theta) sums the N teeth of one axial slice; it repeats every tooth
    period, so W_m are its Fourier coefficients over that period.
    """
    pitch = 2 * np.pi / P.N_TEETH
    th = np.arange(n_theta) / n_theta * pitch
    w = np.zeros(n_theta)
    for i in range(P.N_TEETH):
        phi = np.mod(th + i * pitch, 2 * np.pi)
        cut = (phi >= _PHI_ST) & (phi <= _PHI_EX)
        h = P.FZ * np.sin(phi)
        w += np.where(cut & (h > 0), -(P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * h, 0.0)
    W = np.fft.rfft(w) / n_theta
    return W[: n_harm + 1]


_W = slice_force_harmonics()


def force_harmonics(depths, n_harm=N_HARM):
    """F_m(a) for an array of depths: shape (len(depths), n_harm + 1), N."""
    depths = np.atleast_1d(depths)[:, None]
    m = np.arange(n_harm + 1)[None, :]
    q = m * P.N_TEETH * np.tan(P.HELIX) / (0.5 * P.D_TOOL)      # rad per metre of depth
    with np.errstate(invalid="ignore", divide="ignore"):
        integ = np.where(m == 0, depths, (1 - np.exp(-1j * q * depths)) / (1j * q))
    return _W[None, : n_harm + 1] * integ


def forced_velocity(rpm, g, depths, n_harm=N_HARM, n_t=N_T):
    """Settled workpiece velocity (m/s) over one tooth period, shape (len(depths), n_t)."""
    wT = 2 * np.pi * P.N_TEETH * rpm / 60.0
    m = np.arange(1, n_harm + 1)
    H = 1j * m * wT * Gcl(1j * m * wT, g)                      # velocity per force
    F = force_harmonics(depths, n_harm)[:, 1:]                  # (nd, nh)
    t = np.arange(n_t) / n_t                                    # fraction of a tooth period
    E = np.exp(2j * np.pi * np.outer(m, t))                     # (nh, nt)
    return 2 * np.real((F * H[None, :]) @ E)                    # (nd, nt)


def forced_demand(rpm, g, depths, n_harm=N_HARM, n_t=N_T):
    """Settled peak demand K_a g max|xdot| (N) at each depth, from the forced response."""
    if g == 0:
        return np.zeros(np.size(depths))
    v = forced_velocity(rpm, g, depths, n_harm, n_t)
    return P.GA_GAIN * g * np.abs(v).max(axis=1)


def demand_envelope(rpm, g, kappa=KAPPA, n_depth=N_DEPTH):
    """(a_lim, worst settled demand over all depths up to kappa * a_lim)."""
    al = a_lim(rpm, g)
    if not np.isfinite(al) or g == 0:
        return al, 0.0
    depths = np.linspace(kappa * al / n_depth, kappa * al, n_depth)
    return al, float(forced_demand(rpm, g, depths).max())


def fitness(rpm, g, kappa=KAPPA, guard=GUARD, fmax=P.F_MAX, detail=False):
    """a_lim in mm if the settled demand stays below guard*F_max at every depth
    up to kappa*a_lim, else 0.  Same shape as objective.fitness."""
    g = float(np.atleast_1d(g)[0])
    al, d = demand_envelope(rpm, g, kappa)
    ok = np.isfinite(al) and d <= guard * fmax
    f = al * 1e3 if ok else 0.0
    return (f, al, d) if detail else f


def edge_gain(rpm, g_hi=3000.0, kappa=KAPPA, guard=GUARD, tol=0.25):
    """Upper edge of the feasible interval by bisection (the optimum)."""
    if fitness(rpm, g_hi, kappa, guard) > 0:
        return g_hi
    lo, hi = 0.0, g_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fitness(rpm, mid, kappa, guard) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def certify(rpm, g, kappa=KAPPA, nrev=300, tail=20):
    """Time-domain certificate at the operating depth kappa*a_lim, from rest.

    The cut is simulated with the real actuator limit (|V| <= F_max/K_a), so
    the entry transient may saturate it.  Returns whether the cut settles
    (D1 detector), the settled peak actuator force, and the closed-form
    estimate at the same depth for comparison.
    """
    from tdboundary import classify
    al = a_lim(rpm, g)
    a = kappa * al
    c = classify(rpm, a, g, vmax=P.F_MAX / P.GA_GAIN, nrev=nrev)
    r = c["run"]
    spr = r["spr"]
    settled = float(np.abs(r["fact"][(nrev - tail) * spr:]).max())
    demand_settled = float(P.GA_GAIN * g * np.abs(r["v"][(nrev - tail) * spr:]).max())
    return dict(rpm=rpm, g=g, a_mm=a * 1e3, stable=bool(c["stable"]),
                fact_settled=settled, demand_settled_td=demand_settled,
                demand_closed_form=float(forced_demand(rpm, g, [a])[0]),
                entry_saturated=bool(r["max_v"] >= P.F_MAX / P.GA_GAIN - 1e-9))
