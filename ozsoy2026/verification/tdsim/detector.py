"""Chatter detector for tdsim runs, based on once-per-tooth-period sampling.

The discrete simulation is exactly periodic with the tooth period (spr/N steps:
tooth j at step s+nd sits where tooth j+1 sat at step s, helix offsets
included).  In a stable cut the forced response is therefore a fixed point of
the once-per-tooth map, and e_k = x(t_k) - x(t_{k-1}), t_k = k T_tooth, decays
geometrically to round-off.  Chatter (Hopf or flip) makes e_k grow until the
tool leaves the cut and the amplitude saturates.

Two decisions are returned on the K = nrev*N samples (see classify()):
  D1 'stable_decay'  - e_k must die out (fitted growth factor < 1 - 2e-5 per
                       tooth period, shrinking envelope); any sustained limit
                       cycle is chatter.
  D2 'stable_growth' - e_k must not grow beyond the engagement transient.
D1 is the default ('stable').  Down milling ends each cut with a zero static
chip, so the h <= 0 switch acts at any vibration amplitude near the exit and
saturates chatter just above the linear limit at small amplitude (a
supercritical Hopf).  D2 then reports a boundary a few % too high; D1 does
not, which check3 demonstrates.
"""
import numpy as np
import params as P
import tdsim


def tooth_samples(x, spr, N=P.N_TEETH):
    nd = spr // N
    return x[::nd]


def classify(x, spr, diverged=False, N=P.N_TEETH, delta=2e-5, nblocks=25):
    """Two decisions on the once-per-tooth differences e_k (see module doc).

    stable_decay  (criterion D1, default 'stable'): the deviation from the
        tooth-periodic steady state dies out, i.e. the growth factor per tooth
        period fitted to log(block RMS of e_k) over the second half of the run
        is below 1 - delta, the largest block RMS of the last 5 blocks is below
        0.98 x that of the first 5 tail blocks, the end RMS is below the
        engagement transient, and max |x(t) - x(t-T)| over the last 20 % is
        below 0.25 fz (or e_k reached round-off).  The last test catches
        saturated, beating limit cycles that are already reached during the
        engagement transient (deep cuts).  A sustained limit cycle
        of any amplitude counts as chatter.  Its boundary is the Hopf/flip point
        of the simulated system (the linear stability limit), within about
        delta / (d mu / d a) -- a few 0.01 % here.
    stable_growth (criterion D2): the vibration does not grow beyond the
        engagement transient (mu < 1 between the 40-60 % and 80-100 % windows
        and RMS at the end below the first 10 %).  This mimics an amplitude-
        growth judgement; it accepts small saturated limit cycles.
    """
    xs = tooth_samples(x, spr, N)
    e = np.diff(xs)
    K = len(e)
    if diverged or not np.all(np.isfinite(e)):
        return dict(stable=False, stable_decay=False, stable_growth=False,
                    mu=np.inf, mu_fit=np.inf, q=np.inf, rms_first=np.nan, rms_end=np.inf)

    def rms(lo, hi):
        seg = e[int(lo * K):int(hi * K)]
        return float(np.sqrt(np.mean(seg ** 2)))

    r_first, r_mid, r_end = rms(0.0, 0.1), rms(0.4, 0.6), rms(0.8, 1.0)
    mu = (r_end / r_mid) ** (1.0 / (0.4 * K)) if r_mid > 0 else 0.0
    tail = e[K // 2:]
    L = len(tail) // nblocks
    br = np.sqrt(np.mean(tail[:nblocks * L].reshape(nblocks, L) ** 2, axis=1))
    floor = (r_end < 1e-7 * r_first) or bool(np.any(br <= 1e-9 * r_first))
    if floor:
        mu_fit, q = 0.0, 0.0
    else:
        slope = np.polyfit(np.arange(nblocks) * L, np.log(br), 1)[0]
        mu_fit = float(np.exp(slope))
        q = float(np.max(br[-5:]) / np.max(br[:5]))   # envelope ratio across the tail
    # D1 requires a fitted decay AND an envelope that really shrinks (q < 0.98)
    # AND no growth beyond the engagement transient: a saturated, slowly
    # beating limit cycle can give a fitted slope just below zero.
    # regenerative displacement over the last 20 % of the run, relative to fz:
    # a decaying cut ends far below the chip thickness, a saturated chatter
    # cycle sits at a sizeable fraction of it (that is what saturates it).
    nd = spr // N
    n0 = int(0.8 * (len(x) - 1))
    dx_end = float(np.max(np.abs(x[n0 + nd:] - x[n0:-nd]))) / P.FZ
    stable_decay = bool(floor or (mu_fit < 1.0 - delta and q < 0.98 and r_end < r_first
                                  and dx_end < 0.25))
    stable_growth = bool(floor or ((mu < 1.0) and (r_end < r_first)))
    return dict(stable=stable_decay, stable_decay=stable_decay,
                stable_growth=stable_growth, mu=float(mu), mu_fit=float(mu_fit), q=q,
                dx_end_over_fz=dx_end,
                rms_first=r_first, rms_mid=r_mid, rms_end=r_end)


def run_and_classify(rpm, a, g, nrev, spr=P.STEPS_PER_REV, vmax=0.0):
    r = tdsim.simulate(rpm, a, g, vmax=vmax, store=True, nrev=nrev, spr=spr)
    c = classify(r["x"], spr, r["diverged"])
    c["max_x_um"] = r["max_x"] * 1e6
    c["max_demand"] = r["max_demand"]
    c["max_fact"] = r["max_fact"]
    c["run"] = r
    return c


def chatter_frequency(x, dt, frac=0.5):
    """Dominant non-tooth-harmonic frequency in the last part of x (Hz)."""
    n0 = int(len(x) * (1 - frac))
    y = x[n0:] - np.mean(x[n0:])
    w = np.hanning(len(y))
    S = np.abs(np.fft.rfft(y * w))
    f = np.fft.rfftfreq(len(y), dt)
    return f, S


def boundary(rpm, g, nrev, spr=P.STEPS_PER_REV, a_guess=None, rtol=2e-3, vmax=0.0,
             a_min=0.05e-3, a_max=80e-3, criterion="stable"):
    """Bisection on the axial depth for the time-domain stability limit (m)."""
    a_guess = a_guess if a_guess is not None and np.isfinite(a_guess) else 2e-3
    lo, hi = None, None
    a = a_guess
    # bracket
    for _ in range(40):
        st = run_and_classify(rpm, a, g, nrev, spr, vmax)[criterion]
        if st:
            lo = a
            if hi is not None:
                break
            a = min(a * 1.5, a_max)
            if lo >= a_max:
                return a_max, np.nan
        else:
            hi = a
            if lo is not None:
                break
            a = max(a / 1.5, a_min)
            if hi <= a_min:
                return 0.0, a_min
    while (hi - lo) / lo > rtol:
        mid = 0.5 * (lo + hi)
        if run_and_classify(rpm, mid, g, nrev, spr, vmax)[criterion]:
            lo = mid
        else:
            hi = mid
    return lo, hi
