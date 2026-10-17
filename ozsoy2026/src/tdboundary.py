"""Time-domain stability classification and boundary search.

The simulated cut is exactly periodic in the tooth period (spr/N steps), so in
a stable cut the once-per-tooth samples x(t_k) converge to a fixed point and
e_k = x(t_k) - x(t_{k-1}) decays to round-off; chatter makes e_k grow until
the tool leaves the cut and then saturates.  A cut is called stable when
(criterion D1 of verification/tdsim/detector.py, adopted here):
  * the per-tooth growth factor fitted to log block-RMS(e_k) over the second
    half of the run is below 1 - 2e-5,
  * the envelope really shrinks across that half (last/first block max < 0.98),
  * the end RMS is below the engagement transient, and
  * the regenerative displacement over the last 20 % stays below 0.25 fz
    (catches saturated limit cycles already reached during engagement),
or when e_k has reached round-off.  Any sustained limit cycle is chatter.
"""
import numpy as np
import params as P
from tdsim import simulate

NREV = 300


def classify_run(x, spr, diverged=False, N=P.N_TEETH, delta=2e-5, nblocks=25):
    nd = spr // N
    e = np.diff(x[::nd])
    K = len(e)
    if diverged or not np.all(np.isfinite(e)):
        return dict(stable=False, mu=np.inf, dx_end_over_fz=np.inf)

    def rms(lo, hi):
        seg = e[int(lo * K):int(hi * K)]
        return float(np.sqrt(np.mean(seg ** 2)))

    r_first, r_end = rms(0.0, 0.1), rms(0.8, 1.0)
    tail = e[K // 2:]
    L = len(tail) // nblocks
    br = np.sqrt(np.mean(tail[:nblocks * L].reshape(nblocks, L) ** 2, axis=1))
    floor = (r_end < 1e-7 * r_first) or bool(np.any(br <= 1e-9 * r_first))
    if floor:
        mu, q = 0.0, 0.0
    else:
        mu = float(np.exp(np.polyfit(np.arange(nblocks) * L, np.log(br), 1)[0]))
        q = float(np.max(br[-5:]) / np.max(br[:5]))
    n0 = int(0.8 * (len(x) - 1))
    dx_end = float(np.max(np.abs(x[n0 + nd:] - x[n0:-nd]))) / P.FZ
    stable = bool(floor or (mu < 1.0 - delta and q < 0.98 and r_end < r_first
                            and dx_end < 0.25))
    return dict(stable=stable, mu=mu, q=q, dx_end_over_fz=dx_end)


def classify(rpm, a, g, vmax=0.0, nrev=NREV, spr=P.STEPS_PER_REV):
    r = simulate(rpm, a, g, vmax=vmax, store=True, nrev=nrev, spr=spr)
    c = classify_run(r["x"], spr, r["diverged"])
    c["chatter"] = not c["stable"]
    c["run"] = r
    return c


def boundary(rpm, g, vmax=0.0, a_lo=0.2e-3, a_hi=40e-3, growth=1.25, rtol=3e-3,
             nrev=NREV):
    """Largest stable depth (m): upward scan from a shallow cut, then bisection.

    Returns a_hi if no chatter is found up to a_hi.
    """
    lo = 0.0
    a = a_lo
    while a <= a_hi:
        if classify(rpm, a, g, vmax, nrev)["stable"]:
            lo, a = a, a * growth
        else:
            break
    else:
        return a_hi
    hi = a
    if lo == 0.0:
        lo = 0.0
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if classify(rpm, mid, g, vmax, nrev)["stable"]:
            lo = mid
        else:
            hi = mid
    return lo
