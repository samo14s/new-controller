"""ctypes wrapper around tdsim.c (compiled on first import)."""
import ctypes
import os
import subprocess
import numpy as np
import params as P
from structure import state_space

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "tdsim.c")
_LIB = os.path.join(_HERE, "_tdsim.so")

if (not os.path.exists(_LIB)) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
    # build beside and rename, so running processes keep their mapped copy
    _tmp = f"{_LIB}.{os.getpid()}.tmp"
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                           "-o", _tmp, _SRC, "-lm"])
    os.replace(_tmp, _LIB)
_lib = ctypes.CDLL(_LIB)
_d = ctypes.c_double
_i = ctypes.c_int
_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_lib.simulate.argtypes = [_d] * 3 + [_d] * 3 + [_d] * 3 + [_d] * 4 + [_i, _d, _d, _d] \
    + [_i] * 4 + [_d, _i, _i] + [_p] * 6 + [_p]
_lib.simulate.restype = _i

PHI_ST, PHI_EX = P.immersion_angles()


FMODE = 1          # 1: force re-evaluated at every RK4 stage; 0: held per step


def simulate(rpm, a, g, vmax=0.0, store=False, nrev=P.N_REVS, nsub=2,
             spr=P.STEPS_PER_REV, nlay=P.N_LAYERS, fz=P.FZ, fmode=None,
             ramp_a=0.0, ramp_f=0.0, f0=0.0, edge=False,
             kick_step=-1, kick_dv=0.0, clip_on=0, nreg=1):
    """Run one cut.  Returns a dict with the peak demand and, if store, histories.

    vmax > 0 limits the applied command |V| (clipped DVF); 0 means no limit.
    fmode selects how the cutting force enters the integrator (see tdsim.c).
    Entry (default: full depth and feed at t = 0):
      ramp_a  revolutions over which the axial depth rises from 0 to a
              (idealised: the depth is uniform around the arc and the delay
              stays T; a real ramp-in descends axially, which shifts the
              delay by a tan(beta) / (2 pi R ramp_a) of T; not used in C1)
      ramp_f  revolutions over which the feed rises from f0*fz to fz
      edge    straight entry through a workpiece face normal to the feed
              (full radial engagement after R/fz teeth, 40 rev here)
    Perturbation: kick_dv (m/s) added to x' at step kick_step (< 0: none);
    clip_on: first step at which the command limit vmax applies.
    nreg: passes the chip remembers (1: x(t - T) only; > 1: least chip over
    the last nreg passes, so the tool leaving the cut is handled; max 8).
    """
    assert spr % P.N_TEETH == 0, "the tooth delay must be a whole number of steps"
    fmode = FMODE if fmode is None else fmode
    nsteps = spr * nrev
    nbuf = nsteps + 1 if store else 1
    bufs = [np.zeros(nbuf) for _ in range(5)]
    summ = np.zeros(5)
    opt = np.array([ramp_a, ramp_f, f0, 1.0 if edge else 0.0,
                    kick_step, kick_dv, clip_on, nreg], dtype=np.float64)
    _lib.simulate(float(rpm), float(a), float(g),
                  P.M, P.C, P.K, P.GA_GAIN, P.GA_A1, P.GA_A0,
                  P.K_TC, P.K_RC, fz, P.D_TOOL, P.N_TEETH, P.HELIX,
                  PHI_ST, PHI_EX, spr, nrev, nlay, nsub, float(vmax), int(fmode),
                  int(store), *bufs, summ, opt)
    out = dict(max_demand=summ[0], max_fact=summ[1], max_x=summ[2],
               max_v=summ[3], diverged=bool(summ[4]))
    if store:
        dt = 60.0 / (rpm * spr)
        out.update(t=np.arange(nsteps + 1) * dt, x=bufs[0], v=bufs[1],
                   fact=bufs[2], fcut=bufs[3], vcmd=bufs[4], spr=spr)
    return out


def entry_revs(ramp_a=0.0, ramp_f=0.0, f0=0.0, edge=False, fz=P.FZ, **_):
    """Revolutions until the entry is complete (depth, feed and width at nominal).

    The face is fully engaged when the tool centre has advanced R, with the
    advance integrated over the feed ramp (same kinematics as tdsim.c).
    The last fresh sliver is cut within one tooth period after that.
    """
    t_edge = 0.0
    if edge:
        R, u = 0.5 * P.D_TOOL, fz * P.N_TEETH          # advance per revolution
        if ramp_f > 0:
            s_f = u * ramp_f * (1 + f0) / 2               # advance at the end of the ramp
            if R <= s_f:
                qa, qb = (1 - f0) / (2 * ramp_f), f0      # u (qb t + qa t^2) = R
                t_edge = R / u / qb if qa == 0 else (-qb + np.sqrt(qb * qb + 4 * qa * R / u)) / (2 * qa)
            else:
                t_edge = ramp_f + (R - s_f) / u
        else:
            t_edge = R / u
    return float(max(ramp_a, ramp_f, t_edge))


def simulate_exact_python(rpm, a, g, nrev=P.N_REVS, spr=P.STEPS_PER_REV,
                          nlay=P.N_LAYERS):
    """Slow reference: exact ZOH discretisation of the linear loop (no limit)."""
    from scipy.linalg import expm
    A, B = state_space(g)
    dt = 60.0 / (rpm * spr)
    Mx = np.zeros((5, 5))
    Mx[:4, :4] = A * dt
    Mx[:4, 4] = B * dt
    E = expm(Mx)
    Phi, Gam = E[:4, :4], E[:4, 4]
    N, R = P.N_TEETH, P.D_TOOL / 2
    dz = a / nlay
    zj = (np.arange(nlay) + 0.5) * dz
    nd = spr // N
    nsteps = spr * nrev
    xh = np.zeros(nsteps + 1)
    v = np.zeros(nsteps + 1)
    z = np.zeros(4)
    for n in range(nsteps):
        th = 2 * np.pi * (n % spr) / spr
        phi = (th + np.arange(N)[:, None] * 2 * np.pi / N
               - zj[None, :] * np.tan(P.HELIX) / R) % (2 * np.pi)
        inside = (phi >= PHI_ST) & (phi <= PHI_EX)
        dx = z[0] - (xh[n - nd] if n >= nd else 0.0)
        h = P.FZ * np.sin(phi) - dx * np.cos(phi)
        h = np.where(inside & (h > 0), h, 0.0)
        f = -np.sum((P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * dz * h)
        v[n] = z[1]
        z = Phi @ z + Gam * f
        xh[n + 1] = z[0]
    v[nsteps] = z[1]
    return xh, v
