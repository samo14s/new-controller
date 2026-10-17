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
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                           "-o", _LIB, _SRC, "-lm"])
_lib = ctypes.CDLL(_LIB)
_d = ctypes.c_double
_i = ctypes.c_int
_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_lib.simulate.argtypes = [_d] * 3 + [_d] * 3 + [_d] * 3 + [_d] * 4 + [_i, _d, _d, _d] \
    + [_i] * 4 + [_d, _i] + [_p] * 6
_lib.simulate.restype = _i

PHI_ST, PHI_EX = P.immersion_angles()


def simulate(rpm, a, g, vmax=0.0, store=False, nrev=P.N_REVS, nsub=2,
             spr=P.STEPS_PER_REV, nlay=P.N_LAYERS, fz=P.FZ):
    """Run one cut.  Returns a dict with the peak demand and, if store, histories.

    vmax > 0 limits the applied command |V| (clipped DVF); 0 means no limit.
    """
    nsteps = spr * nrev
    nbuf = nsteps + 1 if store else 1
    bufs = [np.zeros(nbuf) for _ in range(5)]
    summ = np.zeros(5)
    _lib.simulate(float(rpm), float(a), float(g),
                  P.M, P.C, P.K, P.GA_GAIN, P.GA_A1, P.GA_A0,
                  P.K_TC, P.K_RC, fz, P.D_TOOL, P.N_TEETH, P.HELIX,
                  PHI_ST, PHI_EX, spr, nrev, nlay, nsub, float(vmax),
                  int(store), *bufs, summ)
    out = dict(max_demand=summ[0], max_fact=summ[1], max_x=summ[2],
               max_v=summ[3], diverged=bool(summ[4]))
    if store:
        dt = 60.0 / (rpm * spr)
        out.update(t=np.arange(nsteps + 1) * dt, x=bufs[0], v=bufs[1],
                   fact=bufs[2], fcut=bufs[3], vcmd=bufs[4], spr=spr)
    return out


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
