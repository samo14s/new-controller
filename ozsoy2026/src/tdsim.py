"""ctypes wrapper around tdsim.c (compiled on first import)."""
import ctypes
import os
import subprocess
import numpy as np
import params as P
from structure import state_space

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "tdsim.c")
_d = ctypes.c_double
_i = ctypes.c_int
_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def _load(name, comb):
    """Build (if stale) and load one variant: comb = 0 is the plain-DVF build,
    comb = 1 adds the comb terms of the command (see tdsim.c)."""
    lib = os.path.join(_HERE, name)
    if (not os.path.exists(lib)) or os.path.getmtime(lib) < os.path.getmtime(_SRC):
        # build beside and rename, so running processes keep their mapped copy
        tmp = f"{lib}.{os.getpid()}.tmp"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", f"-DCOMB={comb}",
                               "-o", tmp, _SRC, "-lm"])
        os.replace(tmp, lib)
    so = ctypes.CDLL(lib)
    so.simulate.argtypes = [_d] * 3 + [_d] * 3 + [_d] * 3 + [_d] * 4 + [_i, _d, _d, _d] \
        + [_i] * 4 + [_d, _i, _i] + [_p] * 6 + [_p] + [ctypes.c_void_p]
    so.simulate.restype = _i
    return so


_lib = _load("_tdsim.so", 0)
_lib_comb = _load("_tdsim_comb.so", 1)
# provenance: hash of the source the loaded libraries were built from (taken
# at import, so a later edit of tdsim.c does not relabel results of this run)
import hashlib
SOURCE_SHA = hashlib.sha256(open(_SRC, "rb").read()).hexdigest()

PHI_ST, PHI_EX = P.immersion_angles()


FMODE = 1          # 1: force re-evaluated at every RK4 stage; 0: held per step


def simulate(rpm, a, g, vmax=0.0, store=False, nrev=P.N_REVS, nsub=2,
             spr=P.STEPS_PER_REV, nlay=P.N_LAYERS, fz=P.FZ, fmode=None,
             ramp_a=0.0, ramp_f=0.0, f0=0.0, edge=False,
             kick_step=-1, kick_dv=0.0, clip_on=0, nreg=1, pert=None, g2=0.0, kd=0.0,
             comb_delay=1.0, runout=0.0, runout_phase=0.0):
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
    pert: full-state deviation added at kick_step, a dict with x, v, q, qd
    and hx, hv (x and x' at 0, 1, ..., nd steps back, nd = spr / N).
    g2, kd: comb terms of the command (C4): V = sat(-g x' - g2 (x' - x'(t - T))
    - kd (x - x(t - T)), vmax), which vanish on a tooth-periodic motion
    (kd < 0 opposes the regenerative force).  comb_delay: the comb's delay in
    tooth periods (1: exact; N: spindle-period comb).  runout (m),
    runout_phase (rad): eccentric radial runout of the teeth (needs nreg > 1).
    All of these run in the COMB build, so the default path is untouched.
    """
    if runout != 0.0:
        assert nreg > 1, "runout is modelled in the multiple-regeneration chip only"
    assert spr % P.N_TEETH == 0, "the tooth delay must be a whole number of steps"
    fmode = FMODE if fmode is None else fmode
    nsteps = spr * nrev
    nbuf = nsteps + 1 if store else 1
    bufs = [np.zeros(nbuf) for _ in range(5)]
    summ = np.zeros(5)
    opt = np.array([ramp_a, ramp_f, f0, 1.0 if edge else 0.0,
                    kick_step, kick_dv, clip_on, nreg, g2, kd, comb_delay, runout, runout_phase],
                   dtype=np.float64)
    pbuf = None
    if pert is not None:
        nd = spr // P.N_TEETH
        hx, hv = np.asarray(pert["hx"], float), np.asarray(pert["hv"], float)
        assert len(hx) == len(hv) == nd + 1
        pbuf = np.concatenate([[pert["x"], pert["v"], pert["q"], pert["qd"]], hx, hv]).astype(np.float64)
    comb = g2 != 0.0 or kd != 0.0 or comb_delay not in (0.0, 1.0) or runout != 0.0
    lib = _lib_comb if comb else _lib
    lib.simulate(float(rpm), float(a), float(g),
                  P.M, P.C, P.K, P.GA_GAIN, P.GA_A1, P.GA_A0,
                  P.K_TC, P.K_RC, fz, P.D_TOOL, P.N_TEETH, P.HELIX,
                  PHI_ST, PHI_EX, spr, nrev, nlay, nsub, float(vmax), int(fmode),
                  int(store), *bufs, summ, opt,
                  None if pbuf is None else pbuf.ctypes.data_as(ctypes.c_void_p))
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
