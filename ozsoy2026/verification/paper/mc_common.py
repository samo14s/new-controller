"""Shared set-up of the Monte Carlo robustness analyses (mc_closed_form.py, mc_kick.py).

Not an analysis by itself.  It holds
  * the uncertain parameters, their ranges and the seeded sampler;
  * apply_params(): sets one parameter sample in EVERY src module that reads it,
    including the copies taken by value or computed at import (the caches);
  * Model: a vectorised re-implementation of the closed-form (zero-order,
    averaged) model of src/c4_robust.py and src/zoa.py for one sample, so that
    thousands of samples are affordable.  It is checked against the src
    functions (zoa.a_lim, c4_robust.a_lim_cf, c4_robust.unstable_zeros,
    c4_robust.worst_orbit_command) in mc_closed_form.py.

Uncertain parameters (uniform, independent):
  K_TC, K_RC   tangential / radial cutting coefficients, +-LEVEL (LEVEL = 0.1 or 0.2)
  FN           natural frequency +-5 %, with the modal stiffness K fixed (so the
               modal mass M = K / (2 pi FN)^2 changes; the identified dynamic
               stiffness 2 zeta K at resonance is kept)
  ZETA         modal damping ratio +-20 % around 1.34 % (C = 2 ZETA sqrt(K M))
  GA_GAIN      actuator force constant K_a +-10 % (N/V); the clip is a voltage
               (sampled separately), so K_a does not change it
  runout       eccentric radial runout of the teeth, 0-5 um; tooth i sits
               runout cos(2 pi i / N + psi) further out (tdsim.c, c4_robust)
  vclip        clip level of the command, 8.1-9 V
  tau          loop delay, 0-300 us
Common random numbers: the same uniform draws serve both levels; only the
spread of K_TC and K_RC differs.

Caches handled by apply_params (found by reading src):
  params      K_TC, K_RC, FN, WN, M, ZETA, C (set_damping), GA_GAIN (read at call
              time by tdsim.simulate, feasibility, c2, c4_robust.G_struct)
  structure   GA_GAIN imported by value (used by Ga, Gcl, state_space)
  zoa         K_TC, K_RC imported by value; KR and ALPHA computed at import
  c2_certificate  KAPPA computed at import (VMAX left at 9 V: unused here)
  c4_robust   KAPPA copied from c2 at import (used by delta)
  feasibility _W, the slice-force harmonics computed at import from K_TC, K_RC
  tdsim       nothing: M, C, K, GA_GAIN, K_TC, K_RC are passed from params at
              every call; the immersion angles (fixed here) are cached
  c1_unsafe_zone  VMAX = 9 V at import (the clip is passed explicitly instead);
              _critical_kick calls params.set_damping(zeta), which uses the
              current params.K and params.M
No functools.lru_cache is used in src.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import params as P                          # noqa: E402
import structure                            # noqa: E402
import zoa                                  # noqa: E402
import feasibility                          # noqa: E402
import c2_certificate as c2                 # noqa: E402
import c4_robust as R                       # noqa: E402

ZETA0 = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])      # 1.34 %
P.set_damping(ZETA0)
G_DESIGN = 1500.0                           # V s/m, gain of the C1 basin points
SPEEDS = (1090.0, 1420.0, 2500.0, 2870.0)   # C1 basin speeds
FRACS = (0.5, 0.7, 0.9, 0.98)               # depths as fractions of the NOMINAL a_lim
LEVELS = (0.1, 0.2)                         # spread of K_TC and K_RC
SEED = 20261124

NAMES = ("K_TC", "K_RC", "FN", "ZETA", "GA_GAIN", "runout", "vclip", "tau")
NOMINAL = dict(K_TC=P.K_TC, K_RC=P.K_RC, FN=P.FN, ZETA=ZETA0, GA_GAIN=P.GA_GAIN,
               runout=0.0, vclip=P.F_MAX / P.GA_GAIN, tau=0.0)
UNITS = dict(K_TC="N/m^2", K_RC="N/m^2", FN="Hz", ZETA="-", GA_GAIN="N/V", runout="m", vclip="V", tau="s")
_K0 = P.K                                   # modal stiffness, kept fixed


def ranges(level):
    """(low, high) of every parameter for a cutting-coefficient spread `level`."""
    n = NOMINAL
    return dict(K_TC=(n["K_TC"] * (1 - level), n["K_TC"] * (1 + level)),
                K_RC=(n["K_RC"] * (1 - level), n["K_RC"] * (1 + level)),
                FN=(n["FN"] * 0.95, n["FN"] * 1.05),
                ZETA=(n["ZETA"] * 0.8, n["ZETA"] * 1.2),
                GA_GAIN=(n["GA_GAIN"] * 0.9, n["GA_GAIN"] * 1.1),
                runout=(0.0, 5e-6), vclip=(8.1, 9.0), tau=(0.0, 300e-6))


def uniforms(n, seed=SEED):
    """The (n, 8) uniform draws shared by both levels (columns in NAMES order),
    plus one extra column for the runout phase of the simulated samples."""
    return np.random.default_rng(seed).random((n, len(NAMES) + 1))


def draw(n, level, seed=SEED):
    """n samples (dict of arrays) at a cutting-coefficient spread `level`;
    'psi' is a runout phase in [0, 2 pi) used only by the simulator."""
    U = uniforms(n, seed)
    rg = ranges(level)
    out = {k: rg[k][0] + U[:, j] * (rg[k][1] - rg[k][0]) for j, k in enumerate(NAMES)}
    out["psi"] = 2 * np.pi * U[:, len(NAMES)]
    return out


def sample(S, i):
    """Sample i of a draw() result as a dict of floats."""
    return {k: float(v[i]) for k, v in S.items()}


# ------------------------------------------------------------- caches ----
def alpha_yy(kr):
    """Average directional factor alpha_yy (zoa.alpha_matrix with KR = kr)."""
    st, ex = P.immersion_angles()

    def ev(f):
        return f(ex) - f(st)
    return 0.5 * ev(lambda p: -np.cos(2 * p) - 2 * kr * p - kr * np.sin(2 * p))


def apply_params(p):
    """Set one parameter sample in every src module (see the module docstring).
    p: dict with at least K_TC, K_RC, FN, ZETA, GA_GAIN; returns p."""
    P.K_TC, P.K_RC = float(p["K_TC"]), float(p["K_RC"])
    P.FN = float(p["FN"])
    P.WN = 2 * np.pi * P.FN
    P.K = _K0
    P.M = P.K / P.WN ** 2
    P.set_damping(float(p["ZETA"]))           # C from the new M
    P.GA_GAIN = float(p["GA_GAIN"])
    structure.GA_GAIN = P.GA_GAIN
    zoa.K_TC, zoa.K_RC = P.K_TC, P.K_RC
    zoa.KR = P.K_RC / P.K_TC
    zoa.ALPHA = zoa.alpha_matrix()[1, 1] if zoa.FLEX_DIR == "y" else zoa.alpha_matrix()[0, 0]
    c2.KAPPA = P.N_TEETH * P.K_TC * zoa.ALPHA / (4 * np.pi)
    R.KAPPA = c2.KAPPA
    feasibility._W = feasibility.slice_force_harmonics()
    return p


def restore_nominal():
    return apply_params(NOMINAL)


def cache_state():
    """The values every cached copy holds now (for the checks)."""
    return dict(P_K_TC=P.K_TC, P_K_RC=P.K_RC, P_FN=P.FN, P_WN=P.WN, P_M=P.M, P_C=P.C, P_ZETA=P.ZETA,
                P_GA_GAIN=P.GA_GAIN, structure_GA_GAIN=structure.GA_GAIN, zoa_K_TC=zoa.K_TC,
                zoa_K_RC=zoa.K_RC, zoa_KR=zoa.KR, zoa_ALPHA=float(zoa.ALPHA), c2_KAPPA=float(c2.KAPPA),
                R_KAPPA=float(R.KAPPA), feas_W1=complex(feasibility._W[1]))


# ------------------------------------------------- vectorised closed form ----
N_T = R.N_THETA // 4                        # samples of one revolution for the peak (as orbit_command)
K_HARM = R.K_HARM                           # spindle harmonics kept (160)
_KK = np.arange(1, K_HARM + 1)
_TT = np.arange(N_T) / N_T * 2 * np.pi
_COS = np.cos(np.outer(_TT, _KK))           # (N_T, K_HARM)
_SIN = np.sin(np.outer(_TT, _KK))
_TH = np.arange(R.N_THETA) / R.N_THETA * 2 * np.pi
_TEETH = []
for _i in range(P.N_TEETH):
    _phi = np.mod(_TH + 2 * np.pi * _i / P.N_TEETH, 2 * np.pi)
    _TEETH.append((np.sin(_phi), np.cos(_phi), (_phi >= R._PHI_ST) & (_phi <= R._PHI_EX)))
W_ARG = R._W                                # argument-principle grid of c4_robust (0-5000 Hz)
W_DS = zoa.W_GRID                           # D-subdivision grid of zoa (0.5-600 Hz)


class Model:
    """Closed-form zero-order model of one parameter sample (DVF, g1 = g, no comb).

    Same equations as c4_robust (delta, unstable_zeros, slice_force_revolution,
    orbit_command, worst_orbit_command) and zoa.a_lim (D-subdivision), with the
    sample's parameters held here instead of in the modules."""

    def __init__(self, p, g=G_DESIGN):
        self.p = dict(p)
        self.g = float(g)
        self.Kt, self.Kr = float(p["K_TC"]), float(p["K_RC"])
        self.K = _K0
        wn = 2 * np.pi * float(p["FN"])
        self.M = self.K / wn ** 2
        self.C = 2 * float(p["ZETA"]) * np.sqrt(self.K * self.M)
        self.Ka = float(p["GA_GAIN"])
        self.tau = float(p.get("tau", 0.0))
        self.runout = float(p.get("runout", 0.0))
        self.vclip = float(p.get("vclip", NOMINAL["vclip"]))
        self.alpha = alpha_yy(self.Kr / self.Kt)
        self.kappa = P.N_TEETH * self.Kt * self.alpha / (4 * np.pi)
        self._ds = None
        self._W = None

    # transfer functions (structure.G/Ga, c4_robust.G_struct/controller)
    def G(self, s):
        return 1.0 / (self.M * s ** 2 + self.C * s + self.K)

    def Ga(self, s):
        return self.Ka * s ** 2 / (s ** 2 + P.GA_A1 * s + P.GA_A0)

    def ctrl(self, s):
        return np.exp(-s * self.tau) * (s * self.g)

    def delta(self, s, rpm, a):
        T = 60.0 / (rpm * P.N_TEETH)
        E = 1.0 - np.exp(-s * T)
        return 1.0 - self.G(s) * (self.kappa * a * E - self.Ga(s) * self.ctrl(s))

    # ---------------------------------------------------------- stability
    def zeros(self, rpm, a):
        """Zeros of Delta in Re s > 0 (c4_robust.unstable_zeros); a may be 0."""
        d = self.delta(1j * W_ARG, rpm, a)
        ph = np.unwrap(np.angle(d))
        return int(round(-(ph[-1] - ph[0]) / np.pi))

    def crossings(self, rpm):
        """All positive real roots a(w) of the characteristic equation on the zoa
        grid (zoa.a_lim with the delayed loop): the depths at which a root
        crosses the imaginary axis, sorted (m)."""
        if self._ds is None:
            s = 1j * W_DS
            Gs = self.G(s)
            self._ds = Gs / (1.0 + self.Ga(s) * self.ctrl(s) * Gs)       # Gcl with the delay
        gcl = self._ds
        T = 60.0 / (P.N_TEETH * rpm)
        w = W_DS
        den = P.N_TEETH * self.Kt * self.alpha * (1 - np.exp(-1j * w * T)) * gcl
        a = 4 * np.pi / den
        im = a.imag
        sgn = np.sign(im)
        idx = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
        ph = w * T / (2 * np.pi)
        near = np.abs(ph - np.round(ph)) < 1e-6
        keep = (np.floor(ph[idx]) == np.floor(ph[idx + 1])) & ~near[idx] & ~near[idx + 1]
        idx = idx[keep]
        if idx.size == 0:
            return np.array([])
        t = im[idx] / (im[idx] - im[idx + 1])
        are = a.real[idx] + t * (a.real[idx + 1] - a.real[idx])
        return np.sort(are[are > 0])

    def a_lim(self, rpm):
        c = self.crossings(rpm)
        return float(c[0]) if c.size else np.inf

    # ----------------------------------------------------- forced command
    def slice_force(self, psis):
        """W_k (len(psis), K_HARM + 1): c4_robust.slice_force_revolution for each
        runout phase, with this sample's runout and cutting coefficients."""
        N = P.N_TEETH
        psis = np.atleast_1d(np.asarray(psis, float))
        r = self.runout * np.cos(2 * np.pi * np.arange(N)[None, :] / N + psis[:, None])     # (npsi, N)
        w = np.zeros((len(psis), R.N_THETA))
        for i, (sn, cs, cut) in enumerate(_TEETH):
            h = P.FZ * sn[None, :] + (r[:, i] - r[:, (i + 1) % N])[:, None]
            w += np.where(cut[None, :] & (h > 0), -(self.Kt * sn - self.Kr * cs)[None, :] * h, 0.0)
        return (np.fft.rfft(w, axis=1) / R.N_THETA)[:, : K_HARM + 1]

    def psis(self, n_psi=8):
        """Runout phases of worst_orbit_command: [0, 2 pi / N) in n_psi steps."""
        return (np.arange(n_psi) * 2 * np.pi / (P.N_TEETH * n_psi)) if self.runout else np.array([0.0])

    def orbit_peaks(self, rpm, depths, psis=None, n_psi=8):
        """Peak |command| (V) of the forced orbit over one revolution,
        shape (len(psis), len(depths)) (c4_robust.orbit_command)."""
        psis = self.psis(n_psi) if psis is None else np.atleast_1d(np.asarray(psis, float))
        if self._W is None or self._W[0].shape != psis.shape or np.any(self._W[0] != psis):
            self._W = (psis, self.slice_force(psis))
        W = self._W[1]                                                   # (npsi, K+1)
        depths = np.atleast_1d(np.asarray(depths, float))
        k = _KK
        q = k * np.tan(P.HELIX) / (0.5 * P.D_TOOL)
        integ = (1 - np.exp(-1j * np.outer(depths, q))) / (1j * q)[None, :]          # (nd, K)
        F = W[:, None, 1:] * integ[None, :, :]                           # (npsi, nd, K)
        Om = 2 * np.pi * rpm / 60.0
        s = 1j * k * Om
        T = 60.0 / (rpm * P.N_TEETH)
        E = 1.0 - np.exp(-s * T)
        Gs, Gas, Cs = self.G(s), self.Ga(s), self.ctrl(s)
        dl = 1.0 - Gs[None, :] * (self.kappa * depths[:, None] * E[None, :] - (Gas * Cs)[None, :])   # (nd, K)
        X = Gs[None, None, :] * F / dl[None, :, :]
        U = -Cs[None, None, :] * X                                       # (npsi, nd, K)
        Um = U.reshape(-1, K_HARM).T                                     # (K, npsi*nd)
        u = 2 * (_COS @ Um.real - _SIN @ Um.imag)
        return np.abs(u).max(axis=0).reshape(len(psis), len(depths))

    def bmax(self, rpm, depths, n_psi=8):
        """Worst peak command over the runout phases (worst_orbit_command, one depth each)."""
        return self.orbit_peaks(rpm, depths, n_psi=n_psi).max(axis=0)
