"""
testbed.py — the virtual experiment that replaces the laboratory.
==================================================================
Everything the dSPACE rig of Fig. 3 / Fig. 9 of Ouyang et al. (2026) does,
as a discrete LTI system sampled at fs = 10 kHz with a zero-order hold:

    command v (V) --[amplifier: gain g, 1st-order bandwidth f_amp]-->
    patch voltage --[truth plate: n_modes Ritz modes, modal damping zeta]-->
    { w at the 35 laser points (m), eps_x under S1, S2 }
    eps --[MFC P2 + charge amplifier: S_sens V/strain, high-pass f_hp,
           2nd-order Butterworth low-pass f_lp]--> sensing voltage (V)
    + white measurement noise on every channel.

Why each ingredient is there: the identified sd-DMDc model will be rank-
truncated (it keeps the dominant modes only), will see noisy data, and will not
know the amplifier / charge-amplifier dynamics -- exactly the three sources of
mismatch between "simulation" and "experiment" in the paper.  The truth keeps
40 modes (to 2.8 kHz), the electronics, and the noise, so the mismatch is real
and measurable instead of zero by construction.

Calibration (docs/02): Young/shear moduli updated so the six natural
frequencies match Table 4 (experiment) within 1.3 %; electrical gains and the
damping ratio fitted to the six measured sine amplitudes of Fig. 8 / Table 5
and the two converged actuating voltages of Figs. 11 and 13.
"""
import numpy as np
from scipy.linalg import expm
from scipy.signal import butter, chirp

from plate import build_paper_plate, PLATE, GRID, F_EXP
import sim

FS = 10_000.0

# Moduli updated to the measured frequencies of Table 4 (see calibrate.py):
# max |error| 1.28 % (mode 6) against 4.43 % for the paper's own FEM.
PLATE_UPDATED = dict(PLATE, E1=65.817e9, E2=50.468e9, G12=4.3320e9)

# Electrical chain (scripts/calibrate.py): S_SENS chosen so that 0.27 V is
# ~2.7 microstrain; G_E fitted (geometric mean) to the six sine amplitudes of
# Fig. 8 under 5 V; G_A to the converged actuating voltages of Fig. 11
# (A2 -> S2, 120 Hz) and Fig. 13 (A1 -> S1, 280 Hz).  Both land in the
# physical range of the stated hardware (x10 PZT amplifier, 4 V -> 826 V on
# an MFC P1 whose limit is +1500 V).  sigma_v = 1 mV reproduces the 45 dB
# operational SNR of Sec. 4.2 at the 0.272 V reference amplitude.
S_SENS = 1.0e5
G_E = 9.99
G_A = 206.5


class Testbed:
    """Truth plant + signal chain.

    Inputs  (commands, V):  [v_E, v_A1, v_A2]
    Outputs:                [w_1..w_35 (m), v_S1, v_S2 (V)]
    """

    IN = ('E', 'A1', 'A2')

    def __init__(self, n_modes=40, zeta=0.006, fs=FS, g_e=G_E, g_a=G_A,
                 s_sens=S_SENS, f_amp=5e3, f_hp=0.5, f_lp=3e3,
                 plate=None, pos=None, sigma_v=1.0e-3, sigma_w=0.5e-6,
                 Mx=14, Ny=10, f_override=F_EXP):
        self.fs = fs
        self.P = build_paper_plate(Mx, Ny, n_modes=n_modes,
                                   plate=PLATE_UPDATED if plate is None else plate,
                                   pos=pos)
        P = self.P
        if f_override is not None:
            # pin the first modal frequencies to measured values (Table 4)
            # without touching the mode shapes -- standard modal updating
            fo = np.asarray(f_override, float)
            P.omega = P.omega.copy()
            P.omega[:fo.size] = 2 * np.pi * fo
            P.freq = P.omega / (2 * np.pi)
        self.n_modes = n_modes
        self.zeta = np.broadcast_to(np.asarray(zeta, float), (n_modes,)).copy()
        self.g = np.array([g_e, g_a, g_a], float)
        self.s_sens = float(s_sens)
        self.f_amp, self.f_hp, self.f_lp = f_amp, f_hp, f_lp
        self.sigma_v, self.sigma_w = sigma_v, sigma_w
        F = np.column_stack([P.act[n]['f'] for n in self.IN])      # dof x 3
        self.Bq = P.Phi.T @ F                                       # modal forces / V
        self.Cw = P.w_rows(GRID) @ P.Phi                            # 35 x modes
        self.Ce = np.vstack([P.sen['S1']['row'], P.sen['S2']['row']]) @ P.Phi
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        N = self.n_modes
        w = self.P.omega
        z = self.zeta
        # modal block: [eta; eta_dot]
        Am = np.block([[np.zeros((N, N)), np.eye(N)],
                       [-np.diag(w**2), -np.diag(2 * z * w)]])
        # amplifiers: a' = wa (g u - a), 3 states
        wa = 2 * np.pi * self.f_amp
        # sensor chain per sensor: HP (1 state) then 2nd-order Butterworth LP
        wh, wl = 2 * np.pi * self.f_hp, 2 * np.pi * self.f_lp
        nA = 2 * N + 3 + 2 * 3
        A = np.zeros((nA, nA))
        B = np.zeros((nA, 3))
        A[:2 * N, :2 * N] = Am
        ia = 2 * N
        A[ia:ia + 3, ia:ia + 3] = -wa * np.eye(3)
        B[ia:ia + 3, :] = wa * np.diag(self.g)
        A[N:2 * N, ia:ia + 3] = self.Bq                  # patch voltage drives modes
        # sensors: input = S_sens * eps = S_sens * Ce eta
        Cv = np.zeros((2, nA))
        for s in range(2):
            i0 = ia + 3 + 3 * s
            u_row = np.zeros(nA)
            u_row[:N] = self.s_sens * self.Ce[s]          # strain -> volts
            # HP: xh' = -wh xh + u ; out_hp = u - wh xh
            A[i0, :] += u_row
            A[i0, i0] += -wh
            # LP Butterworth (controllable canonical): x1' = x2,
            # x2' = -wl^2 x1 - sqrt2 wl x2 + wl^2 * out_hp ; y = x1
            A[i0 + 1, i0 + 2] = 1.0
            A[i0 + 2, i0 + 1] = -wl**2
            A[i0 + 2, i0 + 2] = -np.sqrt(2) * wl
            A[i0 + 2, :] += wl**2 * u_row
            A[i0 + 2, i0] += -wl**2 * wh
            Cv[s, i0 + 1] = 1.0
        Cw = np.zeros((35, nA))
        Cw[:, :N] = self.Cw
        self.Ac, self.Bc = A, B
        self.Cc = np.vstack([Cw, Cv])
        self.nx = nA
        # exact ZOH discretisation
        M = np.zeros((nA + 3, nA + 3))
        M[:nA, :nA] = A
        M[:nA, nA:] = B
        Md = expm(M / self.fs)
        self.Ad, self.Bd = Md[:nA, :nA], Md[:nA, nA:]
        self.lam, self.Bz, self.Cz = sim.to_modal(self.Ad, self.Bd, self.Cc)
        self.iv = np.array([35, 36])                      # rows of v_S1, v_S2

    # ------------------------------------------------------------------
    def frf(self, f, out_rows=None):
        """Discrete FRF H(e^{i w}) [n_f, n_out, 3] from command V to outputs."""
        f = np.atleast_1d(f)
        rows = slice(None) if out_rows is None else out_rows
        Cz = self.Cz[rows]
        H = []
        for fi in f:
            zi = np.exp(1j * 2 * np.pi * fi / self.fs)
            H.append(Cz @ (self.Bz / (zi - self.lam)[:, None]))
        return np.array(H)

    def run(self, u, seed=0, noise=True):
        """Open-loop run. u: (n_steps, 3) commands. Returns dict with noisy
        displacement (m), noisy sensing voltages (V), and the clean outputs."""
        y = sim.modal_sim(self.lam, self.Bz, self.Cz, np.ascontiguousarray(u, float),
                          np.zeros(self.nx, np.complex128))
        rng = np.random.default_rng(seed)
        yn = y.copy()
        if noise:
            yn[:, :35] += self.sigma_w * rng.standard_normal((len(u), 35))
            yn[:, 35:] += self.sigma_v * rng.standard_normal((len(u), 2))
        return dict(w=yn[:, :35], v=yn[:, 35:], w_clean=y[:, :35], v_clean=y[:, 35:])


# ----------------------------------------------------------------------
# Section 3.1 identification protocol (Fig. 4b)
# ----------------------------------------------------------------------
def identification_input(fs=FS, seg=20.0, t_chirp=19.0, f0=5.0, f1=300.0,
                         amp=(5.0, 4.0, 4.0)):
    """Three sequential segments (E, then A1, then A2), each a linear chirp
    f0 -> f1 over t_chirp seconds followed by a zero-input tail; amplitude
    5 V for the PZT and 4 V for the MFCs (limits stated in Sec. 3.1).  The
    chirp sweep rate is read from Fig. 5a: resonance peaks at 5.8 s (mode 3)
    and 16 s (mode 5) fit a linear 5->300 Hz sweep over ~19-20 s."""
    n = int(round(seg * fs))
    t = np.arange(n) / fs
    c = np.where(t < t_chirp, chirp(t, f0, t_chirp, f1, method='linear'), 0.0)
    u = np.zeros((3 * n, 3))
    for j in range(3):
        u[j * n:(j + 1) * n, j] = amp[j] * c
    return u
