"""
control.py — the parallel "simulation vs experiment" architecture of Fig. 9.
=============================================================================
run_case(...) executes one adaptive-control run of Sections 4-5 of Ouyang et
al. (2026) on either plant:

  plant='rom'    the paper's sd-DMDc-FxLMS SIMULATION: plant and compensation
                 path are the same identified model, no noise;
  plant='truth'  the virtual EXPERIMENT: the testbed's truth model with the
                 electronics and 1 mV sensor noise, compensation path = the
                 identified model.

Both start from the steady-state forced response (excitation switched on long
before), run `pre` seconds uncontrolled, then switch FxLMS on -- the paper's
timeline (excitation at 5 s, control at 25 s) without simulating 20 s of a
transient that has already decayed.
"""
import numpy as np

import sim

A_EXC = 5.0            # PZT excitation amplitude [V] (Sec. 3.4)


class Plants:
    """Holds the modal forms of the truth testbed and of an identified ROM."""

    def __init__(self, testbed, rom, comp=None):
        """comp: model used for the compensation path G_c (default: the ROM,
        as in the paper; a model with FEM-like errors reproduces Fig. 17)."""
        self.T, self.M = testbed, rom
        self.C = rom if comp is None else comp
        self.fs = testbed.fs
        lam, Bz, Cz = sim.to_modal(rom.A, rom.B, rom.Cv)
        self.rom_modal = (lam, Bz, Cz)
        self.comp_modal = (lam, Bz, Cz) if comp is None else \
            sim.to_modal(comp.A, comp.B, comp.Cv)

    def plant(self, which, sensors, actuators):
        """Modal (lam, Bz, Cz) with inputs [E, actuators...] and outputs the
        sensing voltages of `sensors` (0 = S1, 1 = S2)."""
        cols = [0] + [1 + a for a in actuators]
        if which == 'truth':
            T = self.T
            return T.lam, T.Bz[:, cols].copy(), T.Cz[T.iv[sensors]].copy()
        lam, Bz, Cz = self.rom_modal
        return lam, Bz[:, cols].copy(), Cz[sensors].copy()

    def comp(self, sensors, actuators):
        """Compensation path G_c = identified secondary path (Eq. 21)."""
        lam, Bz, Cz = self.comp_modal
        cols = [1 + a for a in actuators]
        return lam, Bz[:, cols].copy(), Cz[sensors].copy()


def run_case(P, which, f_exc, sensors, actuators, mu, L, pre=1.0, post=21.0,
             seed=0, noise_sigma=None, a_exc=A_EXC, Lmax=None, return_signals=False):
    fs = P.fs
    sensors = list(np.atleast_1d(sensors))
    actuators = list(np.atleast_1d(actuators))
    na = len(actuators)
    mu = np.broadcast_to(np.asarray(mu, float), (na,)).copy()
    L = np.broadcast_to(np.asarray(L, np.int64), (na,)).copy()
    Lmax = int(L.max()) if Lmax is None else Lmax
    lam_p, Bz_p, Cz_p = P.plant(which, sensors, actuators)
    lam_c, Bz_c, Cz_c = P.comp(sensors, actuators)
    om = 2 * np.pi * f_exc / fs
    k_on = int(round(pre * fs))
    ns = k_on + int(round(post * fs))
    k = np.arange(ns)
    x = a_exc * np.sin(om * k)                    # = Re(-i a e^{i om k})
    # steady state of plant (input 0 = x) and of each compensation filter
    u_amp = np.zeros(Bz_p.shape[1], complex)
    u_amp[0] = -1j * a_exc
    Zp, Zm = sim.steady_state_modal(lam_p, Bz_p, u_amp, om)
    z0 = Zp + Zm
    zc0 = np.zeros((na, lam_c.size), complex)
    for j in range(na):
        ua = np.zeros(na, complex)
        ua[j] = -1j * a_exc
        Zp, Zm = sim.steady_state_modal(lam_c, Bz_c, ua, om)
        zc0[j] = Zp + Zm
    if noise_sigma is None:
        noise_sigma = P.T.sigma_v if which == 'truth' else 0.0
    rng = np.random.default_rng(seed)
    noise = noise_sigma * rng.standard_normal((ns, len(sensors)))
    w0 = np.zeros((na, Lmax))
    e, y, w = sim.fxlms_loop(lam_p, Bz_p, Cz_p, z0, lam_c, Bz_c, Cz_c, zc0,
                             x, noise, mu, L, k_on, w0)
    # convergence time on the (first) error sensor -- the paper's metric
    t_c, ref = sim.convergence_time(e[:, 0], fs, f_exc, k_on)
    out = dict(t_c=t_c, amp0=ref * np.sqrt(2),
               amp_end=np.sqrt(2 * np.mean(e[-int(0.5 * fs):, 0]**2)),
               y_peak=np.abs(y[-int(0.5 * fs):]).max(axis=0))
    if return_signals:
        out.update(e=e, y=y, w=w, k_on=k_on)
    return out


def lti_t5(P, which, f_exc, sensor, actuators, mu, L, a_exc=A_EXC,
           gram_correction=False, fast=True):
    """Averaged-LTI prediction (theory.py) of the convergence time, using
    the identified compensation path and the chosen plant ('rom' | 'truth').
    gram_correction=True replaces L/2 by the smaller eigenvalue of
    Gamma(L, w) (slow direction of the exact two-weight system)."""
    import theory
    fs = P.fs
    actuators = list(np.atleast_1d(actuators))
    na = len(actuators)
    mu = np.broadcast_to(np.asarray(mu, float), (na,))
    L = np.broadcast_to(np.asarray(L, float), (na,))
    w = 2 * np.pi * f_exc / fs
    Hm = P.M.frf([f_exc])[0]
    Hc = P.C.frf([f_exc])[0]
    gc = np.array([Hc[sensor, 1 + a] for a in actuators])
    leff = np.array([theory.gram(int(Lj), w)[1].min() * 2 if gram_correction else Lj
                     for Lj in L])
    betas = mu * leff * a_exc**2 * np.abs(gc)
    Ah, Bh, Ch = theory.averaged_controller(betas, w, np.angle(gc))
    cols = [1 + a for a in actuators]
    if which == 'rom':
        Ag, Bg, Cg = P.M.A, P.M.B[:, cols], P.M.Cv[[sensor]]
        Gp = Hm[sensor, 0]
    else:
        T = P.T
        Ag, Bg, Cg = T.Ad, T.Bd[:, cols], T.Cc[[T.iv[sensor]]]
        Gp = T.frf([f_exc], out_rows=T.iv)[0][sensor, 0]
    Acl, Bd, Ccl = theory.closed_loop(Ag, Bg, Cg, Ah, Bh, Ch)
    return theory.lti_t5(Acl, Bd, Ccl, Gp * (-1j * a_exc), w, fs, f_exc, fast=fast)


class LTIModel:
    """Minimal model container (A, B, Cv) with the frf() of SdDMDc."""

    def __init__(self, A, B, Cv, fs):
        self.A, self.B, self.Cv, self.fs = A, B, Cv, fs

    def frf(self, f):
        res = []
        for fi in np.atleast_1d(f):
            z = np.exp(1j * 2 * np.pi * fi / self.fs)
            res.append(self.Cv @ np.linalg.solve(z * np.eye(self.A.shape[0]) - self.A, self.B))
        return np.array(res)


def fem_like_model(zeta=0.006):
    """Stand-in for the paper's FEM baseline (Sec. 3.3, Fig. 17): the same
    plate and transducers, but with the FEM natural frequencies of Table 4
    (errors +1.8 % ... +4.4 %, mode 5 -3.0 %) and no electronics.  This is the
    secondary-path model a designer would get from an un-updated FE model."""
    from testbed import Testbed
    from plate import F_FEM
    T = Testbed(zeta=zeta, f_override=F_FEM, f_amp=1e7, f_lp=1e6, f_hp=1e-6)
    return LTIModel(T.Ad, T.Bd, T.Cc[T.iv], T.fs)


# Published converged voltages (Figs. 11, 13, 14): sum_j |G_s,j(w0)|^2 of the
# real plate, as d^2 / sum_j |y_j|^2 (minimum-norm FxLMS solution).
PAPER_G2 = {120.0: (0.272 / 1.570)**2, 280.0: (0.341 / 1.160)**2,
            85.0: 0.410**2 / (1.901**2 + 2.215**2)}
NU_CASES = ((120.0, 1, (1,)), (280.0, 0, (0,)), (85.0, 1, (0, 1)))


def gain_normalisation(T):
    """nu per control frequency: rescales mu so that the testbed runs at the
    paper's adaptation gain sum|G_s|^2, relative to the calibration case
    (120 Hz, where nu = 1 by construction because mu_scale was fitted there)."""
    def g2(f, s, acts):
        H = T.frf([f], out_rows=T.iv)[0]
        return float(sum(abs(H[s, 1 + a])**2 for a in acts))
    ref = PAPER_G2[120.0] / g2(120.0, 1, (1,))
    return {f: PAPER_G2[f] / g2(f, s, acts) / ref for f, s, acts in NU_CASES}, \
        {f: g2(f, s, acts) for f, s, acts in NU_CASES}
