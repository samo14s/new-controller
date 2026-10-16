"""
sddmdc.py — scaled-derivative DMDc with a displacement-strain hybrid state.
============================================================================
Section 2.2 of Ouyang et al., Thin-Walled Struct. 228 (2026) 115090,
Eqs. (2)-(14), and the path extraction of Section 2.3, Eqs. (15)-(17).

    s_k   = [x_k ; eps_k] in R^n,  n = n1 + n2                         (2)
    d_k   = ds/dt (numerical, after a low-pass filter on s)
    d^s_k = Lambda_s d_k,  Lambda_s,ii = ||s^(i)||_2 / ||d^(i)||_2      (4)
    S  = [s_1..s_m ; d^s_1..d^s_m],  S' = shifted by one,  Y = [v^e ; v^a]   (5)
    S' = [A, B] [S ; Y] = G Omega                                        (6)
    Omega ~ U Sigma V*  (rank p: 99.9 % of the cumulative singular values)  (7)
    A_bar = S' V Sigma^-1 U1*,  B_bar = S' V Sigma^-1 U2*                (9)
    S' ~ U^ Sigma^ V^*  (rank r = p - l)                                 (11)
    A~ = U^* A_bar U^,  B~ = U^* B_bar                                   (12)
    s~_{k+1} = A~ s~_k + B~_e v^e_k + B~_a v^a_k                         (13)
    s_k = U^_1 s~_k   (U^_1 = upper n rows of U^)                        (14)
    v^s = S_sens C_s s                                                   (15)-(16)
    G_p(z) = S_sens C_s U^_1 (zI - A~)^-1 B~_e,  G_s(z) = ... B~_a       (17)

Memory note: Omega is (2n + l) x m with m = 6e5 samples; the SVDs are computed
from the (2n+l)-square Gram matrices, which is algebraically identical
(V Sigma^-1 = Omega^T U Sigma^-2) and needs 77 x 77 storage instead of
6e5 x 77.  Truncation removes the small singular values whose squares would be
inaccurate, so nothing is lost at rank p.

Choices the paper does not state (docs/01, items A-*): the low-pass filter
(default 4th-order zero-phase Butterworth at 400 Hz), whether the filtered or
raw s enters S (default filtered, for consistency with d), the derivative
stencil (default 2nd-order central), and the units of x vs eps (default SI).
"""
import numpy as np
from scipy.signal import butter, filtfilt


class SdDMDc:
    """Identified reduced-order model and the quantities of Eqs. (2)-(17)."""

    def __init__(self, x, eps, u, fs, lp_cut=400.0, lp_order=4, energy=0.999,
                 r=None, p=None, use_filtered_s=True, scale=True, s_sens=1.0,
                 strain_rows=None, unit_x=1.0, unit_eps=1.0, normalize=None):
        x = np.asarray(x, float)
        eps = np.asarray(eps, float)
        u = np.asarray(u, float)
        # The 99.9 % rank rule acts on singular values of Omega = [S; Y], whose
        # rows mix metres, strain and volts: it is NOT unit-invariant.
        # normalize=None   -> use unit_x, unit_eps as given (paper: unstated)
        # normalize='group'-> displacement block, strain block and input block
        #                     each scaled to unit RMS (relative amplitudes
        #                     inside a block preserved)
        # normalize='channel' -> every channel to unit RMS
        if normalize == 'group':
            unit_x = 1.0 / np.sqrt(np.mean(x**2))
            unit_eps = 1.0 / np.sqrt(np.mean(eps**2))
            unit_u = 1.0 / np.sqrt(np.mean(u**2))
        elif normalize == 'channel':
            unit_x = 1.0 / np.sqrt(np.mean(x**2, axis=0))
            unit_eps = 1.0 / np.sqrt(np.mean(eps**2, axis=0))
            unit_u = 1.0 / np.sqrt(np.mean(u**2, axis=0))
        else:
            unit_u = 1.0
        x = x * unit_x
        eps = eps * unit_eps
        u = u * unit_u
        self.unit_u = unit_u
        self.fs, self.n1, self.n2 = fs, x.shape[1], eps.shape[1]
        self.unit_x, self.unit_eps = unit_x, unit_eps
        n = self.n1 + self.n2
        self.n, self.l = n, u.shape[1]
        s_raw = np.hstack([x, eps])                                  # (m+1) x n
        if lp_cut:
            b, a = butter(lp_order, lp_cut / (fs / 2))
            s_f = filtfilt(b, a, s_raw, axis=0)
        else:
            s_f = s_raw
        d = np.gradient(s_f, 1.0 / fs, axis=0)
        s_use = s_f if use_filtered_s else s_raw
        if scale:
            lam = np.linalg.norm(s_use, axis=0) / np.linalg.norm(d, axis=0)
        else:
            lam = np.ones(n)
        self.Lambda = lam
        X = np.hstack([s_use, d * lam])                              # (m+1) x 2n
        S, Sp, Y = X[:-1], X[1:], u[:-1]
        Om = np.hstack([S, Y])                                       # m x (2n+l)
        # --- SVD of Omega via its Gram matrix -----------------------------
        Goo = Om.T @ Om
        Gpo = Sp.T @ Om
        w, Uo = np.linalg.eigh(Goo)
        idx = np.argsort(w)[::-1]
        w, Uo = np.maximum(w[idx], 0.0), Uo[:, idx]
        sig = np.sqrt(w)
        self.sigma_omega = sig
        if p is None:
            c = np.cumsum(sig) / sig.sum()
            p = int(np.searchsorted(c, energy) + 1)
        self.p = p
        Ut = Uo[:, :p]
        Gmat = Gpo @ Ut @ np.diag(1.0 / sig[:p]**2) @ Ut.T           # Eq. (8)
        self.Abar, self.Bbar = Gmat[:, :2 * n], Gmat[:, 2 * n:]      # Eq. (9)
        # --- POD basis of S' ------------------------------------------------
        Gpp = Sp.T @ Sp
        w2, Up = np.linalg.eigh(Gpp)
        idx = np.argsort(w2)[::-1]
        self.sigma_sp = np.sqrt(np.maximum(w2[idx], 0.0))
        Up = Up[:, idx]
        if r is None:
            r = p - self.l                                           # Eq. (11)
        self.r = r
        Uh = Up[:, :r]
        self.Uhat = Uh
        self.A = Uh.T @ self.Abar @ Uh                               # Eq. (12)
        self.B = Uh.T @ self.Bbar * unit_u                           # back to volts
        self.U1 = Uh[:n]                                             # Eq. (14)
        # output maps: displacement (m) and sensing voltage (V)
        self.Cx = self.U1[:self.n1] / np.reshape(unit_x, (-1, 1))
        self.Ceps = self.U1[self.n1:] / np.reshape(unit_eps, (-1, 1))
        self.s_sens = s_sens
        self.Cv = s_sens * self.Ceps                                 # Eqs. (15)-(16)

    # ------------------------------------------------------------------
    def modal(self):
        """Discrete eigenvalues -> natural frequencies [Hz] and damping
        ratios, sorted by frequency (complex pairs counted once)."""
        lam, V = np.linalg.eig(self.A)
        sc = np.log(lam.astype(complex)) * self.fs
        f = np.abs(sc) / (2 * np.pi)
        zeta = -sc.real / np.abs(sc)
        keep = lam.imag >= 0
        o = np.argsort(f[keep])
        return f[keep][o], zeta[keep][o], V[:, keep][:, o], lam[keep][o]

    def mode_shapes(self):
        """Displacement mode shapes at the n1 measurement points."""
        f, z, V, _ = self.modal()
        return f, self.Cx @ V

    def simulate(self, u, out='v'):
        """ROM response to inputs u (n_steps x l)."""
        from sim import to_modal, modal_sim
        C = self.Cv if out == 'v' else self.Cx
        lam, Bz, Cz = to_modal(self.A, self.B, C)
        return modal_sim(lam, Bz, Cz, np.ascontiguousarray(u, float),
                         np.zeros(self.r, np.complex128))

    def path(self, inputs, out='v'):
        """(A, B, C) of the discrete path from the given input columns to the
        sensing voltages: G_p for [0], G_s for [1], [2] (Eq. 17)."""
        C = self.Cv if out == 'v' else self.Cx
        return self.A, self.B[:, inputs], C

    def frf(self, f, out='v'):
        C = self.Cv if out == 'v' else self.Cx
        res = []
        for fi in np.atleast_1d(f):
            z = np.exp(1j * 2 * np.pi * fi / self.fs)
            res.append(C @ np.linalg.solve(z * np.eye(self.r) - self.A, self.B))
        return np.array(res)


def rel_errors(y_exp, y_mod):
    """Eq. (22): relative L2 errors in time and frequency domain [%]."""
    et = np.linalg.norm(y_exp - y_mod) / np.linalg.norm(y_exp) * 100
    F, Fp = np.fft.rfft(y_exp), np.fft.rfft(y_mod)
    ef = np.linalg.norm(np.abs(F) - np.abs(Fp)) / np.linalg.norm(np.abs(F)) * 100
    return et, ef
