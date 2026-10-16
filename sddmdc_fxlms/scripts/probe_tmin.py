"""
M1 preview -- does t_min scale as 1/sigma_n (P4 in docs/05)?
=============================================================
Truth plant at several damping ratios, compensation path = the truth's own
path (so only damping changes), averaged-LTI convergence time minimised over
beta.  Pre-declared: t_min * sigma_n constant within 15 % while
|Delta|/sigma_n >= 5; the regime is expected to change below that.
Writes results/tmin_probe.txt.
"""
import numpy as np

from _common import Tee, Testbed
import theory

log = Tee('tmin_probe.txt')
CASES = [('120 Hz / S2 / A2', 120.0, 1, 1, 3), ('280 Hz / S1 / A1', 280.0, 0, 0, 5)]
for label, f, s, a, k in CASES:
    log(f"\n=== {label} (adjacent mode {k + 1}) ===")
    log("  zeta %  | sigma_n | Delta/sigma | t_min [s] | t_min*sigma_n | beta_opt/beta_div")
    for zeta in (0.0010, 0.0017, 0.0024, 0.0035, 0.0050, 0.0075, 0.0100):
        T = Testbed(zeta=zeta)
        fs = T.fs
        w0 = 2 * np.pi * f / fs
        H = T.frf([f], out_rows=T.iv)[0]
        g, gp = H[s, 1 + a], H[s, 0]
        sig = zeta * T.P.omega[k]
        dl = abs(2 * np.pi * (f - T.P.freq[k]))
        Ag, Bg, Cg = T.Ad, T.Bd[:, [1 + a]], T.Cc[[T.iv[s]]]

        def run(beta):
            Ah, Bh, Ch = theory.averaged_controller(beta * abs(g), w0, np.angle(g))
            Acl, Bd, Ccl = theory.closed_loop(Ag, Bg, Cg, Ah, Bh, Ch)
            return theory.lti_t5(Acl, Bd, Ccl, gp * (-5j), w0, fs, f)
        lo, hi = 1e-8, 1.0
        while run(hi)[1] <= 1 + 1e-9:
            hi *= 2
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if run(mid)[1] <= 1 + 1e-9 else (lo, mid)
        bdiv = lo
        fr = np.linspace(0.05, 0.99, 60)
        ts = np.array([run(x * bdiv)[0] for x in fr])
        i = int(np.argmin(ts))
        log(f"  {100 * zeta:6.2f}  | {sig:7.3f} | {dl / sig:11.1f} | {ts[i]:9.2f} | {ts[i] * sig:13.2f} | {fr[i]:.2f}")
