"""
Phase 1a -- Section 3 of the paper on the virtual testbed.
==========================================================
  * Table 4 analogue: modal frequencies of the identified sd-DMDc model vs the
    truth and vs the paper (experiment / sd-DMDc / FEM);
  * Fig. 7 analogue: MAC of the identified mode shapes at the 35 points;
  * Table 3 analogue: eps_t, eps_f (Eq. 22) for the six input/output pairs;
  * Table 5 analogue: steady sine amplitudes under 5 V PZT excitation;
  * what the paper did NOT do: an out-of-sample check (fresh noise, random
    excitation) and the sensitivity of the 99.9 % rank rule to the units.
Writes results/identification.txt and results/rom.npz.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from _common import (Tee, identify, Testbed, identification_input, SdDMDc,
                     ROM_FILE)
from sddmdc import rel_errors
from plate import F_EXP, F_FEM, F_SDDMDC, mode_type
import sim

log = Tee('identification.txt')
T, M, u, out = identify()
fs = T.fs
log("=== sd-DMDc identification on the virtual testbed ===")
log(f"record: {u.shape[0]} samples ({u.shape[0] / fs:.0f} s), n = {M.n} channels "
    f"(35 w + 2 eps), l = {M.l} inputs")
log(f"normalize='group'  ->  p = {M.p}  (99.9 % cumulative singular values of Omega), "
    f"r = p - l = {M.r}")

# ---------------------------------------------------------------- Table 4
f, z, V, lam = M.modal()
light = (z < 0.05) & (f > 3) & (f < 320)
fl, zl = f[light], z[light]
ftrue = T.P.freq[:6]
pick = [int(np.argmin(np.abs(fl - ft))) for ft in ftrue]
fid = fl[pick]
zid = zl[pick]
log("\nTable 4 analogue -- natural frequencies [Hz]")
log(" mode | truth (=Table 4 exp) | sd-DMDc here | err %  | zeta id | paper sd-DMDc err % | paper FEM err %")
for k in range(6):
    log(f"  {k + 1}   | {ftrue[k]:8.2f} ({mode_type(T.P, k)})       | {fid[k]:8.2f}     | "
        f"{100 * (fid[k] / ftrue[k] - 1):+6.3f} | {zid[k]:.4f}  | "
        f"{100 * (F_SDDMDC[k] / F_EXP[k] - 1):+6.2f}              | {100 * (F_FEM[k] / F_EXP[k] - 1):+6.2f}")
log(f" truth damping ratio: {T.zeta[0]:.4f} (all modes)")
nspur = int(light.sum() - 6)
log(f" extra lightly-damped (zeta < 5 %) eigenvalues below 320 Hz: {nspur} "
    f"(frequencies {np.round(np.delete(fl, pick), 1)})")

# ---------------------------------------------------------------- MAC
Cw = T.Cw[:, :6]
_, Phi_id = M.mode_shapes()
light_idx = np.nonzero(light)[0][pick]
Phi_id = Phi_id[:, light_idx]
mac = np.abs(Cw.T @ Phi_id)**2 / np.outer(np.sum(Cw**2, 0), np.sum(np.abs(Phi_id)**2, 0))
log("\nFig. 7 analogue -- MAC(truth mode i, identified mode i) at the 35 points: "
    + " ".join(f"{mac[i, i]:.4f}" for i in range(6)))
log(" max off-diagonal MAC: %.3f" % np.max(mac - np.diag(np.diag(mac))))

# ---------------------------------------------------------------- Table 3
vr = M.simulate(u)
n = u.shape[0] // 3
log("\nTable 3 analogue -- Eq. (22) on the identification record (in-sample)")
log(" pair     | eps_t here | eps_f here | paper eps_t | paper eps_f")
paper3 = {('E', 'S1'): (14.84, 9.47), ('E', 'S2'): (13.57, 8.25), ('A1', 'S1'): (14.11, 9.87),
          ('A1', 'S2'): (15.17, 9.80), ('A2', 'S1'): (13.93, 9.20), ('A2', 'S2'): (15.78, 10.51)}
for j, name in enumerate(['E', 'A1', 'A2']):
    seg = slice(j * n, (j + 1) * n)
    for s, sn in enumerate(['S1', 'S2']):
        et, ef = rel_errors(out['v'][seg, s], vr[seg, s])
        pt, pf = paper3[(name, sn)]
        log(f" {name:2s} -> {sn} | {et:9.2f}% | {ef:9.2f}% | {pt:10.2f}% | {pf:10.2f}%")

# ---------------------------------------------------------------- out-of-sample
rng = np.random.default_rng(7)
nt = int(20 * fs)
b, a = butter(4, [5 / (fs / 2), 300 / (fs / 2)], btype='band')
uo = np.zeros((3 * nt, 3))
for j, amp in enumerate([5.0, 4.0, 4.0]):
    sig = filtfilt(b, a, rng.standard_normal(nt))
    uo[j * nt:(j + 1) * nt, j] = amp * sig / np.abs(sig).max()
oo = T.run(uo, seed=99)
vo = M.simulate(uo)
log("\nOut-of-sample (band-limited random 5-300 Hz, fresh noise) -- NOT in the paper")
for j, name in enumerate(['E', 'A1', 'A2']):
    seg = slice(j * nt, (j + 1) * nt)
    row = []
    for s in range(2):
        et, ef = rel_errors(oo['v'][seg, s], vo[seg, s])
        row.append(f"S{s + 1}: eps_t={et:5.2f}% eps_f={ef:5.2f}%")
    log(f" {name:2s}: " + "   ".join(row))

# ---------------------------------------------------------------- Table 5
log("\nTable 5 analogue -- steady sine amplitude under 5 V PZT, model vs testbed")
log(" f [Hz] sensor | testbed (noisy) | sd-DMDc | rel err | paper exp | paper err")
paper5 = {(20, 0): (0.225, 4.89), (245, 0): (0.688, 9.53), (280, 0): (0.341, 0.59),
          (35, 1): (0.0190, 7.89), (85, 1): (0.410, 2.93), (120, 1): (0.272, 6.25)}
Hm = M.frf(np.array([k[0] for k in paper5], float))
for i, ((fq, s), (pe, perr)) in enumerate(paper5.items()):
    ns = int(4 * fs)
    k = np.arange(ns)
    us = np.zeros((ns, 3))
    us[:, 0] = 5.0 * np.sin(2 * np.pi * fq * k / fs)
    lam_t, Bz_t, Cz_t = T.lam, T.Bz, T.Cz[T.iv]
    Zp, Zm = sim.steady_state_modal(lam_t, Bz_t, np.array([-5j, 0, 0]), 2 * np.pi * fq / fs)
    y = sim.modal_sim(lam_t, Bz_t, Cz_t, us, Zp + Zm)[:, s] + T.sigma_v * rng.standard_normal(ns)
    amp_t = np.sqrt(2 * np.mean(y**2))
    amp_m = 5.0 * np.abs(Hm[i, s, 0])
    log(f" {fq:5d}  S{s + 1}     | {amp_t:12.4f}    | {amp_m:7.4f} | {100 * abs(amp_m - amp_t) / amp_t:6.2f}% |"
        f" {pe:8.4f}  | {perr:5.2f}%")

# ---------------------------------------------------------------- units
log("\nSensitivity of the 99.9 % rank rule to units/normalisation (paper: unstated)")
log(" option                       |  p  |  r  | stable | max|lam| | mean eps_f (6 pairs)")
for label, kw in [('SI (m, strain, V)', dict(normalize=None)),
                  ('mm, microstrain, V', dict(normalize=None, unit_x=1e3, unit_eps=1e6)),
                  ('0.1 mm, microstrain, V', dict(normalize=None, unit_x=1e4, unit_eps=1e6)),
                  ('10 mm, microstrain, V', dict(normalize=None, unit_x=1e2, unit_eps=1e6)),
                  ('group RMS (default here)', dict(normalize='group')),
                  ('channel RMS', dict(normalize='channel'))]:
    Mi = SdDMDc(out['w'], out['v'] / T.s_sens, u, fs, s_sens=T.s_sens, **kw)
    if Mi.r < 1:
        log(f" {label:28s} | {Mi.p:3d} | {Mi.r:3d} |   --   |   --     | model empty")
        continue
    lmax = np.max(np.abs(np.linalg.eigvals(Mi.A)))
    stable = lmax < 1
    if stable:
        vi = Mi.simulate(u)
        efs = [rel_errors(out['v'][j * n:(j + 1) * n, s], vi[j * n:(j + 1) * n, s])[1]
               for j in range(3) for s in range(2)]
        ef = f"{np.mean(efs):.2f}%"
    else:
        ef = "unstable"
    log(f" {label:28s} | {Mi.p:3d} | {Mi.r:3d} | {str(stable):6s} | {lmax:.6f} | {ef}")

np.savez(ROM_FILE, A=M.A, B=M.B, Cv=M.Cv, Cx=M.Cx, p=M.p, r=M.r, fs=fs)
log(f"\nROM saved to {ROM_FILE}")
