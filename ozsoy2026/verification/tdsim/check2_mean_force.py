"""Check 2: mean and instantaneous cutting force of src/tdsim.c.

In a stable cut the steady state is tooth-periodic, so x(t) - x(t-T) -> 0 and
the chip is the static one, fz sin(phi).  The mean force on the workpiece over
whole revolutions must then equal the analytic average for half-immersion down
milling (Altintas 2012, Eq. 2.x, integrated from pi/2 to pi):

    Fy_mean(workpiece) = -(N a fz / 2 pi) * (Kt pi/4 + Kr/2)

and, since Ga(0) = 0, the mean actuator force is zero and mean x = Fy_mean / k.
The instantaneous force waveform over the last revolution is also compared
with a fine (4000-slice) quadrature of the static chip force, with the helix
lag in both directions, and with a helix-free run that shows the down-milling
entry/exit pattern (full chip at entry pi/2, zero chip at exit pi).
Writes check2_mean_force.csv / .json and check2_waveform.png.
"""
import json
import common
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ref_sim as R
import params as P
import tdsim

rows = []
CASES = [(1000, 0.5e-3, 0.0), (2000, 1.0e-3, 0.0), (2600, 0.8e-3, 0.0),
         (1300, 3.0e-3, 600.0), (3000, 8.0e-3, 1200.0), (650, 6.0e-3, 0.0)]
NREV = 120
for rpm, a, g in CASES:
    for spr in (512, 2048, 8192):
        nrev = NREV if spr == 512 else max(40, NREV * 512 // spr)
        r = tdsim.simulate(rpm, a, g, store=True, nrev=nrev, spr=spr)
        last = slice(len(r["x"]) - 1 - 10 * spr, len(r["x"]) - 1)   # last 10 revs
        fm = float(np.mean(r["fcut"][last]))
        fa = float(np.mean(r["fact"][last]))
        xm = float(np.mean(r["x"][last]))
        # residual regeneration in steady state (once-per-tooth difference)
        nd = spr // P.N_TEETH
        x = r["x"]
        dxt = float(np.max(np.abs(x[last][nd:] - x[last][:-nd])))
        fan = R.mean_force_analytic(a)
        rows.append(dict(rpm=rpm, a_mm=a * 1e3, g=g, spr=spr, nrev=nrev,
                         mean_fcut_sim=fm, mean_fcut_analytic=fan,
                         rel_err=(fm - fan) / abs(fan), mean_fact=fa,
                         mean_x_um=xm * 1e6, fmean_over_k_um=fan / P.K * 1e6,
                         steady_dx_tooth_um=dxt * 1e6))
        print(rows[-1])

keys = list(rows[0].keys())
with open(common.out("check2_mean_force.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r[k]) for k in keys) + "\n")


# ---- instantaneous waveform vs fine quadrature ----------------------------
def static_force(theta, a, lag_sign=-1.0, nfine=4000, helix=P.HELIX):
    z = (np.arange(nfine) + 0.5) * a / nfine
    dz = a / nfine
    out = np.zeros_like(theta)
    for j in range(P.N_TEETH):
        phi = np.mod(theta[:, None] + 2 * np.pi * j / P.N_TEETH
                     + lag_sign * 2 * z[None, :] * np.tan(helix) / P.D_TOOL, 2 * np.pi)
        ins = (phi >= np.pi / 2) & (phi <= np.pi)
        h = np.where(ins, P.FZ * np.sin(phi), 0.0)
        out -= np.sum((P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * h * dz, axis=1)
    return out


wave = {}
rpm, a, g = 2000, 1.0e-3, 0.0
r = tdsim.simulate(rpm, a, g, store=True, nrev=NREV)
spr = P.STEPS_PER_REV
f_last = r["fcut"][-1 - spr:-1]
theta = 2 * np.pi * np.arange(spr) / spr          # step n = k*spr + s -> angle 2 pi s/spr
f_lag = static_force(theta, a, -1.0)
f_lead = static_force(theta, a, +1.0)
wave["rms_rel_diff_vs_lag_quadrature"] = float(np.sqrt(np.mean((f_last - f_lag) ** 2))
                                               / np.sqrt(np.mean(f_lag ** 2)))
wave["rms_rel_diff_vs_lead_quadrature"] = float(np.sqrt(np.mean((f_last - f_lead) ** 2))
                                                / np.sqrt(np.mean(f_lead ** 2)))
wave["peak_sim"] = float(f_last.min())
wave["peak_quadrature"] = float(f_lag.min())

# helix-free run through the C library directly (one layer): down-milling pattern
nrev0 = 60
bufs = [np.zeros(spr * nrev0 + 1) for _ in range(5)]
summ = np.zeros(5)
tdsim._lib.simulate(float(rpm), float(a), 0.0, P.M, P.C, P.K, P.GA_GAIN, P.GA_A1, P.GA_A0,
                    P.K_TC, P.K_RC, P.FZ, P.D_TOOL, P.N_TEETH, 0.0,
                    tdsim.PHI_ST, tdsim.PHI_EX, spr, nrev0, 1, 2, 0.0, 1, *bufs, summ)
f0 = bufs[3][-1 - spr:-1]
f0_exact = static_force(theta, a, -1.0, nfine=1, helix=0.0)
wave["no_helix_max_abs_diff_N"] = float(np.max(np.abs(f0 - f0_exact)))
wave["no_helix_force_at_entry_pi/2_N"] = float(f0[spr // 4])
wave["no_helix_expected_entry_-a*fz*Kt_N"] = float(-a * P.FZ * P.K_TC)
wave["no_helix_force_just_before_entry_N"] = float(f0[spr // 4 - 1])
wave["no_helix_force_just_before_exit_N"] = float(f0[spr // 2 - 1])
print(json.dumps(wave, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
deg = np.degrees(theta)
ax[0].plot(deg, f_last, "k-", lw=1.5, label="tdsim (30 layers), last rev")
ax[0].plot(deg, f_lag, "C1--", lw=1, label="quadrature, helix lag")
ax[0].plot(deg, f_lead, "C2:", lw=1, label="quadrature, helix lead")
ax[0].set_xlabel("spindle angle (deg)"); ax[0].set_ylabel("force on workpiece y (N)")
ax[0].set_title(f"{rpm} rpm, a = {a*1e3:.1f} mm, steady state", fontsize=9)
ax[0].legend(fontsize=7); ax[0].set_xlim(0, 180)
ax[1].plot(deg, f0, "k-", lw=1.5, label="tdsim, helix 0, 1 layer")
ax[1].plot(deg, f0_exact, "C1--", lw=1, label="-a fz (Kt sin^2 - Kr sin cos)")
for v in (90, 180):
    ax[1].axvline(v, color="0.6", lw=0.6)
ax[1].set_xlabel("spindle angle (deg)"); ax[1].set_xlim(0, 360)
ax[1].set_title("helix-free: entry at 90 deg (full chip), exit at 180 deg", fontsize=9)
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(common.out("check2_waveform.png"), dpi=130)

with open(common.out("check2_mean_force.json"), "w") as f:
    json.dump(dict(mean_force=rows, waveform=wave,
                   max_abs_rel_err_spr512=max(abs(r["rel_err"]) for r in rows if r["spr"] == 512),
                   max_abs_rel_err_spr8192=max(abs(r["rel_err"]) for r in rows if r["spr"] == 8192)),
              f, indent=1, default=float)
