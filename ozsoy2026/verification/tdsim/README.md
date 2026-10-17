# Verification of the time-domain simulator (src/tdsim.c, src/tdsim.py)

This is an independent check of the milling time-domain simulator used in the
reproduction of Ozsoy, Sims & Ozturk (J. Manuf. Processes 171, 2026). Nothing in
`src/` was changed. All scripts, results and figures are in this folder.

## How to run

```
cd /home/user/new-controller/ozsoy2026/verification/tdsim
python3 check1_integration.py      # ~10 s
python3 check2_mean_force.py       # ~5 s
python3 check3_stability.py        # ~15-30 min (Pool(2)); writes check3_*.csv/json/png
python3 check3b_exit_chip.py       # ~10 min
python3 check4_vmax.py             # ~4 min
python3 check5_demand_at_alim.py   # ~1 min
python3 check6_slice_coincidence.py
```

`common.py` sets the import paths and forces single-threaded BLAS. On this
shared machine the small matrix exponentials in `sd.py` ran about 100x slower
without that setting.

Supporting modules (written for this check, sharing no code with `src/tdsim*`):

| file | content |
|---|---|
| `ref_sim.py` | Independent reference simulator. Constants are re-entered from Tables 1-2 and Eq. 3 and compared with `params.py` (no mismatch). The cutting-force geometry is re-derived. Two loops: an exact-ZOH loop built from transfer functions with scipy `tf2ss` + `cont2discrete`, and a clipped nonlinear loop integrated per step with DOP853 at rtol 1e-11. |
| `sd.py` | Linear stability of the continuous-time periodic delay equation by first-order semi-discretisation (Insperger & Stepan). `H(t)` is integrated exactly over the helix. The `mean` mode replaces `H(t)` by its average, which is exactly the ZOA. |
| `detector.py` | Chatter detector that samples x once per tooth period. Two criteria, D1 and D2 (below). Also has a bisection helper. |

---

## 1. Line-by-line review of tdsim.c

| item | verdict | evidence |
|---|---|---|
| Chip thickness `h = fz sin(phi) - (x(t)-x(t-T)) cos(phi)` (l.106) | correct | Altintas convention: phi is measured clockwise from +y, and the radial unit vector is (sin phi, cos phi). The workpiece moves by x along y, so the tool moves by -x relative to it, which adds -dx cos(phi) to the chip. For phi in [pi/2, pi], cos(phi) <= 0, so a workpiece moving toward the tool thickens the chip. Substituting w = -y_tool gives exactly the tool-flexible equations, so stability matches the tool-flexible ZOA (confirmed: `sd.py` mean mode = `zoa.a_lim` to 1e-5, sec. 4). |
| Force on the workpiece `f -= (Kt sin - Kr cos) dz h` (l.107) | correct | The y force on the tool is Ft sin(phi) - Fr cos(phi) (Altintas). The workpiece gets the opposite sign. The simulated mean force is negative, i.e. the workpiece is pushed away from the tool, and matches the analytic mean (check 2). |
| Zero force for h <= 0 (l.107) | correct | Applied per slice and per tooth. |
| Helix lag `phi = th + 2 pi i/N - z tan(beta)/R` (l.88) | correct (lag, Altintas) | The steady force waveform matches a 4000-slice quadrature with lag to 0.45 % RMS, against 36 % with lead (check2_waveform.png). The lag sign cannot change stability anyway: z -> a - z maps lag onto lead plus a constant time shift. |
| Layers at mid-points `(j+0.5) dz` | fine | Midpoint rule. With 30 layers the waveform error is 0.45 % RMS at a = 1 mm. |
| Entry/exit angles, down milling, half immersion | correct | `params.immersion_angles` gives (pi/2, pi), the same as arccos(2 ae/D - 1) and pi. A helix-free run (C library called with helix 0, 1 layer) gives the full chip at entry, F = -a fz Kt = -33.0 N exactly at 90 deg, and a force that falls to 0 at 180 deg (check2_waveform.png, right; max error vs closed form 1.3e-7 N). |
| Inclusive bounds `phi >= phist && phi <= phiex` (l.94) | minor | Only matters when a slice sits exactly on a bound. At the special depths where a slice lag is an exact multiple of 2 pi/spr (e.g. a* = 5.890486225 mm), rounding includes the entry slice for some teeth and not others. This breaks the exact tooth periodicity and adds a tiny once-per-revolution forcing: the steady once-per-tooth difference stays at 5e-8 m instead of 1e-21 m. 1e-9 relative away from a* the effect is gone (check 6). This is harmless for the paper's use and does not change the boundary. |
| Delay `nd = spr / N` (l.69), `xh[n-nd]` (l.100) | correct for 512/4 | The delay is exactly one tooth period: 128 steps, and the steady once-per-tooth difference falls to ~1e-21 m. There is no check that spr is divisible by N; otherwise the delay is silently truncated. Before the first tooth period the previous surface is taken as unvibrated (xd = 0), which is consistent with entering a fresh surface. |
| Cutting force held over the step (l.109) | correct as specified, **main source of bias** | The force is evaluated from x(t_n) and held over [t_n, t_{n+1}], so the regenerative force acts about dt/2 late. At 512 steps/rev and 500 rpm that is a phase lag of w dt/2 ~ 5.5 deg at 130 Hz. It lowers the stability limit by up to ~9 % at 500-800 rpm (sec. 4). This is a property of the scheme, not a coding error. |
| RK4 on structure + actuator + DVF, nsub = 2 (l.36-47, l.124) | correct, 4th order | Error ratio nsub 1/nsub 2 = 15.8-16.0 in all 7 cases (check 1). |
| Actuator realisation `f_act = ga (V - a1 q' - a0 q)`, `q'' + a1 q' + a0 q = V` | correct | Equals 3 s^2/(s^2 + a1 s + a0). Agrees with the independent tf2ss realisation to 6.5e-5 (RK4 error; the exact-ZOH comparison agrees to 1e-12). |
| Voltage limit inside every RK4 stage (l.27) | correct | check 4. |
| Reported maxima (l.112-118) | correct | They are sampled at step starts n = 0..nsteps-1, so the final state is excluded. They equal the maxima of the stored histories to 1e-12 and the reference maxima to <= 2.7e-6 (check 1). max_demand is the unclipped abs(g xdot) (V); max_v is the applied abs(V). |
| Divergence flag `fabs(x) >= 1e-2 m` (l.126) | note | After a break the stored histories are zero, so `objective.demand(skip_revs>0)` would read zeros. Chatter in this model is normally bounded by loss of contact, so the flag only fires for cuts far above the uncontrolled limit (e.g. clipped DVF at 11.6 mm, check 4). An ordinary chatter run therefore returns a finite demand, not inf. |
| Compilation `-O3 -march=native` | info | Results are not bit-portable. For example, vmax = 2 x peak and vmax = 0 differ at 3e-15 relative, because the two `sat()` branches compile differently. |
| `malloc` without NULL check | info | Not a problem at these sizes. |

## 2. Integration accuracy (check1_integration.*)

Cases (rpm, a, g): (500, 1.5 mm, 0), (500, 3 mm, 0), (1300, 5 mm, 600), (2600, 2 mm, 0),
(3000, 10 mm, 1200), (1950, 30 mm, 1200), (800, 14 mm, 1200), each run for 20 revolutions.
Three of these are growing chatter: 500/3 mm, 2600/2 mm and 800/14 mm.

* `tdsim.simulate_exact_python` vs the independent tf2ss exact-ZOH reference:
  max relative difference **3.8e-12**. The two exact ZOH implementations agree
  to round-off.
* C RK4 (nsub = 2) vs exact ZOH: x error <= 4.5e-5 relative for all cases
  except the fast-growing 500 rpm / 3 mm chatter run (x = 278 um after 20 rev),
  where it is **1.0e-3**. f_act error <= 6.5e-5. Halving the sub-step divides
  the error by 15.8-16.0 (4th order).
* Reported `max_demand` and `max_fact` match the reference maxima to <= 2.7e-6
  relative.

## 3. Mean cutting force (check2_mean_force.*, check2_waveform.png)

For a static chip, the analytic mean y force on the workpiece over a revolution
in half-immersion down milling is

    F_mean = -(N a fz / 2 pi) (Kt pi/4 + Kr/2) = -19.365 N per mm of depth.

In a stable cut the steady state is exactly tooth-periodic (the once-per-tooth
difference falls to ~1e-21 m), so this formula must hold in the simulation:

| rpm | a (mm) | g | F_mean sim, spr 512 (N) | analytic (N) | rel. err 512 | rel. err 8192 |
|---|---|---|---|---|---|---|
| 1000 | 0.5 | 0 | -9.684 | -9.682 | -1.7e-4 | -8.6e-6 |
| 2000 | 1.0 | 0 | -19.380 | -19.365 | -7.9e-4 | +1.1e-5 |
| 2600 | 0.8 | 0 | -15.487 | -15.492 | +3.4e-4 | +3.0e-5 |
| 1300 | 3.0 | 600 | -58.233 | -58.094 | -2.4e-3 | +3.9e-6 |
| 3000 | 8.0 | 1200 | -154.940 | -154.918 | -1.4e-4 | +2.9e-5 |
| 650 | 6.0 | 0 | -116.123 | -116.189 | +5.6e-4 | +7.8e-6 |

The <= 0.24 % residual at 512 steps/rev comes from the angular grid (entry and
exit sampling) and goes away as spr grows. The mean actuator force is 0 to
round-off (Ga(0) = 0), and the mean deflection equals F_mean / k.

## 4. Time-domain stability boundary vs zoa.a_lim (check3_*, check3b_*)

### 4.1 Chatter detector (detector.py) and run length

The simulated system repeats exactly every tooth period (nd = 128 steps), so a
stable cut settles on a tooth-periodic orbit. The detector samples
x(t_k), t_k = k T, once per tooth period and examines e_k = x(t_k) - x(t_{k-1}).
For a stable cut e_k falls to round-off (~1e-21 m). Two decisions are computed:

* **D1 (decay, the main criterion).** The deviation must die out. All of the
  following are required: the per-tooth growth factor fitted to log(block RMS
  of e_k) over the second half of the run is < 1 - 2e-5; the envelope shrinks
  (max block RMS of the last 5 of 25 blocks < 0.98 x that of the first 5); the
  end RMS is below the engagement transient; and max |x(t) - x(t-T)| over the
  last 20 % is < 0.25 fz. A sustained limit cycle of any size counts as chatter.
  Runs that reach round-off are stable.
* **D2 (growth).** The vibration must not grow beyond the engagement
  transient. This is what "amplitude growth" in a run looks like, but it
  accepts small saturated limit cycles.

Bisection: start from the ZOA limit corrected for the step-hold delay (sec.
4.3), bracket with 3 % steps, bisect to 0.1 %, then check that 0.5x and
0.75x the result are stable (no lower crossing was found in any case).
`check3c_recheck.py` re-ran all 99 D1 boundaries at 0.99x and 1.01x with the
final detector: 0 inconsistencies.

**Run length.** Near the Hopf point the transient relaxes slowly, so D1 needs
long runs. The D1 boundary with 300 revolutions differs from the one with
1000 revolutions by -0.3 % to +1.2 % (mean +0.4 %; it is biased high because
slowly growing runs still look decaying). 1000 revolutions are used for spr
512 and 2048. 300 revolutions are used for spr 8192 at 500/1300/2600 rpm,
which may carry that +1 % bias. The paper's 50 revolutions are far too short
for a stability decision near the boundary.

### 4.2 Results (10 speeds x 3 gains; check3_stability.csv, check3_ratio.png, check3_sld.png)

Ratios to `zoa.a_lim`. Spr 512 is the paper's setting.

| quantity | min | max | mean |
|---|---|---|---|
| SD, H(t) replaced by its mean (= ZOA equation) / ZOA - 1 | -6e-6 | +8e-7 | 0 |
| SD exact linear periodic model, helix 45 deg / ZOA - 1 | 0.000 | +0.007 | +0.002 |
| SD with helix 0 / ZOA - 1 | -0.049 | +0.032 | +0.013 |
| ZOA with dt/2 hold delay, spr 512 / ZOA - 1 | -0.096 | -0.012 | -0.044 |
| **tdsim spr 512, D1, 1000 rev / ZOA - 1** | **-0.089** | **-0.003** | **-0.039** |
| tdsim spr 512 D1 / (ZOA + dt/2 delay, 512) - 1 | +0.000 | +0.015 | +0.006 |
| tdsim spr 2048, D1, 1000 rev / ZOA - 1 | -0.022 | +0.008 | -0.007 |
| tdsim spr 8192, D1, 300 rev / SD - 1 (9 cases) | -0.002 | +0.017 | +0.004 |
| tdsim spr 512, D2, 300 rev / tdsim spr 512 D1 - 1 | -0.004 | +0.033 | +0.011 |
| chatter frequency tdsim (spr 512) / ZOA - 1 | -0.008 | 0.000 | -0.004 |

Absolute values (mm) at four speeds:

| rpm | g | ZOA | SD (exact) | ZOA+dt/2 (512) | tdsim 512 D1 | tdsim 2048 D1 | tdsim 8192 D1 | tdsim 512 D2 |
|---|---|---|---|---|---|---|---|---|
| 500 | 0 | 1.986 | 1.989 | 1.858 | 1.868 | 1.958 | 1.984 | 1.869 |
| 1300 | 0 | 2.746 | 2.762 | 2.532 | 2.559 | 2.719 | 2.772 | 2.569 |
| 1950 | 0 | 29.157 | 29.166 | 28.361 | 28.375 | 28.966 | - | 28.455 |
| 2600 | 0 | 1.213 | 1.217 | 1.191 | 1.203 | 1.219 | 1.224 | 1.204 |
| 500 | 600 | 10.759 | 10.763 | 9.989 | 10.031 | 10.565 | 10.751 | 10.068 |
| 1300 | 600 | 7.753 | 7.776 | 7.398 | 7.443 | 7.701 | 7.806 | 7.478 |
| 2600 | 600 | 6.591 | 6.624 | 6.470 | 6.539 | 6.631 | 6.735 | 6.719 |
| 500 | 1200 | 14.551 | 14.556 | 13.323 | 13.367 | 14.235 | 14.526 | 13.492 |
| 1300 | 1200 | 12.858 | 12.858 | 12.293 | 12.321 | 12.741 | 12.923 | 12.471 |
| 2600 | 1200 | 12.793 | 12.793 | 12.532 | 12.585 | 12.780 | 12.926 | 12.773 |

All 30 cases are in check3_stability.csv, together with nrev 300, chatter
frequencies, the critical Floquet multiplier from SD, and the D2 results at
spr 2048.

### 4.3 Where the differences come from

1. **ZOA approximation: negligible here.** The semi-discretisation (`sd.py`)
   solves the linearised continuous model exactly, including the time-varying
   directional factor and the helix. It lies 0 to +0.7 % above `zoa.a_lim`.
   With H(t) replaced by its mean, it reproduces `zoa.a_lim` to 6e-6, which
   independently validates `zoa.py` (sign of alpha_yy, the factor N Kt a/4 pi,
   and the root search). The ZOA works well because the 45 deg helix spreads
   each engagement over the depth. At a = (pi/2) D/(2 tan beta) = 12.57 mm the
   lag equals the tooth pitch, the total coefficient is constant, and ZOA is
   exact (e.g. 800 rpm, g = 1200, a = 12.43 mm: SD = ZOA).
2. **Helix: matters for the time-varying part only.** Without helix the exact
   limit departs from ZOA by -4.9 % to +3.2 %. The largest change is at
   1950 rpm, g = 0, where the critical multiplier is close to -1 (period
   doubling). The ZOA itself does not depend on the helix.
3. **Step-hold of the cutting force (dominant).** tdsim evaluates the
   regenerative force at t_n and holds it for dt = 60/(rpm spr), so the force
   acts about dt/2 late. Replacing Gcl(iw) by Gcl(iw) exp(-iw dt/2) in the ZOA
   predicts tdsim's D1 limit to within 0 to +1.5 % at every speed and gain. At
   spr 512 the bias is -9 % at 800 rpm, -6 to -8 % at 500 rpm and -1 to -2 %
   at 2300-3000 rpm. It is roughly proportional to w_c dt/2, which grows at low
   speed. The bias shrinks about 4x per 4x refinement of spr:
   spr 2048 gives -2.2 to +0.8 %, and spr 8192 is within -0.2 to +1.7 % of
   the exact limit (300 revolutions). This is not a coding error; it is the
   accuracy of 512 steps/rev with a zero-order hold.
4. **Nonlinearity (loss of contact).** In down milling the static chip is
   zero at the exit angle, so h <= 0 occurs near the exit for any vibration,
   with a force loss that grows faster than the amplitude. Just above the
   linear limit the chatter therefore saturates at small amplitude, roughly
   proportional to sqrt(a - a_c). For example, at 2600 rpm, g = 600, the
   steady e_k RMS is 1.4, 8, 11 and 13 um at 0.99, 1.01, 1.02 and 1.03 x a_zoa.
   This is a supercritical Hopf bifurcation, and it is why D2 reports
   boundaries up to 3.3 % above D1. At 1950 rpm the deep cut (about 30 mm)
   makes the engagement transient itself nonlinear, and limit cycles reached
   during the transient were the reason for the extra 0.25 fz test in D1.
5. **Isolating the linear core (check3b_exit_chip.*).** The exit angle was
   moved to pi - 0.35, which leaves a 17 um static chip at the exit, and the C
   routine was called directly. With that change small vibrations never switch
   the force off. D1 and D2 then coincide (within 0.2 %; one deep-cut
   exception at 1950 rpm, spr 2048, where D2 accepts a limit cycle reached
   during the transient). The Richardson extrapolation of tdsim's D1 limit,
   a_2048 + (a_2048 - a_512)/3 (first order in dt), agrees with the exact SD
   limit for the same immersion to **-0.33 % ... +0.01 %** (12 cases,
   mean -0.11 %). The linear dynamics of tdsim (geometry, delay, structure,
   actuator, DVF) are therefore exact up to the first-order step-hold error.


## 5. Voltage limit vmax (check4_vmax.*, check4_vmax.png)

Run at 1300 rpm, g = 1200 V s/m, a = 0.9 x a_zoa = 11.57 mm. The unclipped
peak demand is 11.72 V.

* **The limit is respected.** For vmax = 2, 1.0001, 0.75, 0.5, 0.25, 0.1 and
  0.03 x peak, max |V_applied| <= vmax in every run. At every stored sample,
  V_applied equals clip(-g xdot, +-vmax) exactly (difference 0.0). `max_v`
  equals max |V_applied|, and `max_demand` stays the unclipped max |g xdot|.
* **A limit above the demand changes nothing.** The trajectory equals the
  unlimited one to <= 3e-15 relative; the two `sat()` branches round
  differently. At vmax = 1.0001 x the sampled peak the difference is 1e-6,
  because inside the RK4 stages the continuous demand slightly exceeds the
  sampled peak and the clip briefly acts there.
* **Actuator force is consistent.** It was compared with `ref_sim.ref_clipped`,
  which integrates the clipped loop with DOP853 (rtol 1e-11) and realises
  Ga(s) independently with tf2ss.
  - Case A: a = 5 mm, vmax = 0.3 x its peak, clipped on 77 % of the samples,
    6 revolutions. Errors with nsub = 2: x 6.9e-6, f_act 6.4e-5, V 6.7e-5 of
    vmax. nsub = 8 makes them 10x smaller; convergence is below 4th order
    because of the kinks at the clip.
  - Case B: growing run, vmax = 0.5 x peak, 3 revolutions. x 5.8e-5,
    f_act 6.4e-4 (nsub = 2).
* **The force limit is not 3 vmax.** Under clipping, |f_act| reaches 1.10-1.11
  x 3 vmax, because Ga(s) overshoots for a square-ish input (check4_vmax.png).
  A 9 V clip therefore does not strictly cap the force at 27 N.
* **Effect on stability** (D1, 300 rev, 1300 rpm, g = 1200):

  | vmax (V) | none | 20 | 10 | 5 | 2 | 1 | 0.5 |
  |---|---|---|---|---|---|---|---|
  | TD limit (mm) | 12.40 | 12.40 | 12.40 | 10.27 | 5.45 | 3.91 | 3.28 |

  The limit is unchanged while the steady forced demand stays below vmax.
  Below that it falls toward the uncontrolled limit (ZOA 2.75 mm; tdsim
  spr 512: 2.56 mm). With strong clipping at 11.57 mm the run diverges
  (|x| > 1 cm). Loss of contact does not bound chatter that far above the
  uncontrolled limit, so this is the only situation where `diverged` was seen
  to trigger. Case A (5 mm, heavy clipping) also ends in chatter over 300
  revolutions.


## 6. Consequence for the SADE feasibility check (check5_demand_at_alim.*)

`objective.fitness` runs tdsim at exactly a = a_lim(ZOA) for 50 revolutions
at spr 512. Because of the step-hold bias (sec. 4.3), at speeds below about
2000 rpm that depth is 1-9 % above the simulator's own limit. The run is then
a weakly chattering cut, and the peak demand depends on spr:

| rpm | g | a_lim (mm) | 'force' = 3 max\|g xdot\|, spr 512 / 2048 (N) | 'actual' max\|f_act\|, 512 / 2048 (N) | D1 at a_lim, 512 / 2048 |
|---|---|---|---|---|---|
| 1300 | 1240 | 13.22 | 47.8 / 45.1 | 47.9 / 44.2 | chatter / chatter |
| 500 | 600 | 10.76 | 30.0 / 16.4 | 30.2 / 16.1 | chatter / chatter |
| 800 | 1200 | 12.43 | 63.4 / 39.5 | 63.6 / 38.7 | chatter / chatter |
| 1300 | 600 | 7.75 | 25.4 / 12.6 | 25.6 / 13.2 | chatter / chatter |
| 2000 | 600 | 35.04 | 85.9 / 83.2 | 87.4 / 84.8 | chatter / chatter |
| 2600 | 1200 | 12.79 | 41.5 / 41.4 | 40.8 / 40.7 | chatter / stable |
| 3000 | 600 | 6.91 | 22.5 / 22.5 | 22.0 / 22.0 | stable / stable |

At low speed the spr 512 demand is up to 1.9x the spr 2048 value. It can
therefore flip the 27 N feasibility test, e.g. 500 rpm, g = 600: 30.0 N
(infeasible) vs 16.4 N (feasible). Going from 50 to 300 revolutions changes
the demand by <= 1.4 %, because the chatter saturates quickly at small
amplitude. At 2600-3000 rpm the demand comes from the forced vibration and
does not depend on spr.

For reference: with damping 1.34 % the paper's worked example (1300 rpm,
g = 1240) gives a_lim = 13.22 mm here, not the paper's 11.8 mm, and a demand
of 47.8 N ('force') / 15.9 V ('volt'), not the paper's 29 N. Reconciling this
belongs to the objective/damping checks, not to the simulator.


## 7. Summary of issues

| severity | issue | evidence | suggested fix |
|---|---|---|---|
| major | The cutting force is held over each step, which delays it by dt/2. At the paper's 512 steps/rev the simulator's stability limit is 0.3-9 % below the exact model / ZOA, worst at 500-800 rpm. This is a discretisation bias, not a coding error. | check3: TD512 = ZOA x (0.91 ... 0.997); explained to within 0-1.5 % by ZOA with exp(-i w dt/2); 4x smaller per 4x spr | Use spr >= 2048 (error <= 2.2 %) or 8192 (<= 0.2 % + nrev bias) in the feasibility runs, or a first-order hold / mid-step force predictor. |
| major (for objective.py) | Because of that bias, the feasibility run at a = a_lim(ZOA) is a weakly chattering cut below ~2000 rpm. Its demand depends on spr by up to 1.9x and can flip the 27 N test. | check5 | Evaluate the demand with finer spr, or at a depth safely inside the simulator's own limit, or from the steady (tooth-periodic) forced response. |
| minor | A run with spr not divisible by N silently truncates the delay (`nd = spr / N`). | review | Assert spr % N == 0. |
| minor | `diverged` (abs(x) > 1 cm) almost never fires for ordinary chatter, which loss of contact keeps bounded. A chatter run returns a large but finite demand, not inf. Histories are zero after a break. | check1, check4 | Only relevant if the caller relies on `diverged` to detect chatter: use a decay detector (detector.py D1). |
| info | Inclusive entry/exit tests: at measure-zero depths where a slice sits exactly on pi/2, rounding breaks tooth periodicity, leaving a 5e-8 m once-per-revolution residual. | check6 | Compare angles with a tolerance, or compute them from integer step offsets. |
| info | Down-milling loss of contact at the zero exit chip makes chatter saturate just above the limit (supercritical). Amplitude-growth criteria overestimate the limit by up to 3.3 %; a decay criterion does not. | check3 (D1 vs D2), check3b | Use a decay criterion when the simulator is used to judge stability. |
| info | With a voltage clip, abs(f_act) can exceed 3 vmax by ~11 %. | check4 | Bound the force through Ga, not 3 vmax. |
| info | `-march=native` builds are not bit-portable. The two `sat()` branches differ at 3e-15 relative. | check4 | none needed |

**Verdict.** tdsim.c implements the model as described:

* sign conventions, down-milling entry/exit, helix lag and the one-tooth delay are correct;
* RK4 is 4th order and accurate to <= 1e-3 even in growing chatter;
* the reported maxima are right;
* the voltage clip acts inside the loop and is applied exactly;
* the mean force matches the analytic value;
* its linear dynamics converge to the exact stability limit (Richardson-extrapolated limits within -0.33 % ... +0.01 % of semi-discretisation).

The one quantitative caveat is the zero-order step-hold of the cutting force
at 512 steps/rev. It lowers the time-domain stability limit by up to 9 % at
low spindle speed and makes the demand evaluated at a_lim(ZOA) depend on spr.

