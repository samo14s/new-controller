# Verification of `src/zoa.py::a_lim` (ZOA) with an independent SDM solver

Scope: the frequency-domain stability limit `a_lim(rpm, g)` that the SADE objective
uses. Nothing in `src/` was modified. Every script here runs on its own with
`python3 <script>.py` from this folder, using 1 to 2 CPU cores for a few minutes at most.

## 1. Re-derivation of the zero-order condition (done before reading zoa.py)

Conventions: the Altintas immersion angle phi is measured clockwise from +y, the feed is +x, and the
radial unit vector at the cutting edge is (sin phi, cos phi). For half-immersion down milling, a tooth
is in the cut for phi in [pi/2, pi]. The workpiece is flexible along y (normal to feed), with
displacement w, and the tool is rigid.

* The relative tool displacement is -w. The chip is thicker when the tool centre has moved further along
  the radial direction than it had one tooth period earlier, so
  `h = fz sin(phi) - (w(t) - w(t-T)) cos(phi)` (this matches the expression used in `tdsim`).
* The force on the tool along y is `Ft sin(phi) - Fr cos(phi)`, with `Ft = Kt a h` and `Fr = Kr a h`. The force
  on the workpiece is the negative of this, which gives
  `f_w,dyn = a H(t) (w(t) - w(t-T))`, where `H(t) = sum_j g_j [Kt sin(phi_j) cos(phi_j) - Kr cos^2(phi_j)]`.
* The tooth-period average is `<H> = (N/2pi) * integral_{pi/2}^{pi} (...) dphi = (N Kt / 4pi) * alpha_yy` with
  `alpha_yy = -1 - (Kr/Kt) pi/2 = -1.428399` (Kr/Kt = 180/660).
  The two sign flips (relative displacement = -w, and force on workpiece = -force on tool) cancel. The
  1-DOF workpiece therefore has the same characteristic equation as a 1-DOF tool:
  `1 - (N Kt a / 4pi) alpha_yy (1 - e^{-i w T}) Gcl(i w) = 0`, with `Gcl = G / (1 + g s Ga G)`.
* Requiring a to be real gives `w T = eps + 2 k pi` with `eps = mod(2 arg Gcl + pi, 2pi)` and
  `a = Re Gcl / (c (1 - cos eps) |Gcl|^2)`, where `c = N Kt alpha_yy / 4pi < 0`. A positive root needs Re Gcl < 0.

Comparison with `zoa.py`: `alpha_matrix()[1,1]` = -1.428399 (identical), and the characteristic equation and
the sign of `ALPHA` are identical. `model.py` re-types the parameters from the paper. Its `Gcl` matches
`structure.Gcl` to 7e-16, and its closed-loop poles match `structure.closed_loop_poles`.

## 2. Files

| file | content |
|---|---|
| `model.py` | independent model: G, Ga (full or flat 3 N/V), Gcl, 4-state closed-loop matrix, straight-flute H, exact helical a*H (closed form in z) |
| `zoa_lobes.py` | classic lobe sweep (lobes k = 0..60, 0.05 to 1500 Hz, df = 0.004 Hz), lower envelope, with Brent polishing of the candidates within 5 % of the minimum |
| `sdm.py` | first-order semi-discretization, monodromy matrix, boundary search, Richardson extrapolation |
| `convergence.py` -> `sdm_convergence.csv` | SDM error against the number of intervals m |
| `run_compare.py` -> `compare_12speeds.csv` | 12 speeds x g in {0, 300, 800, 1500} x zeta in {1.34, 1.45} % |
| `dense_rootcheck.py` -> `dense_rootcheck.csv`, `dense_sld_z0134.json` | src against the lobe sweep on a 1-rpm grid (2501 speeds), gains 0 to 6000 |
| `spurious_scan.py` -> `spurious_scan.json/.csv` | Monte Carlo test for pole roots on the paper grid |
| `zoa_fix_demo.py` -> `zoa_fix_demo.json` | a pole guard for the per-speed scan (demonstration only, not applied to src) |
| `actuator_mode.py` -> `actuator_mode.json`, `actuator_effect.csv` | role of the 8.4 Hz actuator mode |
| `dense_sdm.py` -> `dense_sdm.csv`; `make_figures.py` | dense SDM sweeps and the figures `fig_zoa_vs_sdm.png`, `fig_rel_diff_12speeds.png` |

## 3. SDM solver

* States are `y = [w, wdot, q, qdot]`, with `qddot + 15.834 qdot + 2785.6 q = V`,
  `f_act = 3 (V - 15.834 qdot - 2785.6 q)` (= 3 qddot, which realises Eq. 3 exactly) and `V = -g wdot`.
  The DDE is `ydot = [A0 + aH(t)/m e2 e1^T] y - aH(t)/m e2 w(t-T)`, with delay T equal to the tooth period.
* First-order SDM: the tooth period is split into m intervals. A_i and B_i use the exact interval means of
  H. The delayed position is interpolated linearly between w_{i-m} and w_{i-m+1}. P_i, Phi0_i and Phi1_i
  come from one block-triangular matrix exponential. The stability criterion is the spectral radius of
  the (4+m) x (4+m) monodromy matrix.
* Cutter models:
  * straight flute (exact H; with 4 teeth at half immersion exactly one tooth is engaged at all times);
  * 45 deg helix (exact integral of H over z, 8-point Gauss in time per interval; a*H depends on a);
  * `mean`: H(t) replaced by `<H>`. This is exactly the ZOA model, so it isolates SDM numerical error.
* Convergence (`sdm_convergence.csv`): the error is O(1/m^2). With m = max(40, 40 steps per natural
  period) and Richardson extrapolation from m and 2m (`a_R = a_2m + (a_2m - a_m)/3`), the Richardson
  correction is 0.02 to 0.19 %, and the residual error is about 2e-5 (see the mean-H check below).
* Boundary search: a geometric scan in a from 0.05 mm (ratio 1.06), then bisection. No ZOA value is used to
  start the search.

## 4. Results

### 4a. ZOA against SDM, 96 cases (`compare_12speeds.csv`, `fig_rel_diff_12speeds.png`)

Values are ZOA/SDM - 1.

| comparison | range over 96 cases |
|---|---|
| SDM with time-averaged force (same model as the ZOA) | max \|rel\| = 2.0e-5, median 7.5e-7 |
| SDM with 45 deg helical cutter | -1.04 % ... -0.001 % (the ZOA is always slightly conservative; median -0.09 %) |
| SDM with straight flute, Hopf boundaries (92 cases) | -3.20 % ... -0.10 % (ZOA conservative) |
| SDM with straight flute, flip boundaries (4 cases: g = 0, 1700 and 1950 rpm, both zetas) | +5.1 % ... +5.5 % (ZOA non-conservative) |
| chatter frequency, helix SDM against ZOA | +0.001 ... +0.042 Hz |

By gain (zeta = 1.34 %; 1.45 % is within 0.03 percentage points of these):

| g [V s/m] | straight-flute Hopf | helix |
|---|---|---|
| 0    | -1.55 ... -0.10 % | -0.68 ... -0.10 % |
| 300  | -2.92 ... -0.41 % | -1.04 ... -0.03 % |
| 800  | -3.04 ... -1.17 % | -0.32 ... -0.00 % |
| 1500 | -3.20 ... -2.23 % | -0.11 ... -0.00 % |

Dense sweeps at zeta = 1.34 % (`dense_sdm.csv`, `fig_zoa_vs_sdm.png`) show the following.
* g = 0: the straight-flute SDM has a flip (period-doubling) lobe from 1700 to 1950 rpm. Its chatter frequency
  is 1.5 times the tooth frequency and its limit is 5.1 to 5.6 % below the ZOA. Elsewhere the difference is
  -2.15 ... -0.06 %. The helical SDM has no flip lobe and lies within -0.68 ... +0.01 % of the ZOA.
* g = 1500: straight flute -3.45 ... -2.22 %, helix -0.11 ... 0.00 %.

Interpretation: the half-immersion force coefficient H(t) is never zero (it runs from 0 at entry through
-420 MN/m^2 to -180 MN/m^2, with mean -300 MN/m^2). The averaged model is therefore accurate. With
the 45 deg helix, the axial pitch is (pi/2)/(2 tan(45 deg)/D) = 12.6 mm. When the depth is close to a
multiple of that pitch, H(t) becomes almost constant, and the helical SDM then reproduces the ZOA
almost exactly (for example 38.4418 against 38.4415 mm at 1950 rpm, g = 1500). For the cutter
modelled in `tdsim` (45 deg helix), the ZOA used by the optimiser is accurate to about 1 % and errs on
the conservative side.

### 4b. Missed and spurious roots of the per-speed scan (`dense_rootcheck.csv`, `spurious_scan.*`)

* Missed roots: none. On the 1-rpm grid (500 to 3000 rpm, 12 gain/damping cases including g = 3000 and 6000),
  src is never larger than the polished lobe envelope by more than 1.6e-6. The selected chatter
  frequencies agree to 3e-5 Hz.
* Spurious roots: they do occur, at tooth-passing poles. At w = 2 pi j / T the factor (1 - e^{-iwT}) vanishes, and
  Im a(w) changes sign through a pole. `zoa.py` rejects such a sign change only when |Im a| >= 0.5 |a|.
  Near the pole, `|Im a|/|a| = |cos(arg Gcl)|`, so the filter lets the pole through whenever a
  tooth harmonic falls where arg Gcl lies between -120 and -60 deg, which is the resonance band. The
  interpolated "root" is then essentially a cancellation residue: for example 29.296875, 31.25,
  39.0625, 46.875 or 62.5 mm, all exact binary fractions, with the root frequency 0.00 or 0.01 Hz from the harmonic
  and a residual of about 1 in the characteristic equation. It counts only when it is below the true limit,
  that is, near lobe peaks or at high gain.
  * At the four check gains on the 1-rpm grid there is 1 hit: zeta = 1.45 %, g = 300, 1983 rpm, 29.30 mm
    instead of 30.07 mm (-2.6 %). On the paper's 10-rpm grid there are 0 hits for these gains.
  * Monte Carlo over the paper grid with g ~ U[0, 2000] gives 3 hits in 3000 draws at zeta = 1.34 % and 1 in 3000 at 1.45 %. The
    errors are -0.3 to -3.1 %, at 1980, 2070 and 2130 rpm (zeta = 1.34 %) and 1080 rpm (zeta = 1.45 %), all multiples of 30 rpm, where the harmonic falls on the 0.01 Hz grid.
  * For larger gains the errors grow: -1.6 % at g = 3000 and up to -17 % at g = 6000.
  * The error always makes a_lim smaller, so it is conservative and isolated. Its effect on the SADE gain map is minor.
  * Fix (`zoa_fix_demo.py`): discard a sign change whose bracket contains or touches a harmonic, i.e. when
    `floor(wT/2pi)` differs between the end points or either end point is within 1e-6 of an integer harmonic.
    This removes every flagged case (agreement with the lobe sweep becomes 3e-9 to 8e-8) and changes
    nothing else (1-rpm grid: max |rel| <= 1.6e-6 for all 8 cases).
* The first lobe sweep (before polishing) showed 0.3 % differences at the k = 0 lobe tip (1953 to 1956 rpm). A
  fine brute-force check showed this was linear-interpolation error of the sweep itself. src was right,
  and the polished sweep now agrees.

### 4c. Actuator mode near 8.4 Hz (`actuator_mode.json`, `actuator_effect.csv`)

* In the chatter band, Ga/3 deviates by only +0.40 % / +1.12 deg at 129.3 Hz and +0.17 % / +0.72 deg at 200 Hz.
* Closed-loop poles without cutting at g = 1500: structural mode 128.97 Hz with zeta = 15.1 %, actuator mode
  8.42 Hz with zeta = 14.1 %. The DVF loop on its own becomes unstable only above g = 25 296 V s/m.
* Re Gcl stays positive below 60 Hz (min Re Gcl * k = 1.000 for g <= 3000 and 0.886 at g = 6000). Re Gcl < 0
  in the actuator band requires g > 24 709 V s/m, just below the loop-instability gain. So the actuator mode cannot
  create a low-frequency lobe family. A brute-force search finds 0 positive ZOA roots below 60 Hz over
  51 speeds x 4 gains, and the 0.5 to 60 Hz part of the `zoa.py` grid never sets the minimum.
* Effect on a_lim of the full Eq. (3) actuator against a flat 3 N/V actuator:
  * 12-speed table: ZOA -1.50 ... +0.45 %, SDM -1.49 ... +0.46 %. The two agree to 0.02 percentage points.
  * 5-rpm grid (ZOA): -2.9 ... +0.46 %. The largest changes are on the steep flanks of the k = 0 lobe peak (1980 to 2090 rpm).
* At the stability boundary, the SDM multiplier of the actuator mode barely moves from its no-cutting value
  e^{lambda_a T} (0.7989 -> 0.7963 at 500 rpm, g = 1500; < 1e-4 change elsewhere). The instability is always the
  structural mode.

## 5. Verdict

`zoa.a_lim` implements the correct zero-order condition, including the sign and value of alpha_yy = -1 - (Kr/Kt) pi/2.
It agrees with an SDM of the same averaged model to 2e-5 and with an independent lobe sweep to 1.6e-6. It
misses no roots. It can occasionally return a spurious pole root: rarely, always on the conservative side,
and fixable as shown above. Against the time-periodic model, the ZOA is within about 1 % (conservative) for the 45 deg
helical cutter. For a straight-flute idealisation it is within 3.5 % (conservative) at Hopf boundaries, but
5 to 6 % non-conservative inside the 1700 to 1950 rpm flip lobe at g = 0. The flip lobe disappears with the helix.
