# Response to the second external review (pre-submission, "Major Revision", 7.5/10)

The review rated the manuscript "Actuator saturation and bistability in active chatter control of milling:
basins of attraction, certified inner estimates and delayed feedback" (62 pp. + 15 pp. supplement). This
file lists every point of the review, what was done, where it is in the revised manuscript and the
numbers behind it. Each new number comes from a script under `../../verification/paper/`; its file and
key are given in the `% src:` comment next to the sentence in the LaTeX source.

Revised title: *Beyond linear stability lobes: saturation-induced finite-amplitude instability and
basin-aware gain scheduling in actively damped milling* (the reviewer's first title, with the basin-aware
scheduling of point 3.4 in place of "basin estimation").

| Point | Request | Status | Where |
|---|---|---|---|
| 2 | Central research question | Done | Introduction, para. 4 |
| 3.1 | Second validation on the published data of Ozsoy et al. 2025 | **Not done** (full text not accessible from this environment) | Discussion, Limitations; README to-do |
| 3.2 | Show a second attractor; terminology | Done | Sec. 4 (definitions), 4.5, Fig. (attractors) |
| 3.3 | Three model levels; contact loss as a second sector | Done | Sec. 5 intro, 5.1, 5.3, 5.8 |
| 3.4 | Basin-aware gain scheduling, basin-aware stability chart | Done | Sec. 6 |
| 3.5 | Monte Carlo robustness | Done | Sec. 7.2, Table (MC) |
| 3.6 | Voltage, force and stroke saturation; 8.1 V vs 9 V basin | Done | Sec. 2.1, 7.2 |
| 4 | Delayed feedback as a short case study | Done | Sec. 7.1; details in Supplement S7 |
| 5 | Numerical checks (eight items) and two claims | Done | Sec. 4.8, 5.1, 4.6, Supplement |
| 6 | Research gap relative to Ozsoy 2025/2026 | Done | Introduction, paras. 2 and 4 |
| 7 | Restructure to eight sections, about 39 pages | Done | whole manuscript |
| 8 | Title | Done | front matter |
| 9 | Abstract | Done | front matter |
| 10 | Journal choice | Discussed below | this file |
| 11 | Author data, repository URL, AI declaration | For the authors | README to-do |
| 12 | No LQG/H-infinity/NMPC | Followed | — |

## Point by point

**2. Research question.** The introduction now states: *how much disturbance can a linearly stable,
actively damped milling cut tolerate before actuator saturation triggers chatter, and how should the gain
schedule account for it?* The answer is organised around the residual authority `V_max - b_max`.

**3.1 Second validation (Ozsoy et al. 2025, MSSP 224:111942).** Not done. The open-access PDF could not
be retrieved from this computing environment, and the review's suggestion requires its figures (6, 10,
12) to be digitised and its set-up to be re-modelled with its own parameters. The revised manuscript says
that these data were not used (Discussion, Limitations) and proposes them as a first check of the forced
command (Discussion, last paragraph). Once the PDF is available, the steps are: rebuild the robot-assisted
model from its parameters, compute the forced command and the residual authority at the published cutting
conditions, compare the predicted saturation zones with the measured ones, and check whether chatter
below the linear limit coincides with a small residual authority.

**3.2 Bistability.** Sec. 4 now defines *finite-amplitude instability* (linearly stable forced orbit, a
finite perturbation ends in chatter) and *bistability* (in addition, the chatter is a bounded attractor);
the band mapped over 126 speeds is called the *unsafe band*. New analysis (`attractors_kick.py`): at all
29 points with a finite basin, kicks of 2, 2.5 and 3 times the critical kick, followed for 2000
revolutions with full surface memory, ended in bounded, statistically stationary clipped chatter (87 of 87
runs; none diverged, none returned); continued to 4000 revolutions the statistics change by < 0.13 %.
PEND: bifurcation diagrams against depth, Poincaré sections and spectra (`attractors_bifurcation.py`,
figure).

**3.3 Certificate limits.** Sec. 5 now separates (i) the averaged delay equation, (ii) its
finite-dimensional collocation models and (iii) the simulator, and uses "numerically verified inner
estimates for finite-dimensional collocation models" throughout. New rigorous result for level (i):
Proposition 1 (circle criterion for the time-varying sector [0, 1] of the clipped deviation) proves global
attractivity of the forced orbit below a_circ, 0.967 a_lim(0) at the median speed of the scheduled design
(`circle_criterion.py`). Contact loss as a second sector condition (the review's suggestion), PEND:
`contact_sector.py`.

**3.4 Basin-aware gain scheduling.** New Sec. 6, Eqs. (admissible) and (arob), exactly the review's
formulation with every shallower depth also required, rho = 0.1, a_phys = 2D = 32 mm, beta_req = 1, 2,
5 mm/s, 24 speeds, gains 0-3000 V s/m. PEND: results, chart, simulated validation (`basin_aware.py`).

**3.5 Monte Carlo.** New Sec. 7.2 and Table (MC) (`mc_closed_form.py`, 4000 samples per case, the
review's parameter ranges and grid). At 0.9 a_lim the sampled cut is linearly unstable in 8-20 % of the
samples and its forced orbit reaches the clip in up to 9 %; at half the limit the forced orbit reaches the
clip in 69 and 50 % of the samples at 2500 and 2870 rpm (runout dominates). Stated as frequencies over an
assumed distribution, not a guarantee. PEND: simulated critical kick under uncertainty (`mc_kick.py`).

**3.6 Saturation types.** Sec. 2.1 now distinguishes voltage, force and stroke limits, states that only
the voltage clip is modelled, that 27 N enters only as `V_max = F_max/K_a` and is not a strict force limit
(|G_a| peaks at 3.4 K_a near 8.6 Hz; 29.5 N in step entries), and that stroke and current limits are not
modelled. PEND: basin at 8.1 V against 9 V (`protocol_clip_level.py`).

**4. Delayed feedback.** Condensed from 3600 to about 900 words as a case study whose message is the
negative result (linear limit x1.78, x1.31 below 2D, step-entry limit x1.03, basin x1.14, loop delay
tolerance a third of DVF's); tables, the certified gain map, the validation and the robustness study
moved to Supplement S7.

**5. Numerics.** (Sec. 4.8 and Supplement S4)
- Eq. (14), 80-depth sampling: PEND (`protocol_eq14_sampling.py`).
- 16 phases vs 64: PEND (`protocol_phases.py`).
- Kick after convergence: PEND (`protocol_kick_timing.py`).
- Band holes (depth grid instead of bisection): PEND (`protocol_band_holes.py`).
- 300-revolution window near the limit: PEND (`protocol_window.py`).
- Collocation convergence at more points: PEND (certs agent).
- C3 at 32/64 segments: M = 8, 16, 32 already in the supplement; M = 64 PEND (certs agent).
- C3 off-axis with full memory: PEND (certs agent).
- Sec. 4.3 claim (stable below a_lim(0) with full saturation): now proved for the averaged model below
  a_circ (Proposition 1) and described as a describing-function argument otherwise.
- Eq. (18): now presented as a describing-function prediction, "not a proven property of the basin at the
  limit itself".

**6. Research gap.** Introduction: Ozsoy et al. 2025 and 2026 concern the saturation caused by the cut
that the design intends; this paper asks how much disturbance a cut that does not saturate can absorb
before saturation sets in, and relates the residual authority, the critical disturbance and the basin.

**7. Structure.** 1 Introduction; 2 Model and experimental benchmark; 3 Actuator authority and the forced
orbit; 4 Saturation-induced finite-amplitude instability; 5 Basin estimation; 6 Basin-aware gain
scheduling; 7 Delayed feedback and robustness; 8 Discussion and conclusions. PEND: page count.

**10. Journal.** Mechanical Systems and Signal Processing states in its scope that theoretical nonlinear
dynamics without experimental validation is out of scope; the study is computational, validated only
against published cutting tests, so a desk rejection there is a real risk. Journal of Sound and Vibration
and Nonlinear Dynamics accept computational nonlinear-dynamics and control studies; International Journal
of Mechanical Sciences and Journal of Manufacturing Processes (where the base paper appeared) are further
options. The manuscript keeps the Elsevier `elsarticle` format, which serves MSSP, JSV, IJMS and JMP;
Nonlinear Dynamics (Springer) would need its own template. Whether a journal is "A" in an institution's
list is a separate question from its quartile.

**11. Checklist.** Author names, affiliations, CRediT, funding, repository URL and the name of the AI tool
remain placeholders for the authors (paper/README.md).
