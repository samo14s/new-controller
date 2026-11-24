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
| 3.1 | Second validation on the published data of Ozsoy et al. 2025 | **Not done** (full text not accessible from this environment) | Discussion, Limitations; README to-do 5 |
| 3.2 | Show a second attractor; terminology | Done | Sec. 4 (definitions), 4.5, Fig. 5 |
| 3.3 | Three model levels; contact loss as a second sector | Done | Sec. 5 intro, 5.1, 5.3, 5.7; Appendix B |
| 3.4 | Basin-aware gain scheduling, basin-aware stability chart | Done | Sec. 6, Table 6, Fig. 8; Supplement S6 |
| 3.5 | Monte Carlo robustness | Done | Sec. 7.2, Table 7; Supplement S7.1 |
| 3.6 | Voltage, force and stroke saturation; 8.1 V vs 9 V basin | Done | Sec. 2.1, 7.2 |
| 4 | Delayed feedback as a short case study | Done | Sec. 7.1; details in Supplement S7 |
| 5 | Numerical checks (eight items) and two claims | Done (see below for 64 phases and M = 64) | Sec. 4.7, 5.1, 4.6, 5.7; Supplement S4, S5 |
| 6 | Research gap relative to Ozsoy 2025/2026 | Done | Introduction, paras. 2 and 4 |
| 7 | Restructure to eight sections, about 39 pages | Done | whole manuscript |
| 8 | Title | Done | front matter |
| 9 | Abstract | Done | front matter |
| 10 | Journal choice | Discussed below | this file; README to-do 6 |
| 11 | Author data, repository URL, AI declaration | For the authors | README to-do 1-3 |
| 12 | No LQG/H-infinity/NMPC | Followed | — |

## Point by point

**2. Research question.** The introduction now states: *how much disturbance can a linearly stable,
actively damped milling cut tolerate before actuator saturation triggers chatter, and how should the gain
schedule account for it?* The answer is organised around the residual authority `V_max - b_max`.

**3.1 Second validation (Ozsoy et al. 2025, MSSP 224:111942).** Not done. The open-access PDF could not
be retrieved from this computing environment, and the review's suggestion requires its figures (6, 10,
12) to be digitised and its set-up to be re-modelled with its own parameters. The revised manuscript says
that these data were not used (Discussion, Limitations) and proposes them as a first check of the forced
command (Discussion, Towards experimental validation). Once the PDF is available, the steps are: rebuild
the robot-assisted model from its parameters, compute the forced command and the residual authority at
the published cutting conditions, compare the predicted saturation zones with the measured ones, and
check whether chatter below the linear limit coincides with a small residual authority. Section 4.6 says
that whether its frequency-domain model already contains the forced command and a threshold amplitude
was not examined.

**3.2 Bistability.** Sec. 4 defines *finite-amplitude instability* (linearly stable forced orbit, a
finite perturbation ends in chatter) and *bistability* (in addition, the chatter is a bounded attractor);
the band mapped over 126 speeds is the *unsafe band*. New results (Sec. 4.5, Fig. 5, Supplement S4.3):
- Long runs (`attractors_kick.py`): at all 29 points with a finite basin, kicks of 2, 2.5 and 3 times the
  critical kick, followed for 2000 revolutions with the full surface memory, ended in bounded, stationary
  clipped chatter (87 of 87 runs; none exceeded 1 cm, none returned); the chatter does not depend on the
  kick (rms within 0.09 %), keeps the command on the clip 89-99 % of the time at 133-153 Hz, 0.08-2.0 mm
  peak to peak; continued to 4000 revolutions its statistics change by < 0.13 %.
- Bifurcation-type diagrams (`attractors_bifurcation.py`): 28 depths from a_lim(0) to 1.05 a_lim, four
  starts, 1600 revolutions: only the forced orbit or bounded chatter. The chatter branch begins at 0.25,
  0.74, 0.31 and 0.44 a_lim (1090, 1420, 2500, 2870 rpm), within 0.15 mm of the band edge, and coexists
  with the forced orbit up to a_lim. Poincaré section: a closed curve; spectrum: 134.6 Hz with
  combination tones: quasi-periodic. Twin runs (`attractors_chaos.py`): regular, not chaotic.
- Spot check at 14 further speeds (`attractors_band_spot.py`, `attractors_abort.json`): bounded chatter at
  six; at five (700, 1020-1060 rpm) a disturbance throws the model beyond the 1 cm stop, and with the stop
  raised to 10 cm all 21 such runs return to the forced orbit after excursions of 12-30 mm. These are
  reported as "loss of attraction"/excursions, not as bistability. (This corrected an earlier reading of
  these runs as divergence.)
- Appendix A: the earlier statement that clipped chatter is chaotic was wrong and is corrected
  (quasi-periodic).

**3.3 Certificate limits.** Sec. 5 separates (i) the averaged delay equation, (ii) its finite-dimensional
collocation models and (iii) the simulator (Sec. 5.7 and Discussion), and uses "numerically verified inner
estimates for finite-dimensional collocation models" throughout.
- Level (i), new rigorous result: Proposition 1 (circle criterion for the time-varying sector [0, 1] of
  the clipped deviation) proves global attractivity of the forced orbit below a_circ, 0.967 a_lim(0) at
  the median speed (`circle_criterion.py`; with the orbit-limited correction where the forced command
  touches 9 V).
- Contact loss as a second sector condition, the review's suggestion (`contact_sector.py`, Sec. 5.3,
  Eq. lmi-dec2): the rectified chip is a sector [0, theta] nonlinearity on the validity set (theta =
  0.17-0.23 at the 12 points examined, 0.14 averaged over the arc). With it the certified kick shrinks
  by 6-15 % at 0.5 and 0.9 a_lim; at 0.98 a_lim no such certificate exists, because the sector condition
  for the unclipped loop requires theta < theta_c = 0.05-0.09 (KYP). Stated in Sec. 5.3 and the Discussion.
- Collocation convergence (`certs_checks.py nconv`): PEND{nconv summary}.
- Level (iii), C3 off-axis with full surface memory (`certs_checks.py offaxis`): PEND{offaxis summary}.

**3.4 Basin-aware gain scheduling.** New Sec. 6, Eqs. (admissible) and (arob), the review's formulation
with every shallower grid depth also required; rho = 0.1, a_phys = 2D = 32 mm, beta_req = 1, 2, 5 mm/s,
21 speeds (500:125:3000 rpm), gains 0-3000 V s/m, depths 2:2:32 mm (`basin_aware.py`). Results:
mean a_rob 21.7, 19.9, 16.0 mm against 15.5 mm for the checked depth of the authority schedule, which
certifies nothing; the linear limit binds at 15-17 of 21 speeds, the certificate at 6 for 5 mm/s; the
basin-aware gain is lower than the authority gain at 1500-2500 rpm and higher elsewhere. Validation in the
simulator at six speeds: worst-phase critical kicks >= 2.27-2.78 times beta_req, 0 of 96 kicks of exactly
beta_req chattered, the face entry settles at every design point; the step entry from rest chatters at
14, 12 and 7 of the 21 design points (the chart certifies the settled cut, not the entry). Reduced-order
scan, design points re-certified at the default order (`basin_aware_nconv.py`): no design point lowered.

**3.5 Monte Carlo.** New Sec. 7.2, Table 7 and Supplement S7.1. Closed form (`mc_closed_form.py`, 4000
samples per case, the review's ranges and grid: 4 speeds x a/a_lim = 0.5, 0.7, 0.9, 0.98): at 0.9 a_lim
the sampled cut is linearly unstable in 8-20 % of the samples and its forced orbit reaches the clip in up
to 9 %; at 0.98 a_lim the two together take 32-79 %; at half the limit the forced orbit reaches the clip in
69 and 50 % of the samples at 2500 and 2870 rpm (runout dominates; 33 and 0.03 % without runout). With
+-20 % cutting coefficients the probability of linear instability at 0.9 a_lim rises to 15-22 %. Simulated
critical kick under uncertainty (`mc_kick.py`, 25 samples x 4 speeds, full simulator): PEND{mc summary}.
Stated as frequencies over an assumed distribution, explicitly not a guarantee.

**3.6 Saturation types.** Sec. 2.1 distinguishes voltage, force and stroke limits, states that only the
voltage clip is modelled, that 27 N enters only as `V_max = F_max/K_a` and is not a strict force limit
(|G_a| peaks at 3.4 K_a near 8.6 Hz; 29.5 N in step entries at 9 V, 26.6 N at 8.1 V), and that stroke and
current limits are not modelled. Basin at 8.1 V against 9 V (`protocol_clip_level.py`, Sec. 7.2): the
residual authority at 0.9 a_lim falls to 0.68-0.87 of its value and the worst-phase critical kick to
0.76-0.88, close to the describing-function prediction 0.78-0.89; at 2500 rpm the forced command at a_lim
alone exceeds 8.1 V, so the basin closes before the limit.

**4. Delayed feedback.** Condensed from 3600 to about 900 words as a case study whose message is the
negative result (linear limit x1.78, x1.31 below 2D, step-entry limit x1.03, basin x1.14, loop-delay
tolerance a third of DVF's); tables, the certified gain map, the validation and the robustness study
moved to Supplement S7.

**5. Numerics.** (Sec. 4.7, 5.7 and Supplement S4.5, S5)
- Eq. (14), 80-depth sampling (`protocol_eq14_sampling.py`): 2000 depths find the maximum larger by at
  most 0.12 % (14 mN); this would lower the gain by at most 0.44 V s/m and a_lim by at most 0.015 mm at the
  94 speeds where the authority binds (Sec. 3.2).
- Kick phases (`protocol_phases.py`): PEND{phases summary}.
- Kick after convergence (`protocol_kick_timing.py`): the face entry leaves at revolution 100 a deviation
  of at most 0.85 nm and 0.73 um/s; kicking at revolution 300 changes the critical kick at one of 16
  speed-phase pairs by 1.1 % (one bisection step) and its minimum over the phases at none.
- Band holes (`protocol_band_holes.py`, depth grid 0.05:0.05:0.95 and 0.98 a_lim at nine speeds):
  PEND{band holes summary}.
- 300-revolution window (`protocol_window.py`): 900 revolutions give the same verdicts in all 53 runs and
  the same critical kicks at the four speeds; the window bias found for step-entry limits near the forced
  resonance does not reach the kick-based results.
- Collocation convergence at more points: see 3.3; Appendix B Table (nconv).
- C3 at 32/64 segments: M = 8, 16 and 32 are in the Supplement; the certified size grows with M (median
  ratio to C2 0.996, 1.06, 1.15), so C3 is conservative, not wrong, since every M gives a valid
  certificate. M = 64 was not run: the memory of a C3 solve roughly doubles with M (Appendix B, Table of
  settings: 4.7 GB at M = 16 with separate vertices, about twice at M = 32), so M = 64 would need about
  20 GB, more than the 15 GB of the machine used.
- C3 off-axis with full memory: see 3.3.
- Sec. 4.3 claim (stable below a_lim(0) with full saturation): now proved for the averaged model below
  a_circ (Proposition 1) and described as a describing-function argument otherwise.
- Eq. (18): now presented as a describing-function prediction, "without proving it".

**6. Research gap.** Introduction: Ozsoy et al. 2025 and 2026 concern the saturation caused by the cut
that the design intends; this paper asks how much disturbance a cut that does not saturate can absorb
before saturation sets in, and relates the residual authority, the critical disturbance and the basin.

**7. Structure.** 1 Introduction; 2 Model and experimental benchmark; 3 Actuator authority and the forced
orbit; 4 Saturation-induced finite-amplitude instability; 5 Basin estimation; 6 Basin-aware gain
scheduling; 7 Delayed feedback and robustness; 8 Discussion and conclusions. Sections 1-8 occupy about 40
pages of the 12-pt preprint (title page, then pp. 3-42), against the review's 39; the appendices (A
simulator, B LMI conditions incl. the periodic certificate, C software) and the references follow. The
protocol checks, the proof of the periodic certificate, the per-speed basin-aware table and the
delayed-feedback details are in the Supplementary Material.

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
