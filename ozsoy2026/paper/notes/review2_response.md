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
| 5 | Numerical checks (eight items) and two claims | Done, except C3 with M = 64 (memory; see below) | Sec. 4.7, 5.1, 4.6, 5.7; Supplement S4, S5 |
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
- Level (i), new result: Proposition 1 (circle criterion for the time-varying sector [0, 1] of the
  clipped deviation) gives global attractivity of the forced orbit below a_circ, whose median over the
  251 speeds is 0.967 a_lim(0) (below 0.9 a_lim(0) at 78 speeds, down to 0.22 at 1940 rpm, mostly where
  the forced command reaches the clip); its frequency condition is checked numerically on a refined grid
  (`circle_criterion.py`, `circle_gainsweep_orbit.py`; with the orbit-limited correction where the forced
  command touches 9 V).
- Contact loss as a second sector condition, the review's suggestion (`contact_sector.py`, Sec. 5.3,
  Eq. lmi-dec2): the rectified chip is a sector [0, theta] nonlinearity on the validity set (theta =
  0.17-0.23 at the 12 points examined, 0.14 averaged over the arc). With it the certified kick shrinks
  by 6-15 % at 0.5 and 0.9 a_lim; at 0.98 a_lim no such certificate exists, because the sector condition
  for the unclipped loop requires theta < theta_c = 0.05-0.09 (KYP). Stated in Sec. 5.3 and the Discussion.
- Collocation convergence (`certs_checks.py nconv`, merged by `certs_nconv_summary.py` into `certs_nconv.json`):
  at 11 points (four speeds, 0.3-0.98 a_lim) doubling N changes the C2 certified kick by -5.0 to +0.6 % at
  ten; at 2500 rpm and 0.98 a_lim no candidate of the 2N = 40 solve passed the independent check, while 24,
  30 and 36 nodes agree with 20 to 0.6 % (`certs_nconv_extra.py`): conditioning of the larger problem, not a
  loss of the certificate. Appendix B, Table (nconv); Sec. 5.7; Discussion.
- Level (iii), C3 off-axis with full surface memory (`certs_checks.py offaxis`): C3 re-solved at the seven
  points where it exists at 0.9 and 0.98 a_lim (sizes within 0.2 % of the stored ones), and the boundary of
  its tube applied in the simulator with the full surface memory: 0 of 504 runs chattered (0 of 504 also
  with the last four passes, as in the original campaign). Sec. 5.7 and Discussion.

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
critical kick under uncertainty (`mc_kick.py`, `mc_kick_stats.py`; the first 25 samples at each of the four
speeds, 0.9 of the nominal a_lim, sampled runout, clip and loop delay in the simulator): one or two
samples per speed did not settle (linearly unstable at the sampled parameters, or, at 2870 rpm, forced
command at the clip); in 92 of the other 93 the worst-phase critical kick was 0.39-2.04 times its nominal
value (medians 0.81-1.16) and below half of it in three, and in one no kick up to 3 m/s caused chatter.
The kick follows the margin to the sampled linear limit (Spearman 0.64-0.90) more closely than the
residual authority (0.21-0.43), except at 2870 rpm (0.67 against 0.74). 95 % upper confidence bound on
the probability of not settling: 18-23 %.
Stated as frequencies over an assumed distribution, explicitly not a guarantee.

**3.6 Saturation types.** Sec. 2.1 distinguishes voltage, force and stroke limits, states that only the
voltage clip is modelled, that 27 N enters only as `V_max = F_max/K_a` and is not a strict force limit
(|G_a| peaks at 3.4 K_a near 8.6 Hz; 29.5 N in step entries at 9 V, 26.6 N at 8.1 V), and that stroke and
current limits are not modelled. Basin at 8.1 V against 9 V (`protocol_clip_level.py`, Sec. 7.2): the
residual authority at 0.9 a_lim falls to 0.68-0.87 of its value and the worst-phase critical kick to
0.76-0.88, close to the describing-function prediction 0.78-0.89; at 2500 rpm the forced command at a_lim
alone exceeds 8.1 V, so the basin closes before the limit.

**4. Delayed feedback.** Condensed from 3600 to about 900 words as a case study whose message is the
negative result: with g2 = 2000 V s/m the linear limit rises x1.78 (x1.31 with the depth capped at 2D) but
the step-entry limit only x1.03; with g2 = 500 V s/m added at the same forced command the simulated basin
widens x1.14; with assumed higher modes the delayed design tolerates 100 us of loop delay against 300 us
for DVF (on a grid of computed delays); tables, the certified gain map, the validation and the robustness study
moved to Supplement S7.

**5. Numerics.** (Sec. 4.7, 5.7 and Supplement S4.5, S5)
- Eq. (14), 80-depth sampling (`protocol_eq14_sampling.py`): 2000 depths find the maximum larger by at
  most 0.12 % (14 mN); this would lower the gain by at most 0.44 V s/m and a_lim by at most 0.015 mm at the
  94 speeds where the authority binds (Sec. 3.2).
- Kick phases (`protocol_phases.py`): 64 phases per tooth period at 0.98 a_lim (48 new phases at each of
  the four speeds): the worst-phase minimum falls by 1.1 % at 1090 rpm (already reached with 32 phases)
  and is unchanged at the other three speeds; the spread over the phases is at most 15.1 %.
- Kick after convergence (`protocol_kick_timing.py`): the face entry leaves at revolution 100 a deviation
  of at most 0.85 nm and 0.73 um/s; kicking at revolution 300 changes the critical kick at one of 16
  speed-phase pairs by 1.1 % (one bisection step) and its minimum over the phases at none.
- Band holes (`protocol_band_holes.py`, depth grid 0.05:0.05:0.95 and 0.98 a_lim at nine speeds, 2028 runs):
  at all nine the lowest chattering grid depth lies next to the bisected edge and nothing below it
  chatters; at eight the chattering depths form one interval up to 0.98 a_lim, but at 520 rpm there is a
  hole: chatter at 0.30-0.55 and 0.95-0.98 a_lim, none at 0.60-0.90 a_lim. The text no longer claims that
  the band has no holes: the mapped edge is the lowest depth at which a kick of the protocol tips the cut,
  and the band above it need not be unsafe throughout (Sec. 4.3, 4.7; Supplement S4).
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
scheduling; 7 Delayed feedback and robustness; 8 Discussion and conclusions. Sections 1-8 occupy about 41
pages of the 12-pt preprint (title page, then pp. 3-43), against the review's 39, after the
additions the review asked for (bistability evidence, contact loss, basin-aware chart, Monte Carlo); the appendices (A
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

## Independent check of the revised text

Before the final build, every number and claim of the revised manuscript, supplement and cover letter
was checked against the stored result named in its `% src:` comment by five independent readers (one per
part of the paper), and each reported problem was re-checked by a second reader trying to refute it. The
corrections that followed:
- Scope of claims: the circle-criterion depth is stated as a median over speeds (not "close to the
  uncontrolled limit"), "proves" is qualified where a frequency condition is checked numerically, the
  bistability near the limit is stated for the four speeds studied in detail, the basin-aware validation
  for the six simulated speeds, the contact-loss extension for the averaged model away from the limit.
- Basin-aware chart: the certificate uses the full clip, so the voltage margin protects the forced
  command but not the certificate (stated); a_rob carries no explicit depth margin (median 0.97, 0.97 and
  0.91 a_lim(g_rob)), unlike a_chk (0.9); the tie-break rule for g_rob; only grid depths are certified;
  "worst phase" in the validation is the smallest of four phases; validation speeds in the supplement
  (1125, 1375, 2875 rpm, not 1090, 1420, 2870); reduced-order overstatement up to 7.0 % at the 64
  re-certified points.
- Section 4: the 29 finite-basin points are in Fig. (basin), not in Table (basin); figures renumbered in
  order of citation; the entry-map figure regenerated (legend "unsafe band"); classifier bias stated as
  at least 4.4 % at 2000 rpm and up to about 3 % at 2040-2480 rpm; the unclipped cut is not called
  bistable (its chatter after 0.78-3 m/s kicks was not checked for boundedness); protocol-check wording
  (kick timing tested at four phases).
- Section 5 and supplement: failure frequencies of the circle condition 124-196 Hz (not 124-129 Hz);
  refinement changes of mu below 1e-5 (not 1e-6); 39 scan depths; gain sweep with the orbit limit (at
  1420 rpm and g = 3000 the forced orbit clips at 0.40 a_lim(0)); symbols made unique (L_0, eta, the
  Petersen multiplier).
- Section 7: the basin factor 1.14 belongs to g2 = 500 V s/m, the limits to g2 = 2000; loop-delay
  tolerances stated on their grid; runout at which the delayed design's command reaches 9 V (1.0-2.4 um)
  compared with DVF's (1.1-4.5 um); Monte Carlo clip frequencies use the worst of eight runout phases; at
  2500 rpm an 8.1 V clip is already reached by the forced orbit at 0.27-0.46 a_lim
  (`clip81_forced_command.py`).
- Two reported problems were refuted on re-checking (the definitions of u0 and of the residual authority
  are consistent; the 0.06 mm chatter amplitude in the supplement is correct).
