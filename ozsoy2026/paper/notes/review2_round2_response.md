# Response to the second round of the external review (8.7/10, "limited Major Revision")

The second round rated the revised manuscript *Beyond linear stability lobes: saturation-induced
finite-amplitude instability and basin-aware gain scheduling in actively damped milling* and listed
seven items before submission (Section 8 of the report). This file answers each one: what was
computed, where it is in the manuscript, and which script and JSON hold the numbers
(`../../verification/paper/`; every number in the LaTeX source has a `% src:` comment next to it).
As the review advised, no new theory and no new control law were added; the work went into
verification and comparison.

| Item | Request | Status | Where |
|---|---|---|---|
| 4.1 | Like-for-like comparison: same gain range, same metrics, certificates for the references too | Done | Sec. 6.3, Table 6, Fig. 8b; Supplement S4.3 |
| 4.2 | Monte Carlo of the new design against the reference on the same samples | Done | Sec. 6.4; Supplement S4.5, Table S6 |
| 4.3 | Abrupt-entry curve in Fig. 8 | Done | Fig. 8a, b; Sec. 6.4 |
| 4.4 | a_phys renamed a_cap (assumed axial-depth cap); sensitivity to 10, 16, 32 mm | Done | Eq. (31), Sec. 6.1, 6.3; Supplement S4.4, Table S5 |
| 4.5 | Circle criterion on the continuous frequency axis; grid optimum of Eq. (32) | Done | Sec. 5.1; Supplement S3.1; Sec. 6.1 |
| 5 | Second validation on Ozsoy et al. 2025 | **Not done**: the PDF could not be obtained from this computing environment | Sec. 8 (Limitations) |
| 6 | Editorial items | Done, except the authors' own data | see below |
| 7 | No further contributions | Followed | — |
| 3.3 | Keep "bifurcation-type diagrams" | Kept | Sec. 4.5 |
| 3.4 | Eq. (30) covers only the memoryless part of contact loss | Kept and stated | Sec. 5.3, 8 |

## 4.1 Fairness of the comparison in Table 6

*Designs.* At the 21 speeds of the chart, the published condition and the authority rule with and without
the voltage margin were recomputed with the gain range of the chart, 0 <= g <= 3000 V s/m, and the former
reference (g <= 1500 V s/m) was kept for continuity; all operate at the depth of Sec. 3, the checked depth
a_chk = min{0.9 a_lim, a_TD} (`fair_designs.py`):
- the published condition, Eq. (13), reproduced as in Sec. 3.1 (step-entry demand at a_lim, its original constraint);
- the authority rule, Eq. (14), with no voltage margin (steady command up to 9 V);
- the authority rule with the 10 % margin of the chart (8.1 V);
- the former reference of the manuscript (g <= 1500 V s/m), kept for continuity;
- the basin-aware chart for 1, 2 and 5 mm/s.

*Same metrics for all* (`fair_compare.py`, `table6.py`): operating depth, step-entry depth, the depth up
to which the design's own gain meets the chart's conditions (`ref_under_ba.py`; `ref_under_ba_tol.py` for
the authority gain with the 10 mV tolerance), the rigorous bound on the forced command, the certified kick
(C2, default collocation order) at the operating depth, the simulated critical kick (smallest of four
phases) at the operating depth and the step entry; these are in the new Table 6. The face entry settles at
the operating depth of every design at all 21 speeds (`fair_compare.json`); the manuscript reports it for
the chart.

*Results.*
- With g <= 3000 V s/m the authority rule reaches 17.1 mm (no margin) and 16.8 mm (10 % margin) on
  average, instead of 15.5 mm with g <= 1500: part of the former gap was the gain range, as the review
  suspected.
- Judged by the chart's own conditions, the reference gains certify less, by an amount that depends on
  the requirement. The authority gain with the same voltage margin keeps every grid depth admissible up
  to 21.2, 18.2 and 9.8 mm for 1, 2 and 5 mm/s, against 21.7, 19.9 and 16.0 mm for the basin-aware gain;
  the other reference gains reach at most about 11 mm. These depths take the voltage test with a
  tolerance of 10 mV (`ref_under_ba_tol.py`): the authority rule sets its sampled steady command to
  8.1 V, and at five speeds its rigorous bound exceeds 8.1 V by 0.5-2.3 mV at one grid depth, a numerical
  tie that stopped the strict scan at 2-4 mm (strictly 17.9, 15.5 and 9.2 mm). No gain of the chart has
  such a tie, so its results do not change. (The final internal audit found this; the first version of
  this response used the strict values and overstated the chart's advantage for 1 and 2 mm/s.)
- The chart had no depth margin below the linear limit (median a_rob / a_lim 0.97, 0.97, 0.91), the
  references had 0.9. With a <= 0.9 a_lim added to the chart (`ba_kappa.py`; binding at 10-13 speeds) the
  chart gives 19.7, 18.8 and 15.1 mm, against 19.6, 17.1 and 9.4 mm for the authority gain under the
  same conditions: the choice of the gain matters little for 1 mm/s (0.1 mm), somewhat for 2 mm/s
  (1.6 mm) and much for 5 mm/s (5.7 mm), i.e. when the required disturbance is large. For a small
  requirement the chart's deeper operating points come from leaving out the entry check that the
  authority rule contains. Against the authority rule's own operating depth (16.8 mm), the chart with the
  margin is 2.9 and 2.0 mm deeper for 1 and 2 mm/s and 1.6 mm shallower for 5 mm/s.
- At their own, shallower depths the references keep larger certified kicks (authority rule with the
  margin: 7.9 mm/s median, against 4.3 and 5.0 mm/s for the chart for 1 and 2 mm/s; similar for 5 mm/s).
  What the authority rule does not check is the shallower depths its gain passes through: there its
  smallest certified kick is 2.3 mm/s (median) and reaches 5 mm/s at only 5 of 21 speeds. The simulated
  critical kick of the chart exceeds its requirement at all 21 speeds at least 1.97 times.
- The chart is a grid optimum: at five speeds between 1625 and 2375 rpm the authority gain, which lies
  between the gains the chart evaluated, is certifiable 2 mm deeper for some requirements (with the depth
  margin at 1375, 1625 and 2625 rpm). The manuscript says so.
- The phrase "certifies none" was removed everywhere; the references now have their own certificates.

## 4.2 Monte Carlo of the new design

Done with the review's protocol (`mc_designs.py`, `mc_designs_table.py`; Sec. 6.4, Supplement S4.5 and
Table S6). The first 30 samples of the +-10 % error draw of Sec. 7.2 (cutting and actuator constants,
natural frequency, damping, runout, clip 8.1-9 V, loop delay), the same at every speed, were applied at
the six validation speeds to the authority rule with the 10 % margin at its checked depth and to the
chart, with and without the depth margin, each at its own depth, and to the reference and the chart with
the margin at the common depth min{a_chk, a_rob}. A sample fails if the cut does not settle, or a kick of
beta_req at one of four phases ends in chatter (the event dv* <= beta_req if the outcome is monotone in
the kick). Pooled over the 180 sample-speed pairs (30 samples at six speeds):

| Requirement | Reference, own depth | Chart with margin, own depth | Chart without margin, own depth | Common depth: reference / chart |
|---|---|---|---|---|
| 1 mm/s | 6.7 % | 16.7 % | 39 % | 6.1 % / 7.2 % |
| 2 mm/s | 6.7 % | 11.7 % | 39 % | 6.1 % / 7.2 % |
| 5 mm/s | 6.7 % | 10.0 % | 11 % | 6.1 % / 9.4 % |

No sample failed for the reference and not for the chart (McNemar p <= 0.031 at the own depths,
approximate because the speeds share the samples; with the margin, exact sign test over the 30 samples
p = 0.002, 0.03, 0.13). Of the 69 failures of the chart with the margin, 55 are cuts whose sampled averaged
model is linearly unstable and 10 cuts that do not settle at 2875 rpm, where runout and a lower clip put
the forced command on the clip; a settled cut chattered after the kick in only 1, 1 and 2 of the 180
pairs. The answer to the review's question is therefore negative: under these errors the chart is not
more robust than the reference. Its operating points sit nearer their own linear limit (0.81-0.90 a_lim
with the depth margin, 0.86-0.996 without it, against 0.51-0.81 for the reference where its step-entry
check binds and 0.90 at 2000 rpm, where its depth margin binds and where alone it fails), and they lose the
margin to the linear limit and, at 2875 rpm, the voltage margin. For 5 mm/s the chart is on average even
shallower than the reference (14.7 against 15.9 mm at these speeds) and still fails more often. At the
common depth the rates are close. The manuscript states that the chart needs margins that cover the
errors of the model, below the linear limit and, where runout and a lower clip consume the margin of the
forced command, in the voltage, which its nominal certificate does not provide, and that these
probabilities are estimates under the sampled distribution, not guarantees.

## 4.3 Entry

Fig. 8a now shows, for each speed and requirement, the basin-aware depth (markers by the condition that
binds at the next grid depth) and, as crosses, the largest grid depth up to which the step entry from
rest also settles with the same gain (`ba_entry.py`). The caption states that this curve is verified in
the simulator, not certified. The face entry settles at every grid depth up to a_rob at all 21 speeds;
the step entry only up to 16.1, 15.8 and 15.1 mm on average (below a_rob at 14, 12 and 7 speeds), and with
the depth margin up to 15.7, 15.4 and 14.4 mm. Sec. 6.4 says plainly that the chart certifies the
settled cut, not the entry, and that with an abrupt entry the usable depth is set by saturation during
the entry: the chart's step entry settles only up to 15.1-16.1 mm (14.4-15.7 mm with the depth margin),
below the 16.8 mm of the authority rule, whose checked depth contains the same entry check. Fig. 8b compares the
step-entry depths of the chart with the checked depths of the references.

## 4.4 Depth cap

a_phys is renamed a_cap and called an assumed axial-depth cap throughout (Eq. (31), nomenclature,
Sec. 6.1, Fig. 8, Supplement). The chart was recomputed for a_cap = 10, 16 and 32 mm (`ba_cap.py`,
`cap_kicks.py`; 10 mm is the deepest published test):
- 32 mm: the stored depths are reproduced at 62 of 63 design points (the one difference is a grid
  effect of the refinement, explained in Supplement S4.4).
- 16 mm: mean depth 15.1, 15.1 and 14.0 mm, against 14.5 mm for the authority rule with the same margin
  and 10.7 mm for the published condition, capped the same way.
- 10 mm: every design except the published condition (8.9 mm) reaches the cap at every speed.
- At equal depth the chart's gain certifies larger kicks than the authority gain: median ratio 2.3 at
  10 mm (21 speeds) and 1.5 at 16 mm (16 speeds), with the same 10 mV tolerance; against the former
  reference 1.24 and 1.00; against the published gain 1.02.
The manuscript now says that most of the depth advantage lies in deep cuts that depend on the assumed
cap, and that below a cap of about 16 mm the advantage over the authority gain is in the certified
disturbance, not in depth (against the published gain the kicks are tied within the 12 % difference between the collocation
orders, 1.02 at 10 mm and 1.03 at 16 mm; against the former reference they are tied at 16 mm, 1.00, but
1.24 times larger at 10 mm).

## 4.5 Rigour of the circle criterion and of Eq. (32)

The frequency condition of Proposition 1 is now proved on the whole frequency axis, not only on a grid
(`circle_interval.py`): Re[1 + L(i w)] is rewritten exactly as N / |B|^2 with
N = |B + h/2|^2 - |h|^2 / 4 (h = -i g K_a w^3 / A), which removes the cancellation near the poles, and is bounded from below by
interval arithmetic with outward rounding on adaptively bisected boxes in frequency and depth, at the
scheduled gains, for every depth up to (1 - 1e-3) a_mu, where a_mu is the depth at which the frequency
condition stops holding (a_mu = a_circ except at 44 speeds, 1530-1990 rpm, where the forced orbit reaches
the clip first and sets a_circ); above 5 kHz an analytic bound |L| <= 0.0071 applies. The condition is
proved at all 251 speeds (785-5500 box evaluations to positivity, 1200-30 000 with the tightening; no
undecided box; smallest certified lower bound 0.00098, at 950 rpm; median 0.043). At the smaller of (1 + 1e-3) a_mu and the midpoint between a_mu and a_lim(0) (the midpoint at 42 speeds) the
enclosures show it failing at the 249 speeds where a_mu < a_lim(0), so the depth where it stops holding
is located to within 1e-3. The second hypothesis, an unclipped forced orbit, rests on the closed-form
command on a dense depth grid (2000 depths), and the manuscript now says so wherever the result is
quoted (Sec. 1, 4.3, 5 and 8; the abstract attributes the proof to the frequency condition only).

Sec. 6.1 states that Eq. (32) is a grid optimum: the best among the gains evaluated by the search (a
250 V s/m grid refined by +-125 and +-62.5 V s/m around the best gain), for depths on the 2 mm grid, not a
continuous optimum; Sec. 6.3 gives the speeds at which a gain between the evaluated ones does better,
and Supplement S4.4 shows that a repeated search can return a different gain. Fig. 8 marks the grid
points and its caption says that only the grid depths are certified.

## 5 Second validation on Ozsoy et al. 2025

Not done. The open-access PDF linked in the review could not be retrieved from this computing
environment (the download was refused by the environment's policy), and only the abstract was
available. The validation the review lists (FRF of the second set-up, saturation zones, stability
regions, residual authority, basins at selected points) needs the paper's figures and parameters.
Limitations (Sec. 8) state that these measurements were not used. With the PDF available, the steps
are: rebuild the robot-assisted model from its parameters and FRF, compute the forced command and the
residual authority at the published cutting conditions, compare the predicted saturation zones and
stability regions with the measured ones, and compute the critical kick at selected points. The authors
can supply the PDF (or allow its download) to have this done.

## 6 Editorial items

| Place | Change |
|---|---|
| Abstract | "This computational study ..."; "searched on grids at 21 speeds"; the comparison stated with equal gain range and margins; no "certifies none" |
| Highlights | the guarantees restricted to their models: "Circle criterion and LMIs bound the basin of the averaged and collocation models"; "A basin-aware gain schedule certifies model depths against a required disturbance" |
| Figure 8 | grid points as markers, lines only guide the eye, caption: only the grid depths are certified; step-entry crosses; a_cap marked as assumed |
| Section 6 | restructured: 6.1 formulation, 6.2 chart, 6.3 comparison with the references (Table 6), 6.4 validation, entry and robustness |
| Section 8 | discussion interprets, conclusions state results; numbers no longer repeated in both |
| Data availability | still a placeholder: the authors must choose the archive (GitHub release or a Zenodo DOI of the repository) |
| Appendix C | the AI tool is named (Claude Code, Anthropic) with its uses; the authors must add who checked what |
| Title page | still placeholders: author names and affiliations are for the authors |

## 7 No further contributions

Followed: nothing new was added to the theory or the control law. "Bifurcation-type diagrams" is kept
(Sec. 4.5), and Sec. 5.3 and 8 keep the statement that Eq. (30) covers only the memoryless part of the
loss of contact.
