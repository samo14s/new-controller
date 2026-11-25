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
| 4.2 | Monte Carlo of the new design against the reference on the same samples | Done (PEND) | Sec. 6.4; Supplement S4.5 |
| 4.3 | Abrupt-entry curve in Fig. 8 | Done | Fig. 8a, b; Sec. 6.4 |
| 4.4 | a_phys renamed a_cap (assumed axial-depth cap); sensitivity to 10, 16, 32 mm | Done | Eq. (31), Sec. 6.1, 6.3; Supplement S4.4 |
| 4.5 | Circle criterion on the continuous frequency axis; grid optimum of Eq. (32) | Done | Sec. 5.1; Supplement S3.1; Sec. 6.1 |
| 5 | Second validation on Ozsoy et al. 2025 | **Not done**: the PDF could not be obtained from this computing environment | Sec. 8 (Limitations) |
| 6 | Editorial items | Done, except the authors' own data | see below |
| 7 | No further contributions | Followed | — |
| 3.3 | Keep "bifurcation-type diagrams" | Kept | Sec. 4.5 |
| 3.4 | Eq. (30) covers only the memoryless part of contact loss | Kept and stated | Sec. 5.3, 8 |

## 4.1 Fairness of the comparison in Table 6

*Designs.* At the 21 speeds of the chart, every reference was recomputed with the gain range of the
chart, 0 <= g <= 3000 V s/m, and the operating depth of Sec. 3, the checked depth
a_chk = min{0.9 a_lim, a_TD} (`fair_designs.py`):
- the published condition, Eq. (13), reproduced as in Sec. 3.3 (its original constraint);
- the authority rule, Eq. (14), with no voltage margin (steady command up to 9 V);
- the authority rule with the 10 % margin of the chart (8.1 V);
- the former reference of the manuscript (g <= 1500 V s/m), kept for continuity;
- the basin-aware chart for 1, 2 and 5 mm/s.

*Same metrics for all* (`fair_compare.py`): operating depth, step-entry depth, the depth up to which the
design's own gain meets the chart's conditions (`ref_under_ba.py`), the certified kick (C2, default
collocation order) at the operating depth, the simulated critical kick (smallest over four phases), the
rigorous bound on the forced command, and the step and face entries.

*Results.*
- With g <= 3000 V s/m the authority rule reaches 17.1 mm (no margin) and 16.8 mm (10 % margin) on
  average, instead of 15.5 mm with g <= 1500: part of the former gap was the gain range, as the review
  suspected.
- Judged by the chart's own conditions, the reference gains certify much less: the authority gain with the
  same margin keeps every grid depth admissible up to 17.9, 15.5 and 9.2 mm for 1, 2 and 5 mm/s, against
  21.7, 19.9 and 16.0 mm for the basin-aware gain; the other reference gains reach at most about 11 mm.
- The chart had no depth margin below the linear limit (median a_rob / a_lim 0.97, 0.97, 0.91), the
  references had 0.9. With a <= 0.9 a_lim added to the chart (`ba_kappa.py`; binding at 10-13 speeds) the
  chart gives 19.7, 18.8 and 15.1 mm, against 16.6, 14.6 and 8.9 mm for the authority gain under the
  same conditions. The choice of the gain, not the gain range or the margin, accounts for 3-6 mm of
  certified depth.
- PEND: certified and simulated kicks of the references at their own depths (Table 6 columns beta, dv*).
- The phrase "certifies none" was removed everywhere; the references now have their own certificates.

## 4.2 Monte Carlo of the new design

PEND (`mc_designs.py`, `mc_designs_kappa.json`, `mc_designs_chart.json`).

## 4.3 Entry

Fig. 8a now shows, for each speed and requirement, the basin-aware depth (markers by the condition that
binds at the next grid depth) and, as crosses, the largest grid depth up to which the step entry from
rest also settles with the same gain (`ba_entry.py`). The caption states that this curve is verified in
the simulator, not certified. The face entry settles at every grid depth up to a_rob at all 21 speeds;
the step entry only up to 16.1, 15.8 and 15.1 mm on average (below a_rob at 14, 12 and 7 speeds), and with
the depth margin up to 15.7, 15.4 and 14.4 mm. Sec. 6.4 says plainly that the chart certifies the
settled cut, not the entry, and that with an abrupt entry the usable depth is similar to that of the
authority rule (16.8 mm), whose checked depth contains the same entry check. Fig. 8b compares the
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
- At equal depth the chart's gain certifies larger kicks than the authority gain: median ratio 2.1 at
  10 mm and 1.4 at 16 mm; against the former reference 1.24 and 1.00; against the published gain 1.02.
The manuscript now says that most of the depth advantage lies in deep cuts that depend on the assumed
cap, and that below a cap of about 16 mm the advantage is in the certified disturbance, not in depth.

## 4.5 Rigour of the circle criterion and of Eq. (32)

The frequency condition of Proposition 1 is now proved on the whole frequency axis, not only on a grid
(`circle_interval.py`): Re[1 + L(i w)] is rewritten exactly as N / |B|^2 with
N = |B + c/2|^2 - |c|^2 / 4, which removes the cancellation near the poles, and is bounded from below by
interval arithmetic with outward rounding on adaptively bisected boxes in frequency and depth, for
every depth up to (1 - 1e-3) a_circ; above 5 kHz an analytic bound |L| <= 0.0071 applies. The condition
is proved at all 251 speeds (smallest certified lower bound 0.00098, at 950 rpm; median 0.043), with
1200-30 000 box evaluations per speed and no undecided box; at (1 + 1e-3) a_circ the enclosures show it
failing at the 249 speeds where a_circ < a_lim(0), so the computed a_circ is sharp (at 2990 and
3000 rpm it reaches a_lim(0)). The second hypothesis, an unclipped forced orbit, still rests on the
closed-form command on a dense depth grid, and the manuscript says so.
Sec. 6.1 states that Eq. (32) is a grid optimum: the best gain of the coarse grid and of the
refinements, for depths on the 2 mm grid, not a continuous optimum; Fig. 8 marks the grid points and
its caption says that only the grid depths are certified.

## 5 Second validation on Ozsoy et al. 2025

Not done. The open-access PDF linked in the review could not be retrieved from this computing
environment (the download was refused by the environment's policy), and only the abstract was
available. The validation the review lists (FRF of the second set-up, saturation zones, stability
regions, residual authority, basins at selected points) needs the paper's figures and parameters.
Limitations (Sec. 8) state that these measurements were not used. With the PDF available, the steps
are: rebuild the robot-assisted model from its parameters and FRF, compute the forced command and the
residual authority at the published cutting conditions, compare the predicted saturation zones and
stability regions with the measured ones, and compute the critical kick at selected points.

## 6 Editorial items

| Place | Change |
|---|---|
| Abstract | "This computational study ..."; "computed at 21 speeds"; the comparison stated with equal gain range and margins; no "certifies none" |
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
