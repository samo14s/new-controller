# Novelty memo: what is already published, and what this article may claim

Keys refer to `paper/refs.bib`. Descriptions of other works are based on their abstracts and
bibliographic records (full texts were not read). Before submission, the integrator should read the
full text of `ozsoy2025frequency`, the closest prior work on saturation (see §2).

## 0. Verification notes (for the integrator)

* 70 entries: all 48 keys of the brief, 19 new keys for the introduction and discussion, and 3 entries
  for `NEW:` placeholders already used in drafted sections. Replace the placeholders as follows:
  `NEW:bayly2003-tfea` -> `bayly2003stability`, `NEW:merdol2004-multifrequency` -> `merdol2004multi`,
  `NEW:davies2002-lowimmersion` -> `davies2002stability`, `NEW:blanes2009-magnus` -> `blanes2009magnus`,
  `NEW:trefethen2000-spectral` -> `trefethen2000spectral` (no DOI confirmed for this SIAM book).
  Every work was matched to a SciSpace
  record (authors, date, DOI) and/or to web-search records (volume, issue, pages). doi.org and
  api.crossref.org were not reachable from this session, so no publisher landing page was opened.
* Fields left out because no source confirmed them: issue of `insperger2004updated`, `wu2016adaptive`,
  `briat2013clock`, `dombovari2019experimental`; volume of `du2022time`; volume and article number of
  `sakthivel2022jdsmc` (first published online 28 Oct 2022; the final year may be 2023).
* No DOI exists for `gelb1968describing` (1968 book) or `insperger2006helix` (conference paper);
  `goulart2024clarabel` is an arXiv preprint (arXiv DOI given). For helix islands prefer
  `zatarain2006helix` (CIRP Annals, with DOI); keep `insperger2006helix` only as a second citation.
* Single-source items: the DOI of `dohner2004mitigation` comes from one catalogue record;
  `tlusty1981basic` pages 299--304 follow most citing records (one gives 21--25).
* `stepan2005nonlinear` was mapped to Stépán et al., J. Vib. Acoust. 127 (2005), "Nonlinear dynamics
  of high-speed milling: analyses, numerics, and experiments" (subcritical/period-doubling behaviour in
  milling). The alternative, Stépán, Insperger & Szalai, Int. J. Bifurcation Chaos 15 (2005) 2783--2798,
  doi 10.1142/S0218127405013642, was verified but is not in the .bib.
* `zhang2015ellipsoid` is Zhang et al., "Model predictive control to mitigate chatters in milling
  processes with input constraints" (the title does not contain "ellipsoid").
* `briat2013clock` is the Automatica version of arXiv:1304.1998 (preferred: journal, with DOI).
* `du2024ijms` has four authors (Du, Liu, Dai, Long); one index lists three.
* All DOIs written in the brief matched the records. DOIs easy to get wrong from memory: Baumann &
  Elliott 10.1121/1.2731018; Elliott et al. 2001 10.1115/1.1350822; Seuret & Gouaisbaut 2014
  10.1109/ECC.2014.6862453; Li, Lam & Cheung 2015 10.1016/j.automatica.2015.08.024; Shi & Tobias 1984
  10.1016/0020-7357(84)90045-3.

## 1. Closest prior work and how this article differs

### 1.1 Actuator saturation in active chatter control of milling (closest)

* `ozsoy2026robust` (base paper). Selects a DVF gain per spindle speed with SADE, rejecting gains whose
  peak actuator demand exceeds the available force, and validates the gain map in cutting tests on a
  flexible workpiece driven by a proof-mass actuator (up to 13 times the stable depth).
  *Difference:* this article keeps the DVF law fixed and asks what happens when the 9 V clip is active:
  coexistence of the forced orbit and clipped chatter, the size of the basin, and certificates for it;
  it is computational and uses that paper's case-study parameters with citation.
* `ozsoy2025frequency`. Studies actuator saturation in active vibration control of milling, presents a
  robot-assisted milling experiment showing the problem, and proposes a frequency-domain saturation model
  to guide the choice of cutting parameters; its abstract states that saturation can cause instability.
  *Difference:* this article works in the time domain around the forced orbit (basins measured by
  simulated kicks, LMI certificates); a describing-function estimate appears only as a comparison and
  must be credited to this line of work and to `gelb1968describing`.
* `ozsoy2022robotically`. Feasibility study of active vibration control with an inertial actuator in
  robot-assisted milling; its keywords include saturation sensitivity. *Difference:* no basin or
  certificate analysis there; context only.
* Soufi et al. (2026), Applied Mechanics 7(2) 50, doi 10.3390/applmech7020050 (not in .bib; MDPI).
  Tunes an ADRC fractional-order PID for half-immersion milling of a flexible workpiece by PSO under a
  ±27 N actuator-saturation constraint. *Difference:* saturation enters only as a tuning constraint; no
  bistability, basin or certificate.

### 1.2 Bistability with active damping, and saturated inertial actuators

* `sahu2021bistability`. Hardware-in-the-loop emulation of turning with three nonlinear cutting-force
  models shows bistable regions, and an active damper with velocity feedback stabilises them if the gain
  is tuned to the force model and chip thickness. *Difference:* there the bistability comes from the
  cutting-force nonlinearity in turning and the damper removes it; here the bistability is created by the
  actuator clip in a linear-force milling model, and the controller's own saturation shrinks the basin.
* `sahu2023adaptive`. Adaptive gain tuning to damp bistabilities in highly interrupted turning on the same
  HiL rig; re-entry after interruptions acts as the perturbation. *Difference:* turning, cutting-force
  nonlinearity, adaptive gains; no actuator-saturation mechanism or certificates.
* `baumann2007destabilization`. Velocity feedback through a stroke-limited inertial actuator adds damping
  at small amplitude and becomes unstable at large amplitude, with limit cycles of predictable frequency.
  *Difference:* stroke (not voltage) saturation and no cutting process; this article transfers the
  "stable small, unstable large" mechanism to regenerative milling with a hard voltage clip.
* `dalborgo2020active`. A nonlinear feedback added to velocity feedback prevents stroke saturation of an
  inertial actuator and improves robustness to perturbations (theory and experiment). *Difference:* not a
  machining loop; it modifies the controller, whereas this article analyses a fixed DVF law.
* `elliott2001feedback`. Shows that velocity feedback with an inertial actuator is only conditionally
  stable (actuator resonance must lie well below the structural one). Context for the actuator model.

### 1.3 Bistability and basins in cutting without control

* `dombovari2015bistable`. Semi-numerical method for the borders of the bistable (unsafe) zone of
  milling, from the tool leaving the workpiece; the stable periodic cut has a limited domain of
  attraction. *Difference:* no actuator, no controller; mechanism is loss of contact.
* `dombovari2019experimental`. Experimental hysteresis loops and burst excitations confirm unsafe zones in
  milling. *Difference:* uncontrolled machine; supports that bistability is physically observable.
* `dombovari2008estimates`. Analytical estimate of the bistable region below the stability limit for
  orthogonal cutting with a nonlinear force law. *Difference:* turning-type model, no control.
* `yan2017basins`. Defines and computes basins of attraction and stability basins for bistable
  time-delayed cutting dynamics. *Difference:* no actuator; this article measures basins by kicks along
  the velocity direction at 16 phases and certifies ellipsoids, for a controlled milling model.
* `stepan2005nonlinear`, `kalmarnagy2001subcritical`, `tlusty1981basic`, `shi1984theory`,
  `balachandran2001nonlinear`. Subcritical bifurcations, loss of contact and finite-amplitude instability
  in turning and milling. *Difference:* these are the uncontrolled mechanisms; in this article the
  unclipped controller is bistable only for very large kicks (loss of contact), and the clip is what
  makes small kicks sufficient.

### 1.4 Regional guarantees for chatter control under input constraints

* `zhang2015ellipsoid`. Input-constrained MPC for milling chatter with stability and feasibility
  guarantees for the perturbation dynamics around the chatter-free motion (README §11.1: ellipsoidal
  invariant set). *Difference:* designs a new constrained controller; this article certifies the basin of
  an existing fixed DVF law with the clip active.
* `wu2016adaptive`. Constrained adaptive chatter control with input saturation; closed-loop conditions via
  LaSalle–Yoshizawa. *Difference:* as above (new controller, adaptive); no measured basins.
* `ma2020active`. Adaptive sliding-mode chatter control in turning that respects an input constraint,
  with experiments. *Difference:* turning, controller design, no basin estimate.
* `cao2020switched`. Piecewise Lyapunov function for active chatter suppression in low-immersion milling
  modelled as a switched system; names controller saturation as a difficulty. *Difference:* no region of
  attraction under saturation.
* `huang2018robust`. LMI state-feedback design enlarging stability domains for variable-pitch milling.
  *Difference:* linear design; shows that LMIs in chatter control are not new.

### 1.5 Certificate tools (all borrowed, none new here)

* `gomesdasilva2005antiwindup`, `tarbouriech2011book`, `hu2001book`, `tarbouriech2009antiwindup`:
  generalised sector condition and LMI estimates of the basin of saturated linear systems.
  *Use here:* the dead zone is centred on the forced command b(t), so the share of the authority used by
  the forced orbit enters the certificate; this adaptation, not the sector condition, is the point.
* `allerhand2011robust`, `briat2013clock`, `li2015periodic`, `li2017hinfinity`: time-varying
  (piecewise-linear or clock-dependent) Lyapunov matrices for switched/periodic systems;
  `li2017hinfinity` already treats periodic piecewise *vibration* systems with actuator saturation.
  `zhou2011periodic`, `yang2015periodic`, `sakthivel2022jdsmc`: domain-of-attraction estimates for periodic
  (and delayed) systems with saturation. `petersen1987stabilization`: the lemma used to merge vertex
  conditions. *Use here:* the application to a clipped DVF law around the forced orbit of the
  time-periodic milling collocation model.
* `butcher2009chebyshev`, `breda2005pseudospectral`, `breda2015book`: Chebyshev/pseudospectral
  collocation of milling and delay equations. *Use here:* the certificates hold for the N-node model.
* `seuret2014bessel`: Bessel–Legendre Lyapunov–Krasovskii functionals; cite as the route to a
  certificate for the delay equation itself (not done here).

### 1.6 Delayed difference term g2[xdot - xdot(t-T)]

* `pyragas1992continuous`. Delayed feedback that vanishes on the target periodic orbit.
* `vandewouw2015ifac`, `vandijk2015ijrnc`. Low-order Pyragas-type/fixed-structure delayed output feedback
  for high-speed milling, designed by nonsmooth optimisation with limited control action.
* `li2022ijamt`. Feedback of x(t) - x(t-T) with PD and fuzzy laws on a piezo-actuated spindle; aims at the
  chatter component only and reduces the required actuator voltage (experiments).
* `mancisidor2019mechatronics`. Delayed feedback that distorts the regenerative effect in turning (HiL and
  vertical lathe). `du2022time`: time-delay feedback reducing the regenerative effect in milling,
  designed with a delay-dependent Lyapunov function, with lower energy cost. `du2024ijms`: robust plus
  time-delay control, 25 % and 11 % lower control voltage than robust control alone.
* *Difference for all six:* the law and its "lower effort" motivation are published. This article
  analyses the term under a hard 9 V clip: the authority budget used by the forced orbit, the certified
  basin with the delayed terms inside the sector argument, the gains that maximise it, simulator checks,
  and robustness limits (comb-delay error, runout, loop delay with an assumed second mode).

### 1.7 Context: active chatter control hardware and design

* `munoa2013chatter`, `mancisidor2015coupled` (time-domain coupled model of milling with an inertial
  actuator, including actuator dimensioning), `kleinwort2021experimental` (auto-tuned DVF, adaptive and
  robust controllers in cutting tests), `dohner2004mitigation`, `chen2007control`, `vandijk2012robust`,
  `monnin2014optimal`. None of these analyses saturation-induced bistability (from abstracts), but
  time-domain simulation of milling with an actuator force limit is itself not new
  (`mancisidor2015coupled`, `ozsoy2026robust`).
* Averaging, helix and runout: `bayly2003stability`, `merdol2004multi`, `insperger2004updated`,
  `zatarain2006helix`, `insperger2006helix`, `insperger2008chatter`. The failure of averaging at low
  immersion and helix-induced flip islands are classical.

## 2. Search for post-2018 work on saturation, bistability or basins in ACTIVE chatter control of milling

Queries (SciSpace and web search; Consensus quota was exhausted): actuator saturation active chatter
control milling; basin of attraction chatter active damping saturation; bistability active chatter
suppression inertial actuator; Ozsoy Sims Ozturk saturation; anti-windup chatter; invariant set /
domain of attraction constrained chatter control.

Found: (i) the Ozsoy–Sims–Ozturk line (`ozsoy2022robotically`, `ozsoy2025frequency`, `ozsoy2026robust`,
and a 2025 Research Square preprint with the same authors and the same abstract claims, apparently an
earlier version of the 2026 paper, doi 10.21203/rs.3.rs-6163421/v1); (ii) Soufi et al.
2026 (saturation as a tuning constraint); (iii) bistability with active damping only in turning
(`sahu2021bistability`, `sahu2023adaptive`); (iv) input-constrained chatter controllers (`ma2020active`
for turning; `zhang2015ellipsoid`, `wu2016adaptive` before 2018); (v) basins of attraction in uncontrolled
cutting (`yan2017basins`, `dombovari2019experimental`). No work was found that measures basins of
attraction, or certifies a region of attraction, for a fixed active-damping law under actuator saturation
in milling. This is a negative search result, not proof of absence: write "to the authors' knowledge"
and check the full text of `ozsoy2025frequency` (its frequency-domain model may already predict
saturation-induced instability above a threshold amplitude, i.e. a hard-excitation boundary).

## 3. Claims the article must NOT make

1. That it is the first to study actuator saturation in active chatter control of milling, or the first
   to show that saturation can destabilise it (`ozsoy2025frequency`, `ozsoy2026robust`, `wu2016adaptive`,
   `zhang2015ellipsoid`).
2. That bistability, unsafe zones or limited domains of attraction in milling are new
   (`dombovari2015bistable`, `dombovari2019experimental`, `stepan2005nonlinear`, `tlusty1981basic`,
   `shi1984theory`, `balachandran2001nonlinear`).
3. That bistability in actively damped cutting with velocity feedback has not been studied
   (`sahu2021bistability`, `sahu2023adaptive`; turning).
4. That a saturated velocity-feedback loop being stable at small and unstable at large amplitude is a new
   finding (`baumann2007destabilization`, `dalborgo2020active`).
5. That this is the first regional, invariant-set or domain-of-attraction guarantee for chatter control
   under input constraints (`zhang2015ellipsoid`, `wu2016adaptive`), or the first use of LMIs in chatter
   control (`huang2018robust`).
6. That basins of attraction of cutting dynamics are computed here for the first time (`yan2017basins`).
7. Any novelty for the tools: generalised sector condition, anti-windup-type basin estimates,
   piecewise-linear or clock-dependent periodic Lyapunov matrices, saturated periodic systems, Petersen's
   lemma, Chebyshev collocation of milling (`gomesdasilva2005antiwindup`, `tarbouriech2011book`,
   `allerhand2011robust`, `briat2013clock`, `li2015periodic`, `li2017hinfinity`, `zhou2011periodic`,
   `yang2015periodic`, `sakthivel2022jdsmc`, `petersen1987stabilization`, `butcher2009chebyshev`).
8. That the delayed difference law, its vanishing on the T-periodic orbit, or the argument that it lowers
   control effort is new (`pyragas1992continuous`, `vandewouw2015ifac`, `vandijk2015ijrnc`, `li2022ijamt`,
   `mancisidor2019mechatronics`, `du2022time`, `du2024ijms`).
9. That the frequency-domain/describing-function view of the clip is new (`ozsoy2025frequency`,
   `gelb1968describing`).
10. That tool entry acting as the perturbation that triggers chatter in a bistable zone is a new idea
    (`sahu2023adaptive` uses re-entry as the perturbation; `ozsoy2026robust` simulates from rest).
11. That time-domain simulation of milling with a force-limited actuator is new (`mancisidor2015coupled`,
    `ozsoy2026robust`).
12. That averaging fails at low immersion, or that helix-angle flip islands exist, as findings of this
    article (`bayly2003stability`, `merdol2004multi`, `insperger2004updated`, `zatarain2006helix`).
13. Anything beyond what was done: a certificate for the delay-differential equation itself (the proofs
    are on the N-node collocation models), a decay rate, global stability, experimental validation, more
    than one structural mode, or a certified sharp (step) entry at productive depths.
14. Any statement that the base paper's gains or experiments are wrong; only the two technical points
    allowed by the brief (peak demand from rest in its Eq. 2; 1.34 % damping), in neutral tone.

Avoid "first", "novel", "unprecedented"; use "to the authors' knowledge" once, for the narrow claim in §4.

## 4. Suggested contribution statement (three sentences)

For a single-mode regenerative milling model in which direct velocity feedback acts through a proof-mass
actuator whose command is clipped at 9 V, time-domain simulation shows that linearly stable cuts can be
bistable, and the size of the basin of the forced orbit, together with its dependence on how the tool
enters the cut, is quantified against the unclipped controller. A region of attraction around the forced
orbit is then certified with sector-condition LMIs whose dead zone is centred on the forced command,
proved on Chebyshev-collocation models of the averaged and of the time-periodic milling dynamics and
checked against an independent simulator. Under the same clip, a published delayed difference term that
vanishes on the forced orbit is analysed for how much actuator authority it frees for the deviation and
how much it enlarges the certified basin, and its limits under loop delay, runout and comb-delay error
are stated.
