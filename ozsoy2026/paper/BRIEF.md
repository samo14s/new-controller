# Manuscript brief (shared by every contributor)

Working title: **Saturation-induced bistability in active chatter control of milling:
basins of attraction, certificates, and an authority-preserving delayed term**

Target: *Mechanical Systems and Signal Processing* (Elsevier, Q1), `elsarticle` class,
`review`/`preprint` layout, author-year citations (`natbib`, `\citet`/`\citep`).
Language: formal technical English, past tense for what was done, present tense for what
holds. No first person singular. No hype ("novel", "unprecedented", "remarkable").
Every number must come from the repository (README.md, results/, verification/);
after each paragraph that quotes numbers add a LaTeX comment `% src: <file or README section>`.
Do not quote numbers that are not stored or stated in README.md. State limitations plainly.

Source of truth: `/home/user/new-controller/ozsoy2026/README.md` (Arabic; sections cited
as README §x.y), `src/*.py`, `results/*`, `verification/*`. The paper being built on is
Ozsoy, Sims & Ozturk 2026 (J. Manuf. Processes 171:247-257, CC BY-NC-ND): cite it, never
copy its prose; its case-study parameters may be used with citation.

## Scope and honesty rules
* The work is computational: a single identified mode, a time-domain simulator, linear
  models, LMI certificates. **No experiments.** Say so where it matters.
* Novelty is narrow and must be stated as such (README §11.1, §12.1, §12.7, §13.2, §14.3 list
  prior work): bistability with velocity feedback is known in turning (Sahu et al. 2021) and
  for stroke-limited inertial actuators (Baumann & Elliott 2007); regional guarantees under
  input constraints exist for other chatter controllers (Zhang et al. 2015, Wu et al. 2016);
  the delayed difference law is published (Li et al. 2022; Pyragas 1992; van de Wouw et al.
  2015; van Dijk et al. 2015). What this paper adds is the analysis of a fixed DVF law under a
  hard voltage clip in a regenerative milling model, the moving dead zone around the forced
  command inside a sector-condition certificate, its validation in an independent simulator,
  the periodic-model certificate, and the authority analysis of the delayed term with its
  robustness limits.
* The certificates are proved on the N-node collocation model (averaged model in C2,
  time-periodic model in C3), not on the delay equation itself. The "proof check" is a
  sampled consistency check. No decay rate is claimed.
* The D1 chatter classifier (300 revolutions) biases near-limit boundaries high by
  0.05-1.5 % (check8) and up to 4.4 % near the forced resonance (check10).
* Ozsoy et al.'s own inconsistencies (README §4) are NOT the subject of this paper. Mention
  only what is technically needed: (i) the published feasibility condition (their Eq. 2)
  takes the peak demand over a simulation from rest, which is dominated by the entry
  transient and makes the feasible set fragmented (README §8.1, §5.2); (ii) the computed
  curves of the paper fit a damping ratio of 1.34 % rather than the 1.45 % of its Table 1,
  so 1.34 % is used (README §4.1). Keep the tone neutral.

## Outline and owners
| § | Title | file | owner |
|---|---|---|---|
| — | Abstract, highlights, keywords | main.tex | integrator |
| 1 | Introduction | sections/s1_intro.tex | integrator |
| 2 | Model and case study | sections/s2_model.tex | agent A |
| 3 | A well-posed authority condition for gain scheduling | sections/s3_feasibility.tex | agent A |
| 4 | Saturation-induced bistability | sections/s4_bistability.tex | agent A |
| 5 | Certified basins of attraction | sections/s5_certificates.tex | agent B |
| 6 | An authority-preserving delayed term | sections/s6_delayed.tex | agent C |
| 7 | Discussion | sections/s7_discussion.tex | integrator |
| 8 | Conclusions | sections/s8_conclusions.tex | integrator |
| A | Appendix A: simulator and its verification | sections/appA_simulator.tex | agent A |
| B | Appendix B: LMI conditions and numerical settings | sections/appB_lmi.tex | agent B |

Length targets (words): §2 1100, §3 900, §4 1800, §5 2300, §6 2100, App. A 700, App. B 700.

## Notation (use exactly these symbols)
| symbol | meaning |
|---|---|
| $n$ | spindle speed (rpm); $\Omega = 2\pi n/60$ |
| $N_t=4$ | number of teeth; $T = 60/(N_t n)$ tooth-passing period (the delay) |
| $a$ | axial depth of cut; $a_{\lim}(g)$ linear stability limit (zero-order approximation, ZOA) |
| $\kappa = 0.9$ | depth margin of the authority condition; $a_{\mathrm{chk}}$ the checked depth |
| $x(t)$ | workpiece displacement in the flexible (feed-normal) direction; $m,c,k$ modal mass, damping, stiffness; $\omega_n=2\pi\cdot129.3$ rad/s, $\zeta=1.34\,\%$ |
| $G(s)=1/(ms^2+cs+k)$ | structure; $G_a(s)=K_a s^2/(s^2+2\zeta_a\omega_a s+\omega_a^2)$ actuator, $K_a=3$ N/V, $2\zeta_a\omega_a=15.834$ s$^{-1}$, $\omega_a^2=2785.6$ s$^{-2}$ |
| $V(t)=\operatorname{sat}_{V_{\max}}(u(t))$ | applied command, $V_{\max}=F_{\max}/K_a=9$ V, $F_{\max}=27$ N |
| $u=-g\dot x$ | direct velocity feedback (DVF), gain $g$ in V s/m (written $g_1$ in §6) |
| $u=-g_1\dot x-g_2[\dot x-\dot x(t-T)]-k_d[x-x(t-T)]$ | delayed difference law (§6) |
| $x_p(t)$, $b(t)=-g\dot x_p(t)$, $b_{\max}=\max_t\lvert b\rvert$ | $T$-periodic forced orbit, its command and peak command |
| $\xi$ | deviation state of the collocation model around the forced orbit (dimension $4+N$, $N$ Chebyshev nodes of the delayed history) |
| $\mathcal E(P)=\{\xi:\xi^\top P\xi\le1\}$ | certified ellipsoid; $\beta$ certified kick (m/s) along the velocity direction |
| $\Delta v$ | velocity kick applied to the workpiece at phase $\theta$; $\Delta v^*$ simulated critical kick (smallest kick that ends in chatter) |
| $f_z$ | feed per tooth (0.05 mm) |

Units: mm for depths, V s/m for gains, mm/s or m/s for kicks, rpm for speeds. Use `siunitx`
(`\SI{9}{\volt}`, `\si{\volt\second\per\metre}`) or plain text units consistently: prefer
`\SI{...}{...}`.

## Figures (labels and content; files written to paper/figures/)
| label | file | content |
|---|---|---|
| fig:model | TikZ (integrator) | block diagram: structure, actuator, clip, DVF, regenerative force with delay |
| fig:feas | fig02_feasibility.pdf | (a) gain maps: Eq. (2) of Ozsoy et al. as written (reproduced) vs reformulated condition ($\mathcal G\le1500$); (b) depths: uncontrolled $a_{\lim}(0)$, $a_{\lim}$ at the Eq. (2) gains, $\kappa a_{\lim}$ and the checked depth $a_{\mathrm{chk}}$ of the reformulated condition |
| fig:two | fig03_two_outcomes.pdf | the same linearly stable cut (2500 rpm, $g=1500$, a fraction of $a_{\lim}$) after a small and a slightly larger kick: displacement and command time histories; one returns to the forced orbit, one ends in clipped chatter |
| fig:map | fig04_entry_map.pdf | over 251 speeds: $a_{\lim}$ (linear), step-entry limit with the 9 V clip, face-entry limit, lower edge of the bistable band from kicks |
| fig:basin | fig05_basin.pdf | critical kick $\Delta v^*$ vs $a/a_{\lim}$ at 1090, 1420, 2500, 2870 rpm: with the clip (min and median over 16 phases) and without it; describing-function estimate |
| fig:cert | fig06_certificates.pdf | certified kick (C2 averaged model; C3 periodic model) vs simulated critical kick (min over phases) at the four speeds |
| fig:surf | fig07_certified_surface.pdf | certified kick over speed and $a/a_{\lim}$ (C2 surface) and the step-entry certificate $\beta$ |
| fig:dlim | fig08_delayed_limits.pdf | linear limit over 251 speeds, DVF vs DVF + D ($g_2$ = 1000/2000/4000) under the same authority rule; simulated face and step entries |
| fig:dbasin | fig09_delayed_basin.pdf | certified kick vs depth (best DVF vs best DVF + D) and simulated critical kicks (best DVF, map design, same-$g_1$ design) at the four speeds |
| fig:robust | fig10_robustness.pdf | comb-delay error, runout tolerance, loop delay with an assumed second mode |

## Citation keys (use these; for any other source write `\citep{NEW:<author><year>-<topic>}`)
ozsoy2026robust (base paper, JMP 171), ozsoy2025frequency (Ozsoy, Sims, Ozturk 2025, MSSP 224:111942),
altintas1995analytical (ZOA), insperger2002semi, insperger2004updated, insperger2011book (SDM),
munoa2016chatter (CIRP review), altintas2020chatter (JMSE review), quintana2011chatter,
dombovari2015bistable (Phil. Trans. R. Soc. A), kalmarnagy2001subcritical, stepan2005nonlinear,
tlusty1981basic, sahu2021bistability (CIRP JMST 2021, velocity feedback turning),
baumann2007destabilization (JASA, stroke-limited inertial actuator), elliott2001feedback,
zhang2015ellipsoid (IJMTM 2015, doi 10.1016/j.ijmachtools.2015.01.002),
wu2016adaptive (IJRNC 2016, doi 10.1002/rnc.3493), gomesdasilva2005antiwindup (IEEE TAC 2005),
tarbouriech2011book, hu2001book, petersen1987stabilization, allerhand2011robust (IEEE TAC 2011),
briat2013clock (arXiv 1304.1998), li2015periodic (Automatica 2015), zhou2011periodic (Automatica 2011),
yang2015periodic (IJRNC 2015), sakthivel2022jdsmc (JDSMC, doi 10.1115/1.4056106), cao2020switched (JMSE 2020),
butcher2009chebyshev (JCND 2009), breda2005pseudospectral, breda2015book, pyragas1992continuous,
vandewouw2015ifac (doi 10.1016/j.ifacol.2015.09.400), vandijk2015ijrnc (doi 10.1002/rnc.3280),
li2022ijamt (doi 10.1007/s00170-022-09128-w), mancisidor2019mechatronics (doi 10.1016/j.mechatronics.2019.102276),
du2024ijms (doi 10.1016/j.ijmecsci.2024.109257), qin2009sade (IEEE TEC), goulart2024clarabel,
insperger2006helix / zatarain2006helix (helix angle), seuret2014bessel (Bessel-Legendre),
dohner2004mitigation, munoa2013chatter, vandijk2012robust (IEEE TCST), monnin2014optimal,
gelb1968describing (describing functions).
The literature agent verifies these and writes `paper/refs.bib`.

## Figure/table/equation conventions
* `\label{sec:...}`, `\label{eq:...}`, `\label{tab:...}`, `\label{fig:...}`; refer with `Fig.~\ref{}`,
  `Eq.~\eqref{}`, `Table~\ref{}`, `Section~\ref{}`.
* Tables with `booktabs`. Equations numbered only when referred to.
* Each section file starts with `\section{...}\label{sec:...}` and contains no preamble.
