# Literature Sweep A: data-driven reduced-order modelling (DMD / DMDc / Koopman) for structural vibration and smart structures

Prepared 2026-10-16 for the project `sddmdc_fxlms` (base article: Ouyang, Dong, Liu, Zhou, Thin-Walled Struct. 228 (2026) 115090).

## How this sweep was run (read first)

- Tools that worked: Consensus `search` (about 30 queries), Wiley Scholar Gateway `semanticSearch`, SciSpace `search-papers`, and WebSearch (standard and extended).
- Tools that failed: WebFetch got a DNS failure on every host tried (sciencedirect.com, arxiv.org, api.crossref.org, api.openalex.org, api.semanticscholar.org), and curl to OpenAlex got a proxy 403. As a result:
  - I could not open full texts. Every annotation below rests on the abstract or snippet returned by the search tools.
  - I could not run a forward-citation ("cited-by") query. Citation counts come from Consensus metadata only.
- Tags:
  - **[VERIFIED]**: title, first author, venue and DOI (or arXiv ID / URL) all appeared in a tool result during this session. Tool output often truncated author lists to "et al.", so I write "et al." unless the co-authors were displayed. Where I added names from memory, the entry says "from memory".
  - **[PARTIAL]**: the item was seen, but one bibliographic field (volume, pages or year) was not. The missing field is named.
  - **[UNVERIFIED]**: the item was not seen in any tool result. It is listed only so that it can be checked later.
- Statements marked *Analysis:* are my own reasoning. They are not claims taken from the cited paper.

---

## 0. Anchor papers (the base and its lineage)

1. **Q. Ouyang, L. Dong, J. Liu, J. Zhou**, "A scaled derivative-based DMDc method for modelling multiple-input multiple-output mechanical systems," *Applied Mathematical Modelling* 140 (2025) 115866. DOI: 10.1016/j.apm.2024.115866. ScienceDirect PII S0307904X2400619X. An SSRN preprint exists under the title "A Derivative-Based DMDc Method…". **[VERIFIED]**
   - What it does (from the abstract): the state is augmented with *scaled derivatives of displacement*, and an *extended forward-backward strategy* is integrated (the forward-backward idea is due to Dawson et al. 2016, see §5). Two parameters are tuned by minimising the response error. Tested on a MIMO cantilever plate, first in simulation and then in experiment.
   - Comparison: **only against delay-DMDc**. The abstract claims better accuracy across noise levels and under partial observation.
   - Consensus lists **4 citations**.
2. **Q. Ouyang, L. Dong, J. Liu, J. Zhou**, "From data-driven DMDc modeling to control parameter optimization: a framework for adaptive vibration control of thin-walled composites," *Thin-Walled Structures* 228 (2026) 115090. DOI: 10.1016/j.tws.2026.115090. **[VERIFIED]**
   - This is the base article. Consensus lists **0 citations** as of 2026-10-16.
3. **Q. Ouyang, L. Dong, J. Liu, J. Zhou**, "Identification of electromechanical participation matrices via scaled-derivative DMDc for improved PPF active vibration control of thin-walled structures," *Engineering Structures* 353 (2026, April) 122232. DOI: 10.1016/j.engstruct.2026.122232. **[VERIFIED]** (no abstract was available in any tool)
   - Same group. It applies sd-DMDc to identify modal participation matrices, which are then used for MIMO positive position feedback (PPF) control.
   - It is the closest sibling of the base paper. The base paper's simulate-then-tune workflow is repeated here with PPF in place of FxLMS.
   - The first author is affiliated with Xi'an Jiaotong-Liverpool University, according to a ResearchGate profile reported in a WebSearch summary.

---

## 1. Q1: DMD / DMDc / ioDMD / Koopman in structural dynamics, smart structures and AVC (2016–2026)

### 1a. Foundations needed for Phase 0 (pinning the method)

4. **J.L. Proctor, S.L. Brunton, J.N. Kutz**, "Dynamic mode decomposition with control," *SIAM J. Appl. Dyn. Syst.* 15(1) (2016) 142–161. DOI: 10.1137/15m1013857. arXiv: 1409.6358. **[VERIFIED]**
   - The original DMDc paper.
   - Its Section V, "Connections to System Identification Methods", compares DMDc with ERA and OKID. It notes that ERA/OKID were built for flexible aerospace structures. I saw this in a WebSearch summary but did not read the section itself.
5. **(authors not displayed by Consensus; Proctor, Brunton & Kutz from memory)**, "Generalizing Koopman theory to allow for inputs and control," *SIAM J. Appl. Dyn. Syst.* (2018). DOI: 10.1137/16m1062296. **[VERIFIED]**
   - Provides the Koopman-with-inputs theory that connects to DMDc.
6. **J.H. Tu, C.W. Rowley, D.M. Luchtenburg, S.L. Brunton, J.N. Kutz**, "On dynamic mode decomposition: theory and applications," *J. Comput. Dyn.* 1(2) (2014) 391–421. DOI: 10.3934/jcd.2014.1.391. **[VERIFIED]**
   - Defines exact DMD and links DMD to ERA and to linear inverse modelling (LIM). It proves DMD is equivalent to LIM under certain conditions.
   - Explains the pitfalls of rank-deficient data ("linear consistency").
7. **P.J. Schmid**, "Dynamic mode decomposition and its variants," *Annu. Rev. Fluid Mech.* 54 (2022). DOI: 10.1146/annurev-fluid-030121-015835. **[VERIFIED]**
   - A practical review of DMD variants. Useful as a checklist of algorithms the base paper did not try.
8. **M.O. Williams et al.**, "A data-driven approximation of the Koopman operator: extending dynamic mode decomposition," *J. Nonlinear Sci.* (2015). DOI: 10.1007/s00332-015-9258-5. **[VERIFIED]**
   - Introduces EDMD.
9. **M. Korda et al.**, "Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control," *Automatica* (2018). DOI: 10.1016/j.automatica.2018.03.046. **[VERIFIED]**
   - EDMD with inputs, leading to a linear predictor for MPC.
10. **H. Arbabi et al.**, "Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the Koopman operator," *SIAM J. Appl. Dyn. Syst.* (2017). DOI: 10.1137/17m1125236. **[VERIFIED]**
    - Hankel (delay) DMD converges to Koopman eigenpairs. This is the theoretical basis of the delay-DMDc baseline that sd-DMDc beat.
11. **S.L. Brunton, M. Budišić, E. Kaiser, J.N. Kutz**, "Modern Koopman theory for dynamical systems," *SIAM Review* (2022). DOI: 10.1137/21m1401243. **[PARTIAL]** (the co-author list beyond Brunton was not shown; the authors given here are from memory)
12. **S.M. Ichinaga et al.**, "PyDMD: a Python package for robust dynamic mode decomposition," arXiv:2402.07463 (2024). **[VERIFIED]**
    - Free Python implementations of BOP-DMD, optimised DMD, Hankel DMD, DMDc and piDMD. Directly usable in a laptop-only project.

### 1b. DMD as a modal-analysis tool for structures (output or free-response data)

13. **A. Saito et al.**, "Data-driven experimental modal analysis by Dynamic Mode Decomposition," *J. Sound Vib.* (2020) 115434. DOI: 10.1016/j.jsv.2020.115434. **[VERIFIED]**
    - Links DMD to the Ibrahim time-domain (ITD) method.
    - Reports that DMD fails to capture modal parameters when measurement errors are large, and that **damping ratios are the first to go**. This is relevant to how sd-DMDc behaves under noise.
14. **Y. Si, B. Jayawardhana, J.N. Kutz, Y. Zhu, L. Cheng**, "Experimental modal analysis for engineering structures via time-delay dynamic mode decomposition with control," arXiv:2603.20249 (March 2026). **[VERIFIED]**
    - Proves that time-delay DMDc and pLSCF (PolyMAX) give mathematically equivalent modal parameters, through the state-space form of an ARX model.
    - The appendix analyses noise: averaging over the delay window reduces how much noise projects onto the dominant modes.
    - Validated on a 6-DOF system and on a camera-measured cantilever.
    - This is the **strongest theoretical counter-position to sd-DMDc's claim that derivative augmentation beats delays.**
15. **H. Li et al.**, "Characteristic value decomposition: a unifying paradigm for data-driven modal analysis," *Mech. Syst. Signal Process.* (2024) 111769. DOI: 10.1016/j.ymssp.2024.111769. **[VERIFIED]**
    - Shows that state-variable decomposition, DMD, ITD and **ERA are special cases** of one quotient-eigenproblem. Claims higher noise robustness than the established methods.
16. **N. Savoeurn et al.**, "Determination of dynamic characteristics of lattice structure using dynamic mode decomposition," *J. Appl. Mech.* (2024). DOI: 10.1115/1.4065055. **[VERIFIED]**
    - Uses HAVOK-DMD (Hankel embedding) to extract frequency and damping of viscoelastic lattices.
17. **A. Talati et al.**, "Identification of excitation frequencies of a vibrating beam using Dynamic Mode Decomposition," *Mech. Based Des. Struct. Mach.* (2024). DOI: 10.1080/15397734.2024.2377255. **[VERIFIED]**
    - Plain DMD fails on standing waves in a forced beam. A Hankel matrix fixes it, with accuracy depending on the number of stacked delays and the rank truncation.
18. **W. Guan et al.**, "Output-only modal identification with recursive dynamic mode decomposition for time-varying systems," *Measurement* (2024). DOI: 10.1016/j.measurement.2023.113852. **[VERIFIED]**
19. **H. Zheng et al.**, "Enhanced subspace dynamic mode decomposition for operational modal analysis of aerospace structures," *Aerospace* 12 (2025) 776. DOI: 10.3390/aerospace12090776. **[VERIFIED]**
    - Uses subspace orthogonal projection of a Hankel matrix before DMD to reduce noise. Reports better damping estimates under white noise.
20. **L. Cheng et al.**, "Camera-based dynamic vibration analysis using transformer-based model CoTracker and dynamic mode decomposition," *Sensors* 24 (2024) 3541. DOI: 10.3390/s24113541. **[VERIFIED]**
21. **F. Colombo et al.**, "Damage identification in structures using dynamic mode decomposition for vibration analysis of low-resolution videos," *Struct. Health Monit.* (2025). DOI: 10.1177/14759217251327555. **[VERIFIED]**
    - Uses optDMD on noisy, low-resolution video.
22. **H. Namiki et al.**, "Short-time dynamic mode decomposition: a data-driven forced response analysis of nonlinear systems under swept-sine excitation," *Nonlinear Dyn.* (2025). DOI: 10.1007/s11071-025-11751-7. **[VERIFIED]**
    - Standard DMDc fails for strong nonlinearity (Duffing). The windowed variant, STDMD, captures jump and hysteresis.
23. **P.T. Frame, A. Towne**, "Space-time POD and the Hankel matrix," *PLOS ONE* (2023). DOI: 10.1371/journal.pone.0289637. **[VERIFIED]**
    - The SVD of a block-Hankel matrix approximates space-time POD. This interprets what delay-DMDc and ERA truncate.

### 1c. DMDc / ioDMD as input–output models of flexible structures (closest to the base paper)

24. **M. Krishnan, S. Gugercin, P.A. Tarazaga**, "A wavelet-based dynamic mode decomposition for modeling mechanical systems from partial observations," *Mech. Syst. Signal Process.* 187 (2023) 109919. DOI: 10.1016/j.ymssp.2022.109919. **[VERIFIED]**
    - Combines wavelets with ioDMD. Builds a linear state-space model from the inputs and a few outputs of an FE beam and an experimental free-free beam.
    - Cited by the base paper as [49]. It directly addresses the same partial-observation problem as sd-DMDc.
25. **M. Krishnan, S. Gugercin, P.A. Tarazaga**, "Wavelet-based dynamic mode decomposition," *PAMM* 20(S1) (2021). DOI: 10.1002/pamm.202000355. **[VERIFIED]**
26. **J. Annoni, P. Seiler**, "A method to construct reduced-order parameter-varying models," *Int. J. Robust Nonlinear Control* 27(4) (2017) 582–597. DOI: 10.1002/rnc.3586. **[VERIFIED]**
    - The ioDMD reference. Combines POD, DMD and direct subspace identification using state, input **and output** data. The introduction explicitly discusses ERA and its link to balanced POD (BPOD).
    - Motivated by flexible aircraft.
27. **H.-P. Wan, Q. Ma, G.-S. Dong, Y. Luo, Y.-Q. Ni**, "Data-driven model reduction approach for active vibration control of cable-strut structures," *Eng. Struct.* 302 (2024) 117434. DOI: 10.1016/j.engstruct.2023.117434. **[VERIFIED]**
    - Builds a DMDc reduced-order state-space model and designs LQR on it. The DMDc-LQR and FEM-LQR design parameters are linked so the two controllers have equal performance indexes. Numerical only (cable dome, tensegrity).
    - Key quantitative finding: **increasing the model order mitigates noise** in the DMDc reduced model.
    - Cited by the base paper as [15]. It is a clear precedent for a *numerical-only* DMDc-for-AVC paper.
28. **N. Fonzi, S.L. Brunton, U. Fasel**, "Data-driven nonlinear aeroelastic models of morphing wings for control," *Proc. R. Soc. A* (2020). DOI: 10.1098/rspa.2020.0079. **[PARTIAL]** (the co-authors were not shown; the names given are from memory)
    - Extends DMDc to handle algebraic equations and interpolates between operating-point models. Used for MPC.
29. **J. Rains, A. House, A.L. Kaminsky, N.A. Tison, V.M. Korivi**, "Constrained optimized dynamic mode decomposition with control for physically stable systems with exogeneous inputs," *J. Comput. Phys.* (2023) 112604. DOI: 10.1016/j.jcp.2023.112604. **[VERIFIED]**
    - cOptDMDc: variable-projection optDMD extended to inputs, with an eigenvalue stability constraint.
    - Maximum relative error below 0.5%. Robust to the choice of subspace dimension, so no trial-and-error rank selection is needed.
    - *Analysis:* this addresses the base paper's sd-DMDc tuning of rank p and r directly.
30. **J. Rains et al.**, "Residual dynamic mode decomposition with control for nonlinear dynamic systems," *Struct. Multidiscip. Optim.* (2026). DOI: 10.1007/s00158-025-04205-6. **[VERIFIED]**
    - ResDMDc removes spurious eigenpairs. Tested on a nonlinear aeroelastic case.
31. **Z. Bai, E. Kaiser, J.L. Proctor, J.N. Kutz, S.L. Brunton**, "Dynamic mode decomposition for compressive system identification," *AIAA J.* (Consensus lists 2017; journal year not confirmed). DOI: 10.2514/1.j057870. **[PARTIAL]** (the co-author list was truncated in the result)
    - Combines DMDc with subsampled measurements. Studies the effect of sensor noise and sensor type.
32. **G. Palma et al.**, "Analysis, forecasting, and system identification of a floating offshore wind turbine using dynamic mode decomposition," *J. Mar. Sci. Eng.* 13 (2025) 656. DOI: 10.3390/jmse13040656. **[VERIFIED]**
    - Hankel-DMDc for system identification, with a full-factorial hyperparameter study and a Bayesian extension for uncertainty quantification.
33. **B. Swaminathan et al.**, "Enhanced Hankel DMDc system identification through noise-normalized total least squares," *2025 Indian Control Conf.* DOI: 10.1109/icc69100.2025.11372444. **[VERIFIED]**
    - Same first author as the base paper's ref [51], which applied DMDc to aircraft parameter estimation.
34. **A. Alassaf et al.**, "Identifying wind farm parameters via dynamic mode decomposition with control," *IEEE Trans. Ind. Inform.* (2025). DOI: 10.1109/tii.2025.3586051. **[VERIFIED]**
    - One of the few papers that **compares DMDc against the prediction error method (PEM)**. Reports DMDc as superior, in power systems.
35. **C. Vega-Zambrano et al.**, "Advanced control strategies for continuous pharmaceutical manufacturing via regularized dynamic mode decomposition with control," *Digital Chem. Eng.* (2026) 100315. DOI: 10.1016/j.dche.2026.100315. **[VERIFIED]**
    - rDMDc (Tikhonov regularisation) outperforms DMDc, noise-handling variants and **N4SID**.
    - Reported numbers: RMSE 0.026; validation RMSE 26% lower (2.35 vs 3.19) in one phase and 63% lower than DMDc (0.85 vs 2.34) in another.
    - An explicit DMDc-vs-N4SID benchmark, though not in structures.
36. **F. Giannini**, "A novel data-driven representation of linear systems based on DMDc to deal with noisy data," *CoDIT 2026*. DOI: 10.1109/codit70676.2026.11631240. **[VERIFIED]**
    - Extended or lifted DMDc for linear time-invariant systems with process and measurement noise.
37. **C.D. Mendez Gaona**, "Data-driven flutter suppression via HODMD, DMDc and constrained MPC," arXiv:2609.13459 (2026). **[VERIFIED]**
    - Five-state reduced model. The flap input matrix B is estimated by DMDc from a PRBS record.
    - Normalised error on an independent chirp: 2.4×10⁻³. Growth-rate error under 2% RMS noise: 1.8×10⁻³ s⁻¹.
38. **"Optimal derivative feedback control for an active magnetic levitation system: an experimental study on data-driven approaches,"** arXiv:2602.06944 (2026). **[PARTIAL]** (authors not seen)
    - Uses DMDc for the initial model and **PEM to refine it**. A hybrid DMDc→PEM pipeline worth copying.
39. **Abdennadher, Williams Riquer, Dao, Grabe (TU Hamburg)**, "AugDMDc," COMPDYN 2025, repository hdl.handle.net/11420/61035. **[PARTIAL]** (exact title and DOI not seen)
    - Adds time-delay coordinates to DMDc for noise robustness, on a structure with an active mass damper. Reports that standard DMDc misestimates damping under noise.
40. **H. Wang et al.**, "Real-time states estimation of a farm tractor using dynamic mode decomposition," *GPS Solutions* (2020). DOI: 10.1007/s10291-020-01051-5. **[VERIFIED]**
    - **tlsDMDc** (total-least-squares DMDc) is more accurate than DMDc on noisy linear data.
41. **H. Wang et al.**, "From features to states: data-driven selection of measured state variables via RFE-DMDc," arXiv:2511.20552 (2025). **[VERIFIED]**
    - Selects the sensor or state set before DMDc. *Analysis:* relevant to whether 35 laser points plus 2 strain channels are needed.

### 1d. Koopman / EDMD control of flexible or mechanical systems

42. **G. Mamakoukas et al.**, "Derivative-based Koopman operators for real-time control of robotic systems," *IEEE Trans. Robot.* (2021). DOI: 10.1109/tro.2021.3076581. **[VERIFIED]**
    - Builds the Koopman basis from **higher-order state derivatives**, with a Taylor-series error bound in terms of horizon and derivative magnitude. Derivatives are estimated numerically from noisy data.
    - *Analysis:* the closest conceptual precedent for sd-DMDc's derivative-augmented state. It is not cited by the base paper.
43. **A. Maksakov et al.**, "Data-driven modeling for damping and positioning control of gantry crane," *Mech. Syst. Signal Process.* (2023) 110368. DOI: 10.1016/j.ymssp.2023.110368. **[VERIFIED]**
    - Koopman model for flexible-structure damping, validated in simulation and on a laboratory crane.
44. **H.-K. Jiang et al.**, "Model predictive control of fluid–structure interaction via Koopman-based reduced-order model," *J. Fluid Mech.* (2026). DOI: 10.1017/jfm.2025.11035. **[VERIFIED]**
    - Kernel DMD plus residual DMD give switched reduced models for MPC of flapping plates.
    - Only three structural probes were enough. Matches reinforcement-learning performance at lower computational cost.
45. **D. Bruder et al.**, "Data-driven control of soft robots using Koopman operator theory," *IEEE Trans. Robot.* (2021). DOI: 10.1109/tro.2020.3038693. **[VERIFIED]**
    - Koopman MPC is more than 3× more accurate than MPC on a linear state-space model.
46. **R. Strässer et al.**, "An overview of Koopman-based control: from error bounds to closed-loop guarantees," *Annu. Rev. Control* (2025) 101035. DOI: 10.1016/j.arcontrol.2025.101035. **[VERIFIED]**
47. **P. Bevanda et al.**, "Koopman operator dynamical models: learning, analysis and control," *Annu. Rev. Control* (2021). DOI: 10.1016/j.arcontrol.2021.09.002. **[VERIFIED]**

---

## 2. Q2: secondary-path (or AVC/ANC plant) identification via DMD, DMDc or another data-driven ROM

**Finding:**
- No paper other than the base paper (item 2) uses DMD or DMDc to identify the secondary or compensation path for FxLMS. Searches covered Consensus (three formulations), WebSearch and SciSpace.
- Secondary paths in the AVC literature are identified with offline FIR/LMS, online auxiliary-noise LMS, ARX/NARX, impact-hammer FRFs, OKID/ERA or N4SID.

48. **F. Li, T. Yang, W. Li, M.J. Brennan, L. Wu, X. Li**, "Predicting the performance and optimizing the control parameters of an active vibration control system by pre-identifying the secondary path," *J. Sound Vib.* 568 (2024) 117953. DOI: 10.1016/j.jsv.2023.117953. **[VERIFIED]**
    - Splits the secondary path into actuator electronics FRF × structure (from impact tests) × AD/DA half-sample phase shift. The result is used to **predict FxLMS performance and optimise its parameters before hardware exists**.
    - Cited by the base paper as [28]. It is the closest precedent to the base paper's idea of tuning on an identified model.
49. **X.-J. Dong, G. Meng, J.-C. Peng**, "Vibration control of piezoelectric smart structures based on system identification technique: numerical simulation and experimental study," *J. Sound Vib.* 297 (2006) 680–693. DOI: 10.1016/j.jsv.2006.04.021. **[VERIFIED]**
    - **OKID** state-space model of a piezo cantilever plate, identified from FE-simulated and from experimental data, followed by LQG control. Cited by the base paper as [46].
    - *Analysis:* this is the classic OKID baseline the base paper should have compared sd-DMDc against.
50. **S.M. Hasheminejad et al.**, "Active vibration control of an arbitrary thick smart cylindrical panel with optimally placed piezoelectric sensor/actuator pairs," *Int. J. Mech. Mater. Des.* (2016). DOI: 10.1007/s10999-015-9293-2. **[VERIFIED]**
    - Frequency-domain subspace identification (**N4SID**) of a smart panel, followed by LQG control.
51. **Y.-X. Pu et al.**, "Multi-channel adaptive active vibration control of piezoelectric smart plate with online secondary path modelling using PZT patches," *Mech. Syst. Signal Process.* (2019). DOI: 10.1016/j.ymssp.2018.10.019. **[VERIFIED]**
52. **W. Niu et al.**, "Adaptive vibration suppression of time-varying structures with enhanced FxLMS algorithm," *Mech. Syst. Signal Process.* 118 (2019) 93–107. DOI: 10.1016/j.ymssp.2018.08.009. **[VERIFIED]**
    - Online secondary-path modelling without injected noise, plus a bang-bang variable step size.
53. **C.-S. Song et al.**, "Active vibration control using nonlinear auto-regressive neural network to identify secondary channel," *J. Low Freq. Noise Vib. Act. Control* (2023). DOI: 10.1177/14613484231186704. **[VERIFIED]**
    - NARX with quantum PSO beats ARX for secondary-channel identification.
54. **Z. Zhou et al.**, "Comparison between different modelling methods of secondary path to maximize control effect for active engine mounts," SAE 2021-01-0668. DOI: 10.4271/2021-01-0668. **[VERIFIED]**
    - Compares FIR, IIR and BP-neural-network secondary-path models for FxLMS.
55. **S. Lian et al.**, "Rapid and ambient-noise-robust synchronous secondary-path modeling via Kronecker product decomposition for multichannel active noise control," *Mech. Syst. Signal Process.* (2026) 114581. DOI: 10.1016/j.ymssp.2026.114581. **[VERIFIED]**
    - Low-rank (Kronecker) structure of the multichannel secondary-path matrix.
    - 81 paths modelled in 2 s with about −15 dB error in a room; 64 paths in 1 s with about −19 dB in a vehicle.
    - *Analysis:* the only low-rank-ROM idea for secondary paths found. Conceptually close to the SVD truncation in DMD.

**What a secondary-path model must get right (links identification accuracy to FxLMS):**

56. **S.D. Snyder et al.**, "The effect of transfer function estimation errors on the filtered-x LMS algorithm," *IEEE Trans. Signal Process.* (1994). DOI: 10.1109/78.285659. **[VERIFIED]**
    - A phase error of ±90° is the stability bound, but inside that bound there is **no simple relationship** between error and stability.
57. **I.T. Ardekani et al.**, "Effects of imperfect secondary path modeling on adaptive active noise control systems," *IEEE Trans. Control Syst. Technol.* (2012). DOI: 10.1109/tcst.2011.2161762. **[VERIFIED]**
    - Stochastic analysis of stability, steady-state error and convergence speed under model mismatch. Intentional misadjustment of the model can *improve* performance.
58. **P.A.C. Lopes, M.D. Piedade**, "The behavior of the modified FX-LMS algorithm with secondary path modeling errors," *IEEE Signal Process. Lett.* (2004). DOI: 10.1109/lsp.2003.821745. **[PARTIAL]** (second author from memory)
    - For the modified FxLMS, stability requires Re{Ŝ/S} > 1/2, i.e. phase error below 60° when amplitude is exact.
59. **M. Wu et al.**, "The statistical behavior of phase error for deficient-order secondary path modeling," *IEEE Signal Process. Lett.* (2008). DOI: 10.1109/lsp.2007.914946. **[VERIFIED]**
    - Relates the order of the secondary-path model to the resulting phase error. *Analysis:* directly relevant to how the DMD rank (p, r) truncation maps to FxLMS risk.
60. **H.-J. Jeon et al.**, "Analysis of frequency mismatch in narrowband active noise control," *IEEE Trans. Audio Speech Lang. Process.* (2010). DOI: 10.1109/tasl.2009.2038815. **[VERIFIED]**
    - Gives the convergence rate and noise reduction as functions of step size, phase error and frequency mismatch. *Analysis:* the base paper's tonal cases (85, 120, 280 Hz) are exactly this narrowband setting.
61. **F.-R. Yang et al.**, "Stochastic analysis of the filtered-x LMS algorithm for active noise control," *IEEE/ACM Trans. Audio Speech Lang. Process.* (2020). DOI: 10.1109/taslp.2020.3012056. **[VERIFIED]**
    - Mean and mean-square step-size bounds for an arbitrary reference signal and a general secondary path.

---

## 3. Q3: papers citing Ouyang et al. 2025 (sd-DMDc) or the 2026 TWS paper

- **sd-DMDc (Appl. Math. Model. 140, 115866):** Consensus metadata gives 4 citations. Two were identified:
  - the base paper itself (item 2);
  - the Engineering Structures 2026 PPF paper (item 3).
  - The other two could not be identified, because forward-citation databases were unreachable (OpenAlex and Semantic Scholar blocked; Google Scholar not queryable). **[UNVERIFIED]**
- **TWS 228 (2026) 115090:** Consensus gives 0 citations. Nothing citing it was found by WebSearch, Consensus or SciSpace.
- **Conclusion:** every follow-up found comes from the same group. *Analysis:* the sd-DMDc-for-AVC niche is open to an independent group, which is useful for novelty but means there is no external validation of sd-DMDc. Confidence is high (about 85%) that no independent paper had evaluated sd-DMDc by 2026-10-16.

---

## 4. Q4: classic alternatives the base paper did NOT compare against

### 4a. Realization and subspace methods (time domain)

62. **J.-N. Juang, R.S. Pappa**, "An eigensystem realization algorithm for modal parameter identification and model reduction," *J. Guid. Control Dyn.* 8(5) (1985) 620–627. DOI: 10.2514/3.20031. **[VERIFIED]**
63. **J.-N. Juang et al.**, "Effects of noise on modal parameters identified by the eigensystem realization algorithm," *J. Guid. Control Dyn.* (1986). DOI: 10.2514/3.20106. **[VERIFIED]**
    - Uses the singular values and modal amplitude coherence of the data matrix to separate noise from system modes. *Analysis:* a principled alternative to sd-DMDc's ad hoc choice of p and r.
64. **J.-N. Juang et al.**, "Identification of observer/Kalman filter Markov parameters: theory and experiments," *J. Guid. Control Dyn.* DOI: 10.2514/3.21006. **[PARTIAL]** (Consensus lists the year as 1991; the journal year was not seen)
    - **OKID**: with noisy data, the deadbeat observer becomes the Kalman filter. Demonstrated on Hubble telescope data.
65. **M. Phan et al.**, "Improvement of observer/Kalman filter identification (OKID) by residual whitening," *J. Vib. Acoust.* (1992). DOI: 10.1115/1.2873927. **[VERIFIED]**
66. **F. Vicario et al.**, "Output-only observer/Kalman filter identification (O3KID)," *Struct. Control Health Monit.* (2015). DOI: 10.1002/stc.1719. **[VERIFIED]**
    - Also relevant: F. Vicario's PhD thesis, "OKID as a general approach to linear and bilinear system identification" (2014), DOI 10.7916/d8ws8rvc. **[VERIFIED]**
67. **P. Van Overschee, B. De Moor**, "N4SID: subspace algorithms for the identification of combined deterministic-stochastic systems," *Automatica* 30(1) (1994) 75. DOI: 10.1016/0005-1098(94)90230-5. **[VERIFIED]**
    - Non-iterative, using only QR and SVD. The state sequences are outputs of Kalman filter banks.
68. **M. Abdelghani, M. Verhaegen, P. Van Overschee, B. De Moor**, "Comparison study of subspace identification methods applied to flexible structures," *Mech. Syst. Signal Process.* (1998). DOI: 10.1006/mssp.1998.0169. **[PARTIAL]** (co-authors other than Abdelghani from memory)
    - On the NASA Mini-Mast five-mode model, **N4SID and MOESP gave better transfer-function and output estimates than ERA/OKID**, under both white and coloured noise.
69. **W. Favoreel et al.**, "Subspace state space system identification for industrial processes," *J. Process Control* (2000). DOI: 10.1016/s0959-1524(99)00030-x. **[PARTIAL]** (year listed as 1998 by Consensus)
    - Compares subspace methods with PEM on 10 industrial data sets.
70. **A. Chiuso et al.**, "On the ill-conditioning of subspace identification with inputs," *Automatica* (2004). DOI: 10.1016/j.automatica.2003.11.009. **[VERIFIED]**
    - N4SID degrades under lack of excitation. *Analysis:* relevant because sinusoidal or narrowband excitation, as in the base paper's tonal cases, is a poor identification input.
71. **Ş.İ. Lök et al.**, "A numerical and experimental investigation of the most fundamental time-domain input–output system identification methods for the normal modal analysis of flexible structures," *Sensors* 25 (2025) 1259. DOI: 10.3390/s25041259. **[VERIFIED]**
    - Head-to-head comparison of ARX, SSEST (PEM), N4SID, ERA/OKID and TFEST on a flexible test rig, plus mass/stiffness/damping recovery.
    - *Analysis:* the closest template for a benchmark section, but it does not include DMDc.
72. **M. Chang, S.N. Pakzad**, "Observer Kalman filter identification for output-only systems using interactive Structural Modal Identification Toolsuite (SMIT)," *J. Bridge Eng.* (2014). DOI: 10.1061/(asce)be.1943-5592.0000530. **[PARTIAL]** (second author from memory)
    - SMIT toolbox comparing ERA variants, ARX, SRIM and N4SID.
73. **G. Wagner et al.**, "Benefits of OKID in system realization with low amount of samples," *J. Vib. Eng. Technol.* (2022). DOI: 10.1007/s42417-022-00671-0. **[VERIFIED]**

### 4b. Prediction error method (PEM)

74. **L. Ljung**, "Prediction error estimation methods," *Circuits Syst. Signal Process.* (2002). DOI: 10.1007/bf01211648. **[VERIFIED]**
    - Also relevant: Ljung's *System Identification: Theory for the User* (2nd ed., 1999), seen through reviews: DOI 10.1109/mra.2012.2192817 and 10.1016/s0005-1098(01)00214-x. **[VERIFIED]**
    - *Analysis:* standard identification theory says that pre-filtering inputs *and* outputs with the same filter preserves the input-output relation. sd-DMDc low-pass filters the measurement vector s_k (paper text, §2.2). The paper does not state that the inputs are filtered identically. This is a Phase-1 audit item.

### 4c. Frequency-domain rational fitting (FRF-based)

75. **B. Gustavsen et al.**, "Rational approximation of frequency domain responses by vector fitting," *IEEE Trans. Power Deliv.* (1999). DOI: 10.1109/61.772353. **[VERIFIED]**
76. **B. Gustavsen**, "Improving the pole relocating properties of vector fitting," *IEEE Trans. Power Deliv.* (2006). DOI: 10.1109/tpwrd.2005.860281. **[VERIFIED]**
    - Relaxed vector fitting, with a major improvement on noisy responses.
77. **S. Grivet-Talocia, M. Bandinu**, "Improving the convergence of vector fitting for equivalent circuit extraction from noisy frequency responses," *IEEE Trans. Electromagn. Compat.* (2006). DOI: 10.1109/temc.2006.870814. **[PARTIAL]** (second author from memory)
    - Spurious poles from noise, and the VF-AS fix.
78. **Z. Drmač, S. Gugercin, C. Beattie**, "Vector fitting for matrix-valued rational approximation," *SIAM J. Sci. Comput.* (2015). DOI: 10.1137/15m1010774. **[PARTIAL]** (co-authors from memory)
    - The MIMO extension, which is what the 2-actuator × 2-sensor paths require.
79. **B. Peeters, H. Van der Auweraer, P. Guillaume, J. Leuridan**, "The PolyMAX frequency-domain method: a new standard for modal parameter estimation?" *Shock Vib.* (2004). DOI: 10.1155/2004/523692. **[PARTIAL]** (co-authors from memory)
    - Equivalent to time-delay DMDc according to Si et al. 2026 (item 14).

### 4d. Loewner framework (interpolatory, data-driven)

80. **A.J. Mayo et al.**, "A framework for the solution of the generalized realization problem," *Linear Algebra Appl.* (2007). DOI: 10.1016/j.laa.2007.03.008. **[VERIFIED]**
81. **A.C. Antoulas, S. Lefteriu, A.C. Ionita**, "A tutorial introduction to the Loewner framework for model reduction," Chapter 8 in *Model Reduction and Approximation*, SIAM (2017). DOI: 10.1137/1.9781611974829.ch8. **[PARTIAL]** (co-authors from memory)
82. **G. Dessena et al.**, "A Loewner-based system identification and structural health monitoring approach for mechanical systems," *Struct. Control Health Monit.* (2023) 1891062. DOI: 10.1155/2023/1891062. **[VERIFIED]**
83. **G. Dessena et al.**, "The accuracy and computational efficiency of the Loewner framework for the system identification of mechanical systems," *Aerospace* 10 (2023) 571. DOI: 10.3390/aerospace10060571. **[VERIFIED]**
    - Loewner is more accurate than LSCE and computationally faster than **N4SID** on numerical and experimental data.
84. **Z. Drmač, B. Peherstorfer**, "Learning low-dimensional dynamical-system models from noisy frequency-response data with Loewner rational interpolation," (2019/2022). DOI: 10.1007/978-3-030-95157-3_3. **[PARTIAL]** (second author from memory)
    - The error grows at most linearly with the standard deviation of Gaussian noise, with high probability.
85. **I.V. Gosea, C. Poussot-Vassal, A.C. Antoulas**, "Data-driven modeling and control of large-scale dynamical systems in the Loewner framework," *Handbook of Numerical Analysis* (2022). DOI: 10.1016/bs.hna.2021.12.015. **[PARTIAL]** (co-authors from memory)
    - Includes data-driven feedback controller design.

### 4e. Sparse identification (SINDy with control)

86. **S.L. Brunton et al.**, "Sparse identification of nonlinear dynamics with control (SINDYc)," *IFAC-PapersOnLine* (2016). DOI: 10.1016/j.ifacol.2016.10.249. **[VERIFIED]**
    - Explicitly connects SINDYc to DMD and Koopman.
87. **E. Kaiser et al.**, "Sparse identification of nonlinear dynamics for model predictive control in the low-data limit," *Proc. R. Soc. A* (2018; Consensus lists 2017). DOI: 10.1098/rspa.2018.0335. **[VERIFIED]**
    - SINDy-MPC needs less data and is more robust to noise than neural networks. The paper notes that linear data-driven models are a stopgap until enough data are available.
88. **U. Fasel et al.**, "Ensemble-SINDy: robust sparse model discovery in the low-data, high-noise limit, with active learning and control," *Proc. R. Soc. A* (2022). DOI: 10.1098/rspa.2021.0904. **[VERIFIED]**
    - Bagging gives inclusion probabilities. Recovers PDEs with more than 2× the previously reported noise levels.
89. **K. Kaheman et al.**, "SINDy-PI: a robust algorithm for parallel implicit sparse identification of nonlinear dynamics," *Proc. R. Soc. A* (2020). DOI: 10.1098/rspa.2020.0279. **[VERIFIED]**
    - *Analysis:* for a *linear* plate, SINDYc reduces to least-squares linear regression on [x, ẋ, u]. That is essentially sd-DMDc without the SVD truncation, so SINDYc is mainly relevant if MFC or hysteresis nonlinearity is added.

### 4f. Known links between DMD and classical identification

90. **Z. Ma et al.**, "Reduced-order models for control of fluids using the eigensystem realization algorithm," *Theor. Comput. Fluid Dyn.* (2011). DOI: 10.1007/s00162-010-0184-8. **[PARTIAL]** (journal year from DOI; Consensus lists 2008)
    - **ERA gives theoretically the same reduced models as balanced POD**, without adjoint information.
- Further links established elsewhere in this bibliography:
  - Tu et al. 2014 (item 6): DMD↔ERA and DMD = LIM.
  - Proctor et al. 2016 §V (item 4): DMDc↔ERA/OKID.
  - Annoni & Seiler (item 26): ioDMD includes a direct subspace-identification step.
  - Si et al. 2026 (item 14): time-delay DMDc ≡ pLSCF through ARX.
  - Li et al. 2024 (item 15): DMD, ITD and ERA are special cases of characteristic value decomposition (CVD).
  - Saito & Kuno 2020 (item 13): DMD ~ ITD.
  - Frame & Towne 2023 (item 23): Hankel SVD = space-time POD.
- *Analysis:* together these imply that delay-DMDc, ERA/OKID, N4SID, ARX and pLSCF form one family of shift-invariance and least-squares estimators. sd-DMDc chooses a particular *linear* augmentation: a numerical derivative is a fixed linear combination of neighbouring samples, so the augmented state [s_k; Λ·d_k] is a linear map of a short Hankel window, before low-pass filtering. **No paper was found that writes sd-DMDc in this form or compares it with ERA/OKID or N4SID at equal model order.**

---

## 5. Q5: effect of noise on DMD, and noise-robust variants

91. **S.T.M. Dawson, M.S. Hemati, M.O. Williams, C.W. Rowley**, "Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition," *Exp. Fluids* (2016). DOI: 10.1007/s00348-016-2127-7. **[VERIFIED]**
    - Shows analytically that **DMD is biased by sensor noise**, with the bias depending on data size and noise level.
    - Proposes three fixes: direct bias correction, **forward-backward DMD**, and TLS DMD.
    - sd-DMDc's "extended forward-backward strategy" descends from this paper.
92. **M.S. Hemati, C.W. Rowley, E.A. Deem, L.N. Cattafesta**, "De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets," *Theor. Comput. Fluid Dyn.* (2017). DOI: 10.1007/s00162-017-0432-2. **[PARTIAL]** (co-authors other than Hemati from memory)
    - Total DMD (TDMD): the projection step is biased because it treats snapshots asymmetrically. The fix is TLS on an augmented snapshot matrix.
93. **T. Askham et al.**, "Variable projection methods for an optimized dynamic mode decomposition," *SIAM J. Appl. Dyn. Syst.* (2018; Consensus lists 2017). DOI: 10.1137/m1124176. **[VERIFIED]**
    - optDMD shows less bias under noise and handles non-uniform sampling.
94. **D. Sashidhar et al.**, "Bagging, optimized dynamic mode decomposition for robust, stable forecasting with spatial and temporal uncertainty quantification," *Phil. Trans. R. Soc. A* (2022; Consensus lists 2021). DOI: 10.1098/rsta.2021.0199. **[VERIFIED]**
    - BOP-DMD: an ensemble of optDMD fits provides uncertainty quantification on eigenvalues and modes.
    - *Analysis:* this would give a distribution of identified secondary paths, and hence a distribution of predicted FxLMS convergence times.
95. **N. Takeishi, Y. Kawahara, T. Yairi**, "Subspace dynamic mode decomposition for stochastic Koopman analysis," *Phys. Rev. E* (2017). DOI: 10.1103/physreve.96.033310. **[PARTIAL]** (co-authors from memory)
    - Projects future snapshots onto past ones (an instrumental-variable or subspace idea). Consistent under observation noise.
96. **T. Nonomura et al.**, "Dynamic mode decomposition using a Kalman filter for parameter estimation," *AIP Adv.* (2018). DOI: 10.1063/1.5031816. **[VERIFIED]**
    - KF-DMD beats tlsDMD under severe noise when the noise statistics are known. tlsDMD is better at low and medium noise.
97. **T. Nonomura et al.**, "Extended-Kalman-filter-based dynamic mode decomposition for simultaneous system identification and denoising," *PLoS ONE* (2018). DOI: 10.1371/journal.pone.0209836. **[VERIFIED]**
98. **L. Jiang, N. Liu**, "Correcting noisy dynamic mode decomposition with Kalman filters," *J. Comput. Phys.* (2022) 111175. DOI: 10.1016/j.jcp.2022.111175. **[PARTIAL]** (second author from memory)
    - Derives the maximum-likelihood estimator of the DMD matrix and proposes KF and EnKF DMD.
99. **Y. Ohmichi et al.**, "Stable dynamic mode decomposition algorithm for noisy pressure-sensitive paint measurement data," *AIAA J.* (Consensus lists 2021; journal year not confirmed). DOI: 10.2514/1.j061086. **[VERIFIED]**
    - Truncated TLS DMD.
    - Standard and exact DMD eigenvalues **shift toward the damped side** under noise, i.e. noise inflates the apparent damping. *Analysis:* this matters for lightly damped CFRP plates.
100. **A. Das et al.**, "Transonic shock buffet flowfield assessment using various dynamic mode decomposition techniques," *J. Aircraft* (2023). DOI: 10.2514/1.c037082. **[VERIFIED]**
     - Benchmarks standard, forward-backward, TLS, higher-order and robust DMD under white and salt-and-pepper noise. Higher-order DMD is best above 5% noise variance.
101. **Z.-Y. Wu et al.**, "Challenges in dynamic mode decomposition," *J. R. Soc. Interface* (2021). DOI: 10.1098/rsif.2021.0686. **[VERIFIED]**
     - With noise and mild nonlinearity, DMD variants often fail for oscillators whose transients decay only slightly faster than one period. *Analysis:* lightly damped plates are exactly this case.
102. **L. Lortie, J.R. Forbes**, "Asymptotically stable data-driven Koopman operator approximation with inputs using total extended DMD," *Mach. Learn.: Sci. Technol.* (2024). DOI: 10.1088/2632-2153/ada33b. **[PARTIAL]** (second author from memory)
     - TLS with inputs plus an LMI stability constraint. Compared with forward-backward EDMD.
103. **K. Aishima**, "Strong consistency of the projected total least squares dynamic mode decomposition for datasets with random noise," *Jpn. J. Ind. Appl. Math.* (2022). DOI: 10.1007/s13160-022-00547-6. **[VERIFIED]**
104. **P.J. Baddoo, B. Herrmann, B.J. McKeon, J.N. Kutz, S.L. Brunton**, "Physics-informed dynamic mode decomposition," *Proc. R. Soc. A* (2023). DOI: 10.1098/rspa.2022.0576. **[PARTIAL]** (co-authors from memory)
     - piDMD as a Procrustes problem with conservation, self-adjointness and similar constraints. Fewer degrees of freedom make it less noise-prone.
     - *Analysis:* enforcing a second-order (M, C, K)-consistent or stable structure on the plate model is not done in sd-DMDc.
105. **O. Azencot et al.**, "Consistent dynamic mode decomposition," *SIAM J. Appl. Dyn. Syst.* (2019). DOI: 10.1137/18m1233960. **[VERIFIED]**
     - Forward-backward consistency regularisation solved with ADMM.
106. **M.J. Colbrook et al.**, "Residual dynamic mode decomposition: robust and verified Koopmanism," *J. Fluid Mech.* (2023; Consensus lists 2022). DOI: 10.1017/jfm.2022.1052. **[VERIFIED]**
     - Uses residuals to flag spurious modes. *Analysis:* could replace rank picking by response error.
107. **W. Bennett et al.**, "Weak-DMD: a Galerkin approach to the problem of noise in the dynamic mode decomposition algorithm," arXiv:2604.14350 (2026). DOI: 10.48550/arxiv.2604.14350. **[VERIFIED]**
     - A weak (integral) form that avoids differentiation and timestep sensitivity and filters noise.
     - *Analysis:* a direct alternative to sd-DMDc's "low-pass filter, then numerically differentiate, then scale" step.
108. **M. Wanner, I. Mezić**, "Robust approximation of the stochastic Koopman operator," *SIAM J. Appl. Dyn. Syst.* (Consensus lists 2020; journal year not confirmed). DOI: 10.1137/21m1414425. **[PARTIAL]** (second author from memory)
109. **R. Chartrand**, "Numerical differentiation of noisy, nonsmooth data," *ISRN Appl. Math.* (2011). DOI: 10.5402/2011/164564. **[VERIFIED]**
     - Total-variation-regularised differentiation.
110. **F. van Breugel et al.**, "Numerical differentiation of noisy data: a unifying multi-objective optimization framework," *IEEE Access* (2020). DOI: 10.1109/access.2020.3034077. **[VERIFIED]**
     - Python package PyNumDiff (JOSS 2022, DOI 10.21105/joss.04078). **[VERIFIED]**
     - One hyperparameter, chosen from the power spectrum. *Analysis:* a principled replacement for sd-DMDc's filter and derivative step.
111. **J. Stevens-Haas et al.**, "Learning nonlinear dynamics using Kalman smoothing," *IEEE Access* (2024). DOI: 10.1109/access.2024.3465390. **[VERIFIED]**
     - Kalman smoothing was the best or tied-best derivative estimator on 7 of 8 systems inside pysindy.

---

## 6. Synthesis: what the literature says about the base paper's modelling step

1. **The method family.**
   - DMDc, delay-DMDc, ERA/OKID, N4SID/MOESP, ARX/PEM and pLSCF/PolyMAX are formally connected. See Tu 2014, Proctor 2016 §V, Annoni & Seiler 2017, Ma et al. 2011, Li et al. 2024 (CVD), and Si et al. 2026, who prove time-delay DMDc ≡ pLSCF.
   - sd-DMDc was benchmarked **only against delay-DMDc** (sd-DMDc abstract) and against an FEM model (TWS paper).
   - The closest structural benchmark in the literature, Abdelghani et al. 1998, found N4SID/MOESP better than ERA/OKID for a flexible truss.
   - **No study places sd-DMDc or DMDc in a benchmark against ERA/OKID, N4SID, PEM, VF or Loewner on a piezo-composite plate.**
2. **Noise.**
   - Standard DMD is biased by measurement noise (Dawson 2016; Hemati 2017). Damping is overestimated, as eigenvalues shift toward the damped side (Ohmichi 2022), and large errors break damping estimates first (Saito & Kuno 2020).
   - Established fixes: forward-backward, TLS, optDMD/BOP-DMD, subspace DMD, KF-DMD, piDMD, residual DMD and weak DMD.
   - sd-DMDc uses only an *extended forward-backward* strategy, plus a low-pass filter and derivative scaling.
   - The base paper adds a closed-loop noise test (band-limited noise injected down to about 45 dB SNR). It never studied the *identification-noise → path error → FxLMS convergence-time* chain.
3. **What matters for FxLMS is phase accuracy at the reference frequency, not time-domain fit.**
   - Stability needs |phase error| < 90° (Snyder & Hansen 1994). Inside that bound the relationship to convergence is not simple.
   - Convergence rate depends on step size, phase error and frequency mismatch (Jeon 2010; Ardekani & Abdulla 2012).
   - The model order of the secondary path controls phase error (Wu et al. 2008).
   - The base paper validates sd-DMDc by response reconstruction and modal parameters. It does not report phase error of Ŝ(e^{jω}) at 85, 120 or 280 Hz.
4. **Model-based tuning of AVC parameters before hardware exists** has a direct precedent (F. Li et al. 2024 JSV, item 48). The base paper cites it as [28].
   - Numerical-only DMDc-for-AVC papers are publishable: Wan et al. 2024 (Eng. Struct.), Rains et al. 2023 (JCP), Mendez Gaona 2026 (arXiv).
5. **Same-group continuation.** The Engineering Structures 2026 PPF paper (item 3) shows the authors moving to *feedback* control. The FxLMS-specific questions remain open: identification-to-convergence error propagation, comparison with classical identification, and robust step-size bounds.

---

## 7. GAPS: what has NOT been done, most relevant to a numerical-only project

Confidence values are my estimates of the probability that the gap is real (that no published work fills it as of 2026-10-16), given the searches above. They are not statistics.

| # | Gap | Evidence for it | Confidence | Feasible on a laptop? |
|---|-----|-----------------|-----------|-----------------------|
| G1 | **No benchmark of sd-DMDc (or any DMDc) against ERA/OKID, N4SID, PEM, vector fitting or Loewner** for identifying the primary, secondary and compensation paths of a piezo-composite plate. The benchmark should be at equal model order, with the noise level controlled. | sd-DMDc was compared only to delay-DMDc. The nearest benchmarks are Lök 2025 and Abdelghani 1998, which both omit DMDc. Vega-Zambrano 2026 and Alassaf 2025 compare DMDc with N4SID or PEM, but in non-structural domains. | 85% | Yes. A Ritz plate as the virtual truth, plus Python implementations of ERA/OKID, N4SID (UNVERIFIED which Python package to use), PyDMD and vector fitting. |
| G2 | **No study links identification error to FxLMS convergence-time maps.** The chain to quantify: (identification noise / SNR) → (phase or magnitude error of Ŝ at f_ref) → (stability margin and t_5% over the μ–L grid). | FxLMS theory gives the ±90° bound and stochastic analyses (Snyder & Hansen, Ardekani & Abdulla, Jeon), but always for a *given* model error. No paper drives it with a data-driven ROM. The base paper shows FEM vs sd-DMDc maps, but not as a function of identification error. | 80% | Yes. Monte Carlo over SNR (e.g. 20–60 dB, around the paper's 45 dB) and over rank (p, r). |
| G3 | **No formal analysis of sd-DMDc itself**, on four points: (a) its equivalence to a linearly transformed short-Hankel (delay) DMDc; (b) the bias introduced by the amplitude-matching scaling Λs (*Analysis:* for a single tone ‖d‖/‖s‖ ≈ ω, so Λs ≈ 1/ω_dominant per channel, which makes the scaling frequency-dependent); (c) the effect of low-pass filtering the outputs but not, as far as stated, the inputs (phase lag absorbed into A and B); (d) whether its improvement over delay-DMDc survives against TLS, optDMD or subspace DMD. | No paper found. Si et al. 2026 argue the opposite, in favour of delays. Mamakoukas 2021 is the only paper using derivative bases, and it is in robotics. | 75% | Yes. Purely analytical and numerical. |
| G4 | **Noise-robust DMD variants are untested on smart-structure input-output identification for AVC.** The variants are tlsDMDc, optDMD with control (cOptDMDc), BOP-DMD, piDMD with stability or second-order structure, and weak DMD. | They were applied in fluids, vehicles, aeroelastic and pharmaceutical settings (items 29, 35, 40, 94, 104, 107). None is applied to piezo or MFC plates with FxLMS. | 70% | Yes. PyDMD provides BOP-DMD and piDMD; cOptDMDc would have to be implemented. |
| G5 | **No uncertainty-aware FxLMS tuning from a data-driven model.** The idea: use BOP-DMD ensembles to produce probabilistic (μ, L) "effective-convergence" regions, or robust step-size bounds, instead of a single deterministic map. | BOP-DMD gives uncertainty quantification (item 94), and FxLMS bounds exist (item 61). Combining them was not found. | 80% | Yes. |
| G6 | **No secondary-path identification method that exploits the low-rank structure of the multichannel path matrix** is applied to structural AVC. The candidates are DMD's SVD truncation and Kronecker factors. | Lian 2026 (item 55) does this for acoustic ANC only. | 65% | Yes, but it matters only for the 2×2 case. Stronger if more channels are simulated. |
| G7 | **The 4-D (μ1, μ2, L1, L2) optimisation, stated as future work by the base paper, has not been done on any identified model.** | Base paper §6. The authors' follow-up (item 3) moved to PPF. | 75% | Yes. Cheap on an identified model; PSO matches the user's skills. |
| G8 | **Koopman/EDMD or SINDYc for composite-plate AVC with MFC nonlinearity** (hysteresis, as modelled by Umar et al. 2024 with ARX plus Bouc-Wen). | Koopman work is concentrated in robotics, cranes and fluid-structure interaction (items 42–45). | 60% | Only if a nonlinear actuator model is added to the virtual testbed. Lower priority. |

**Ranking for a numerical-only kernel (my recommendation):**
1. **G2 + G1 combined** has the highest value. Build a virtual truth plate, then identify its paths with sd-DMDc, delay-DMDc, ERA/OKID, N4SID and a noise-robust DMDc (TLS or optDMD). Map identification error to FxLMS t_5% over (μ, L), and declare falsifiable hypotheses before the runs. An example hypothesis: "at SNR ≥ 40 dB, sd-DMDc's phase error at f_ref is no smaller than N4SID's at equal order."
2. **G3** is a theory section that makes the kernel publishable even if sd-DMDc wins.
3. **G5 or G7** are extensions once the pipeline exists.

All of these need only Python (numpy/scipy, PyDMD, pysindy, PyNumDiff) and the Chebyshev-Ritz plate and patch model the user already has.

---

## 8. Items that need verification before citing in a manuscript

- The 2 unidentified citing papers of sd-DMDc (Consensus count 4; 2 found). **[UNVERIFIED]**
- Full author lists marked "from memory" in items 5, 11, 28, 58, 68, 72, 77, 78, 79, 81, 84, 85, 92, 95, 98, 102, 104 and 108. The DOIs and titles were seen; check the authors on the publisher page.
- The exact title and DOI of the AugDMDc COMPDYN 2025 paper (item 39). **[PARTIAL]**
- The authors of arXiv:2602.06944 (item 38). **[PARTIAL]**
- Python N4SID / ERA implementations (e.g. SIPPY, nfoursid). Names recalled from memory, not seen in this sweep. **[UNVERIFIED]**
