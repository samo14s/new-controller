# Literature sweep C: time variation, robustness and advanced adaptive algorithms (the base paper's stated future work)

Prepared 2026-10-16 for the project `sddmdc_fxlms`. Base article: Q. Ouyang, L. Dong, J. Liu, J. Zhou, *Thin-Walled Structures* 228 (2026) 115090, DOI 10.1016/j.tws.2026.115090.

Companion sweeps cover related ground, and this file does not repeat them:
- `lit_A_dmdc.md`: DMD/DMDc/Koopman for structures.
- `lit_B_fxlms.md`: FxLMS theory and the analytical (μ, L) convergence map.

## How this sweep was run (read first)

**Tools that worked:**
- Consensus `search`: 28 queries.
- SciSpace `search-papers`: 2 novelty checks.
- Wiley Scholar Gateway `semanticSearch`: 2 full-text passage queries.
- WebSearch: 2 queries.

**Tools that failed.** WebFetch could not resolve any host (`getaddrinfo ENOTFOUND api.crossref.org`). As a result:
- No full text could be opened.
- Every annotation rests on the abstract or passage returned by a search tool.
- No forward-citation query was possible. Citation counts are Consensus metadata as of 2026-10-16.

**Tags:**
- **[VERIFIED]**: title, first author, venue and DOI (or arXiv ID / URL) were all shown in a tool result in this session. Tools usually print only the first author plus "et al.". Co-authors are given only when a tool displayed them, when the task prompt named them, or when they appear in the base paper's reference list; each such case is marked.
- **[PARTIAL: field]**: the item was seen, but one field was missing or inconsistent. The field is named.
- **[UNVERIFIED]**: the item was not seen in any tool result.

*Analysis:* marks my own reasoning. It is not a claim taken from the cited paper.

**Follow-up check.** The base paper has **0 citations** in Consensus. A sister paper by the same group was found:
- Ouyang Q. et al., "Identification of electromechanical participation matrices via scaled-derivative DMDc for improved PPF active vibration control of thin-walled structures", *Engineering Structures* (2026), DOI 10.1016/j.engstruct.2026.122232, 3 citations, abstract not shown. **[VERIFIED]**
- No third-party follow-up of the base paper was found.

---

## 0. What the base paper itself says about robustness and future work (from the extracted text)

**Limitations, verbatim** (paper_text.txt lines 1472–1478): "the current framework is implemented using a fixed-step FxLMS algorithm without incorporating advanced adaptive or noise-mitigation strategies … the experimental validation is limited to a relatively simple plate structure … Future work will therefore focus on integrating advanced adaptive algorithms, extending the framework to more complex composite structures, and further evaluating its robustness under more demanding dynamic conditions."

**4-D tuning** (lines 1411–1418):
- A manual grid search over (μ1, μ2, L1, L2) is called "computationally prohibitive".
- Future work will "integrate advanced optimization algorithms to efficiently navigate the 4D parameter space".
- Differentiated parameters already reached 3.48 s (simulation) and 3.81 s (experiment). This beats the identical-parameter case.

**What is missing from the text:**
- The secondary path is identified **once, offline**. No online secondary-path modelling (SPM) is used.
- No uncertainty band is given on the identified model or on the predicted convergence map.
- No temperature or time-variation study is made.
- Noise robustness is tested only by injecting band-limited white noise into the simulated loop (lines 1033–1053).

**State dimension:**
- The hybrid state is s_k = [x_k; ε_k] ∈ R^n, with n = n1 + n2.
- With 35 laser points and 2 strain sensors, the scaled-derivative state has 2n ≈ 74 entries. *Analysis:* this is below the "about 200" state dimension under which Zhang et al.'s online DMD is the most efficient real-time algorithm (item 1.1). Online updating is therefore computationally plausible at this size.

**The base paper already cites the closest prior art** (its refs [26]–[29]):
- Liu T. et al. 2024 (MSSP, model-based FxLMS on an MFC composite plate).
- Chen X. et al. 2025 (MSSP, Fx-NLMS on a rotating composite beam; filter order and learning factor studied).
- Li F. et al. 2024 (JSV, predicting performance and optimising control parameters from a pre-identified secondary path).
- Zhang Y. et al. 2025 (MSSP, bolted composite plates under variable loads).

Any new project must position itself against these four.

---

## 1. Online, recursive and windowed DMD/DMDc for time-varying systems, and their use for updating a control model

1.1 **Zhang H., Rowley C.W., Deem E.A., Cattafesta L.N.**, "Online dynamic mode decomposition for time-varying systems", *SIAM J. Appl. Dyn. Syst.* DOI 10.1137/18m1192329. 206 citations. Author list per task prompt. **[PARTIAL: year]** Consensus lists 2017, the preprint year; the task prompt gives 2019.
- Computes the exact DMD matrix with rank-1 updates and stores no past data.
- An exponential weighting factor forgets old data, and a windowed variant is included.
- The most efficient real-time DMD for state dimension below about 200, and orders of magnitude faster than batch DMD.
- *Relevance:* the obvious engine for updating the sd-DMDc A, B matrices as the plate drifts. The rank-1 update extends directly to DMDc by stacking [x; u].

1.2 **Alfatlawi M. et al.**, "An incremental approach to online dynamic mode decomposition for time-varying systems with applications to EEG data modeling", *J. Comput. Dyn.* (2019/2020), DOI 10.3934/jcd.2020009. **[VERIFIED]**
- Incremental-SVD online DMD, using a discount factor or a sliding window.
- Explicitly includes **non-autonomous (with-input, i.e. DMDc) algorithms**. These work even when the data matrix is singular.
- Tested on random linear time-varying systems.
- *Relevance:* the closest ready-made "online DMDc" formulation.

1.3 **Hamdan T.B. et al.**, "Tracking distributed parameters system dynamics with recursive dynamic mode decomposition with control", *SIAM J. Appl. Dyn. Syst.* (2023), DOI 10.1137/22m1478665. **[VERIFIED]**
- DMDc initialisation, followed by a recursive-least-squares (RLS) update with a forgetting factor.
- Includes a recursive POD basis update.
- Output is a linear time-varying (LTV) reduced state-space model "usable for controller design".
- Test cases: heat equation with time-varying diffusivity, and the Burgers equation. Simulation only.
- *Relevance:* a direct template for a recursive sd-DMDc.

1.4 **Deem E.A. et al.**, "Adaptive separation control of a laminar boundary layer using online dynamic mode decomposition", *J. Fluid Mech.* (2020), DOI 10.1017/jfm.2020.546. 53 citations. **[VERIFIED]**
- Online DMD gives a time-varying linear model, from which LQR gains are recomputed recursively.
- Tested at 4 Reynolds numbers (0.9–1.25 × 10^5).
- Control is established in about 10 characteristic periods.
- *Relevance:* the **only verified case** of online DMD closing a control loop with model updates. It is a flow problem, not a structure, and LQR, not FxLMS.

1.5 **Hemati M.S. et al.**, "Dynamic mode decomposition for large and streaming datasets", *Phys. Fluids* (2014), DOI 10.1063/1.4901016. 297 citations. **[VERIFIED]**
- Streaming DMD, with an optional compression step that improves robustness to noisy measurements.

1.6 **Matsumoto D. et al.**, "On-the-fly algorithm for dynamic mode decomposition using incremental singular value decomposition and total least squares". **[PARTIAL: venue/year]** Consensus shows 2017 with DOI 10.1016/j.compfluid.2026.107265; these metadata are inconsistent.
- Incremental total-least-squares DMD. Total least squares reduces noise bias.

1.7 **Kale A. et al.**, "Efficient streaming dynamic mode decomposition", *IEEE Control Syst. Lett.* (2025), DOI 10.1109/lcsys.2025.3622516. **[VERIFIED]**
- Keeps a single orthonormal basis, giving a constant-factor saving in cost and memory.

1.8 **Guan W. et al.**, "Output-only modal identification with recursive dynamic mode decomposition for time-varying systems", *Measurement* (2024), DOI 10.1016/j.measurement.2023.113852. **[VERIFIED]**
- Weighted recursive DMD tracks time-varying modal parameters.
- Validated numerically and experimentally.

1.9 **Ma Z.-S. et al.**, "A novel recursive dynamic mode decomposition method for modal parameter identification of time-varying structures based on input-output data", *J. Vib. Control* (2025), DOI 10.1177/10775463251406526. **[VERIFIED]**
- Input-output recursive DMD built on PAST subspace tracking.
- Validated on a numerical time-varying-stiffness system and an **experimental abrupt-mass-change** system.
- *Relevance:* closest structural analogue of an "online DMDc" tracker.

1.10 **Chen B.-Q. et al.**, "Online physics-informed dynamic mode decomposition: theory and applications", *Proc. R. Soc. A* (2025), DOI 10.1098/rspa.2024.0437. **[VERIFIED]**
- Convex, online proximal-gradient formulation within a Bayesian-DMD framework.
- R² = 0.991 on the noisy Lorenz system.

1.11 **Chen B.-Q. et al.**, "Adaptive physics-informed system modeling with control for nonlinear structural system estimation" (APSMC), *Comput. Methods Appl. Mech. Eng.* (2025), DOI 10.1016/j.cma.2025.118330. **[VERIFIED]**
- Kalman filter plus physics-constrained online state-space update.
- On a scaled bridge experiment: 19 consecutive 10 s predictions from one 10 s update, minimum NMSE 0.398%.

1.12 **Mobini E. et al.**, "Data-driven model predictive control of nonlinear systems using dynamic mode decomposition", *Int. J. Syst. Sci.* (2025), DOI 10.1080/00207721.2025.2535538. **[VERIFIED]**
- Online DMD feeds an LTV model to MPC.
- The model is re-identified only when the tracking error exceeds a threshold, which avoids divergence from model fluctuations.
- *Relevance:* an "event-triggered re-identification" idea transferable to FxLMS.

1.13 **Prakash O. et al.**, "Real-time update of data-driven reduced and full order models with applications", *Comput. Chem. Eng.* (2024), DOI 10.1016/j.compchemeng.2024.108923. **[VERIFIED]**
- Two SVD-update schemes for DMD models: temporal updates and additive modifications.

1.14 **Liu N. et al.**, "Estimate of Koopman modes and eigenvalues with Kalman filter", *SIAM J. Sci. Comput.* (2026), DOI 10.1137/24m1684207. **[VERIFIED]**
- Ensemble-Kalman-filter update of DMD modes and eigenvalues from noisy, non-autonomous data.

1.15 **Wan H.-P. et al.**, "Data-driven model reduction approach for active vibration control of cable-strut structures", *Eng. Struct.* (2024), DOI 10.1016/j.engstruct.2023.117434. **[VERIFIED]**
- Offline DMDc model used for LQR design.
- Reports that **increasing model order mitigates noise** in the DMDc model.
- *Relevance:* a structural DMDc-to-controller precedent, but not online.

1.16 **Ortolano F. et al.**, "An adaptive framework applied to structural health monitoring and damage-tolerant active control of smart structures", *J. Braz. Soc. Mech. Sci. Eng.* (2022), DOI 10.1007/s40430-022-03861-9. **[VERIFIED]**
- An online ERA model is compared with the healthy model through a subspace distance.
- When the distance is exceeded, the observer-based controller is reconfigured.
- *Relevance:* the "detect drift, then re-identify" logic in a smart structure.

1.17 **Kamaldar M. et al.**, "A numerical and experimental investigation of predictive cost adaptive control for noise and vibration suppression", *Mech. Syst. Signal Process.* (2024), DOI 10.1016/j.ymssp.2024.111711. **[VERIFIED]**
- RLS with **variable-rate forgetting based on an F-test** for closed-loop identification.
- Receding-horizon Riccati control.
- Needs no prior model.

1.18 **Koike T. et al.**, "Streaming Operator Inference for model reduction of large-scale dynamical systems", arXiv 2601.12161 (2026), DOI 10.48550/arxiv.2601.12161. **[VERIFIED]**
- Incremental SVD plus recursive least squares, with more than 99% memory reduction.

**Synthesis for Q1.**
- **Algorithms exist and are mature:**
  - Rank-1 online DMD (1.1).
  - Incremental-SVD online DMDc (1.2).
  - Recursive DMDc with RLS and forgetting (1.3).
  - PAST-based recursive DMD for structures (1.9).
- **Use for updating a control model is rare.** The verified cases are:
  - LQR in flow control (1.4).
  - MPC on toy nonlinear systems (1.12).
  - Event-triggered state feedback with ERA, not DMD (1.16).
- **No item was found in which online/recursive DMD(c) updates the secondary-path model of an FxLMS controller.** SciSpace and Consensus checks found nothing; see §2 and §8.

---

## 2. Online secondary-path modelling (SPM) in FxLMS and time-varying secondary paths in smart structures

### 2a. Classical online SPM

2.1 **Eriksson L.J., Allie M.C.**, "Use of random noise for on-line transducer modeling in an adaptive active attenuation system", *J. Acoust. Soc. Am.*, DOI 10.1121/1.397552. 350 citations. **[PARTIAL: year]** Consensus shows 1986; Akhtar 2005 (item 2.4) cites it as 1989.
- The original auxiliary-random-noise (Galois sequence) online SPM.

2.2 **Kuo S.M. et al.**, "A secondary path modeling technique for active noise control systems", *IEEE Trans. Speech Audio Process.* (1997), DOI 10.1109/89.593319. 196 citations. **[PARTIAL: co-author]** The tool shows "Kuo S. et al."; Vijayan is named only in the task prompt.
- Optimum delay of the adaptive prediction-error filter equals the length of the secondary-path impulse response.
- Works online and offline, with faster convergence and higher accuracy.

2.3 **Zhang M., Lan H., Ser W.**, "Cross-updated active noise control system with online secondary path modeling", *IEEE Trans. Speech Audio Process.* (2001), DOI 10.1109/89.928924. 223 citations. **[VERIFIED]** The tool shows "Ming Zhang et al."; Lan and Ser are named in the task prompt.
- Three cross-updated LMS filters decouple the controller from the modelling process.

2.4 **Zhang M., Lan H., Ser W.**, "On comparison of online secondary path modeling methods with auxiliary noise", *IEEE Trans. Speech Audio Process.* (2005), DOI 10.1109/tsa.2005.848887. 46 citations. **[VERIFIED]**
- Identifies four performance factors: auxiliary noise power, detection of path change, optimal step sizes, and independence of control and modelling.

2.5 **Akhtar M.T., Abe M., Kawamata M.**, "A new variable step size LMS algorithm-based method for improved online secondary path modeling in active noise control systems", *IEEE Trans. Audio Speech Lang. Process.* (2006), DOI 10.1109/tsa.2005.855829. 234 citations. **[VERIFIED]** The tool shows "M. Akhtar et al."; Abe and Kawamata appear as co-authors on the SciSpace record of their 2004 sister paper.
- Two adaptive filters instead of three. Uses modified FxLMS (MFxLMS).
- The SPM step size is scheduled on the estimated disturbance power.
- Earlier versions:
  - ISCAS 2005, DOI 10.1109/iscas.2005.1464575, which modifies Eriksson & Allie. **[VERIFIED]**
  - Multichannel extension, *Signal Process.* 2008, DOI 10.1016/j.sigpro.2008.02.004. **[VERIFIED]**

2.6 **Carini A. et al.**, "Optimal variable step-size NLMS algorithms with auxiliary noise power scheduling for feedforward active noise control", *IEEE Trans. Audio Speech Lang. Process.* (2008), DOI 10.1109/tasl.2008.2002757. 91 citations. **[VERIFIED]**
- Optimal variable-step-size (VSS) NLMS for both the SPM and control filters.
- Auxiliary noise scheduled to keep a fixed ratio of auxiliary to residual power.

2.7 **Lopes P. et al.**, "Auxiliary noise power scheduling algorithm for active noise control with online secondary path modeling and sudden changes", *IEEE Signal Process. Lett.* (2015), DOI 10.1109/lsp.2015.2415875. 44 citations. **[VERIFIED]**
- Handles sudden, strong path changes.

2.8 **Chan S., Chu Y.**, "Performance analysis and design of FxLMS algorithm in broadband ANC system with online secondary-path modeling", *IEEE Trans. Audio Speech Lang. Process.* (2012), DOI 10.1109/tasl.2011.2169789. 62 citations. **[PARTIAL: co-author]** Only "S. Chan et al." shown.
- Coupled mean and mean-square difference equations, closed-form steady-state excess MSE, and a design procedure.
- *Relevance:* an analytic prediction of convergence with online SPM, which could extend the base paper's deterministic map.

2.9 **Xiao Y. et al.**, "Properties of FXLMS-based narrowband active noise control with online secondary-path modeling", *IEEE Trans. Signal Process.* (2009), DOI 10.1109/tsp.2009.2020766. **[VERIFIED]**
- Closed-form steady-state MSE for the tonal case with online SPM.
- *Relevance:* matches the base paper's tonal tests at 120, 280 and 85 Hz.

**Methods without auxiliary noise:**
- Hu M. et al., *J. Acoust. Soc. Am.* (2019), DOI 10.1121/1.5129380 (multichannel). **[VERIFIED]**
- Wang C. et al., *Signal Process.* (2024), DOI 10.1016/j.sigpro.2024.109737 (DFT coefficients of S). **[VERIFIED]**
- Ji J. et al., Inter-Noise 2023, DOI 10.3397/in_2023_0737 / arXiv 2306.11408: switches between control and SPM when divergence is detected. **[VERIFIED]**

**Data-driven secondary-path update:**
- Oh J.Y. et al., *MSSP* (2024), DOI 10.1016/j.ymssp.2023.110940. A DNN updates the secondary-path estimate from operating conditions; PCA gives 98.5% storage reduction; 10.2 dB and 17.2 dB extra reduction. **[VERIFIED]**
- Shi C. et al., IEEE TASLP (2025), DOI 10.1109/taslpro.2025.3544089. Digital-twin online SPM with data reuse. **[VERIFIED]**
- Yang Z.-Y. et al., ICASSP 2026, DOI 10.1109/icassp55912.2026.11463219. MAML co-initialisation of the control filter and the secondary-path model. **[VERIFIED]**

### 2b. Online SPM applied to structural AVC (plates and beams)

2.10 **Pu Y.-X. et al.**, "Multi-channel adaptive active vibration control of piezoelectric smart plate with online secondary path modelling using PZT patches", *Mech. Syst. Signal Process.* (2019), DOI 10.1016/j.ymssp.2018.10.019. 83 citations. **[VERIFIED]**
- Multichannel FxLMS with auxiliary-noise online SPM.
- Each SPM filter and each control filter has its own VSS, defined through an "indirect error".
- Auxiliary-noise power scheduling (ANPS) per secondary path.
- Experiment: cantilever smart plate, NI CompactRIO.
- *Relevance:* **the direct experimental counterpart** to "online SPM in the base-paper setting". It uses PZT rather than MFC, and FIR-LMS SPM rather than DMDc.
- Related papers by the same group:
  - *Smart Mater. Struct.* 2014, DOI 10.1088/0964-1726/23/6/065015 (beam). **[VERIFIED]**
  - *Trans. Inst. Meas. Control* 2018, DOI 10.1177/0142331218802618 (epoxy plate). **[VERIFIED]**
  - *J. Low Freq. Noise Vib. Act. Control* 2019, DOI 10.1177/1461348419831119. **[VERIFIED]**

2.11 **Niu W. et al.**, "Adaptive vibration suppression of time-varying structures with enhanced FxLMS algorithm", *Mech. Syst. Signal Process.* (2019), DOI 10.1016/j.ymssp.2018.08.009. 65 citations. **[VERIFIED]**
- Online SPM from existing signals, with **no injected noise**.
- VSS adjusted by a bang-bang rule.
- Applied to buffeting of a vertical tail, numerically and experimentally.
- Classical FxLMS with an offline path **fails** on the time-varying structure; the enhanced version works.

2.12 **Zheng H.-B. et al.**, "Active vibration control for the time-varying systems with a new adaptive algorithm", *J. Vib. Control* (2019), DOI 10.1177/1077546319878297. **[VERIFIED]**
- FxLMS plus a gradient-modified recursive prediction error (RPE) estimate of the control path.
- No auxiliary excitation, at a higher computational cost.
- Follow-up: *IEEE SPL* 2020, DOI 10.1109/lsp.2020.2986139 (auxiliary-filter SPM; handles abrupt path changes). **[VERIFIED]**

2.13 **Xie X. et al.**, "Active vibration control of a time-varying shafting system using an adaptive algorithm with online auxiliary filter estimation", *J. Sound Vib.* (2021), DOI 10.1016/j.jsv.2021.116430. **[VERIFIED]**
- Thrust-bearing stiffness varies with shaft speed. The method works where FxLMS fails.

2.14 **Wu M.-Y. et al.**, "Active vibration control of inlet piping system with a time-variant secondary path using the mirror modified filtered-x least mean square algorithm", *J. Vib. Control* (2023), DOI 10.1177/10775463231195405. **[VERIFIED]**
- Theory: Lopes P. et al., *Int. J. Adapt. Control Signal Process.* (2021), DOI 10.1002/acs.3246. MMFxLMS "cannot diverge due to secondary path modelling errors", but converges slowly for large errors. **[VERIFIED]**

2.15 **Yang Y.-L. et al.**, "Multiple channel SPSA algorithm with improved cost function in active vibration control", *Mech. Syst. Signal Process.* (2024), DOI 10.1016/j.ymssp.2024.111862. **[VERIFIED]**
- Simultaneous-perturbation algorithm (SPSA) for multichannel control with **no secondary-path identification**, under a time-varying secondary path.
- Applied to a seawater-pump piping system.

2.16 **Wang M. et al.**, "A dual-loop active vibration control technology with an RBF-RLS adaptive algorithm", *Mech. Syst. Signal Process.* (2023), DOI 10.1016/j.ymssp.2022.110079. **[VERIFIED]**
- "The accuracy of the online model affects the amplitude attenuation performance by 57.1%."
- A 25.02 dB resonance reduction.
- *Relevance:* a quantitative statement of how much model accuracy matters.

### 2c. Time-varying secondary paths in smart structures: sources and magnitudes

| Source | Quantity | Magnitude found | Item and tag |
|---|---|---|---|
| Temperature, polymer-composite beam (GFRP; carbon not found) | Clamped-free natural frequency | Mode 1: 18.35 → 17.2 → 15.6 Hz at 30/40/50 °C, i.e. **−6.3% per 10 °C (30→40 °C), −9.3% (40→50 °C)**. Mode 3: 313.32 → 310.62 → 307.6 Hz, **≈ −0.9% per 10 °C** | "Dynamic characterization of CNT-reinforced hybrid polymer composite beam under elevated temperature—an experimental study", *Polymer Composites*, DOI 10.1002/pc.24668, full-text table passage via Scholar Gateway. **[PARTIAL: authors/year not shown]** |
| Temperature, CFRP plates | Natural frequency | **No per-degree value found** in any abstract. FE and experiment on clamped CFRP plates exist (Galea & White 1993) but the abstract gives no number | Galea S. et al., *J. Sound Vib.* (1993), DOI 10.1006/jsvi.1993.1225 **[VERIFIED]**. Magnitude **UNVERIFIED** |
| Temperature, CF120 carbon/epoxy shells, 20–200 °C | Frequency; damping | Frequency variation up to **8.2%** (amplitude- and temperature-dependent). Damping up to **+300.5%** from 20 to 200 °C | Li H. et al., *MSSP* (2021), DOI 10.1016/j.ymssp.2021.107665 **[VERIFIED]** |
| Temperature, piezo-fibre composites (MFC-like), −15 to 60 °C | Free strain | Rises, then falls with temperature. Low-Tg epoxy or PU matrix: longitudinal free strain **−27.5% from −15 to 40 °C**. High-Tg epoxy: value at −15 °C only **10.5%** below that at 40 °C | *Acta Materiae Compositae Sinica*, DOI 10.13801/j.cnki.fhclxb.20201015.002, URL https://fhclxb.buaa.edu.cn/cn/article/doi/10.13801/j.cnki.fhclxb.20201015.002, via WebSearch summary. **[PARTIAL: authors/year not shown]** |
| Temperature, piezo-fibre composites, −15 to 80 °C | d33, d31 | Increase below room temperature (glassy epoxy), then **decrease above room temperature** | Lin X.-J. et al., *Ceram. Int.* (2017), DOI 10.1016/j.ceramint.2017.04.172 **[VERIFIED]**. No number in the abstract |
| Thermal cycling of MFC, −60 to 90 °C and −50 to 150 °C | Actuation amplitude at 60–90 Hz, 20 Vpp | Drifts and stabilises after about 20 cycles. Residual **increase up to 70%** | Hobeck J. et al., *Appl. Phys. Lett.* (2016), DOI 10.1063/1.4943947 **[VERIFIED]** |
| MFC actuator in AVC, temperature chamber | Control performance | With a temperature-scheduled gain, the RMS reduction ratio improves by **17.26% at −60 °C** | Niu W. et al., ICMSD 2022 (IET), DOI 10.1049/icp.2022.1606 **[VERIFIED]** |
| MFC P1 vs P2 (P2 is the base paper's sensor type) | Pyroelectric coefficient | P2 about **−343**, P1 about **−82 µC m⁻² K⁻¹** | Acosta K.L. et al., *Smart Mater. Struct.* (2018), DOI 10.1088/1361-665x/aadc70 **[VERIFIED]** |
| PZT, 25–75 °C | e31, ε33 dependence | AVC performance is **not maintained** unless the temperature dependence of e31 and ε33 is included in the control law | Gupta V. et al., *Smart Mater. Struct.* (2011), DOI 10.1088/0964-1726/20/10/105023 **[VERIFIED]**. Also *MSSP* (2012), DOI 10.1016/j.ymssp.2012.07.009 **[VERIFIED]** |
| PZT-5H on titanium, −70 to 70 °C | AVC performance | "Even moderate fluctuations" degrade AVC considerably | Sharma A. et al., *J. Intell. Mater. Syst. Struct.* (2016), DOI 10.1177/1045389x15615968 **[VERIFIED]** |
| Active engine mounts | Secondary path | Significant variations **only near the actuator resonance**, mainly from **component temperature** | Hausberg F. et al., *J. Sound Vib.* (2015), DOI 10.1016/j.jsv.2014.11.024 **[VERIFIED]** |
| Added mass (plate) | Robustness test | Small added masses used to emulate parameter variation. μ-controller more robust than H∞ | Li P. et al., *Thin-Walled Struct.* (2003), DOI 10.1016/s0263-8231(03)00049-1 **[VERIFIED]** (see 4.1) |
| Boundary or bolt loosening | Natural frequencies | Effect of loosening degree, position and number of bolts studied. No number in the abstract | Liu H. et al., *Compos. Struct.* (2022), DOI 10.1016/j.compstruct.2022.115634 **[VERIFIED]** |
| Damage (cracks) | Plant uncertainty | A ±10% uncertainty on the system matrix is used for robustness checks | Huang Z.-C. et al., *Smart Mater. Struct.* (2026), DOI 10.1088/1361-665x/ae748f **[VERIFIED]** |
| Variable loads (bolted composite plates) | AVC under varying load | Adaptive control designed; abstract not shown | Zhang Y. et al., *MSSP* (2025), DOI 10.1016/j.ymssp.2025.112496, 50 citations. **[PARTIAL: abstract not shown]**. Cited by the base paper as ref [29] |

*Analysis.* For a CFRP plate whose resin dominates the transverse and shear stiffness (G12 = 7.17 GPa), a frequency drift of order 1% per 10 °C near room temperature is plausible, but **UNVERIFIED** by this sweep. The GFRP beam data bracket it between −0.9% (mode 3) and −6% to −9% (mode 1) per 10 °C. A numerical project should therefore not adopt one value. It should sweep:
- the frequency shift Δf/f over 0–5%;
- the actuator gain change ΔG/G over ±30%, bracketed by the −27.5% free-strain change and the +70% residual cycling drift.

The FxLMS stability boundary of ±90° phase error (item 3.9) can then be mapped onto these physical drifts.

**Synthesis for Q2.**
- Online SPM for FxLMS is mature in acoustics (2.1–2.9). It has been demonstrated on piezo plates and beams (2.10–2.13).
- Every structural implementation found models the secondary path as an **FIR filter adapted by LMS/NLMS**. They use auxiliary noise (2.10) or noise-free decorrelation tricks (2.11, 2.12).
- None uses a state-space, DMD-type, physically interpretable model as the online secondary-path tracker.
- Temperature sensitivity of MFC actuation is documented, with a large effect (tens of percent). For CFRP plate frequencies, no per-degree number was obtained.

---

## 3. FxLMS variants for noise robustness and fast convergence in AVC, 2019–2026, plus robustness theory

### 3a. Reviews and comparisons

3.1 **Akbar H. et al.**, "Adaptive filtering algorithm for active vibration control in piezoelectric smart structures", *Rev. Sci. Instrum.* (2026), DOI 10.1063/5.0308388. **[VERIFIED]**
- Review of LMS, FxLMS and VSS-FxLMS on piezo beams, blades and plates.
- States that "comprehensive comparative studies across different applications remain limited".
- *Relevance:* supports a like-for-like benchmark on one identified twin (gap G4).

3.2 **Luo S. et al.**, "Evaluation of FxLMS and FxRLS algorithms in active vibration control using piezoelectric stack actuators for gear transmission", *Smart Mater. Struct.* (2025), DOI 10.1088/1361-665x/adbf7a. **[VERIFIED]**
- FxRLS is less sensitive to the eigenvalue spread of the reference correlation matrix.
- FxRLS is faster and suppresses vibration better than FxLMS. Confirmed by co-simulation and experiment.

### 3b. Variable-step-size and nonlinear step-size families (AVC)

3.3 **Jiang J. et al.**, "A variable step-size FxLMS algorithm for active vibration control of cantilever beam" (VSS-FxtanhLMS), *J. Low Freq. Noise Vib. Act. Control* (2026), DOI 10.1177/14613484261470208. **[VERIFIED]**
- Mean and mean-square analysis.
- In experiment, against the best competitor (FxasinhLMS): steady-state error **−17.48%**, convergence time **−33.33%**, PSD **−1.72 dB/Hz**.
- Related: Jiang J. et al., *J. Vib. Control* (2024), DOI 10.1177/10775463241260110 (sech-based VSS on a blade). **[VERIFIED]**

3.4 **Li W. et al.**, "Error signal differential term feedback enhanced variable step size FxLMS algorithm for piezoelectric active vibration control" (DVSFxLMS), *Shock Vib.* (2020), DOI 10.1155/2020/8832467. **[VERIFIED]**
- Related, same group: hybrid PID-FxLMS, *J. Sound Vib.* (2021), DOI 10.1016/j.jsv.2021.116243, 95 citations. **[VERIFIED]** The base paper cites it as ref [25].

3.5 **Gao Z.-Y. et al.**, "Theoretical and experimental investigation study of discrete time rate-dependent hysteresis modeling and adaptive vibration control for smart flexible beam with MFC actuators", *Sens. Actuators A* (2022), DOI 10.1016/j.sna.2022.113738. **[VERIFIED]**
- FX-VSSFT-LMS: **variable step and variable tap length**, auto-tuned by a hybrid Jaya metaheuristic.
- Hammerstein (Bouc-Wen + ARX) secondary path for the MFC.
- Related:
  - *J. Vib. Eng. Technol.* (2021), DOI 10.1007/s42417-021-00417-4 (DE-Jaya, piezo stack). **[VERIFIED]**
  - *Aerosp. Sci. Technol.* (2024), DOI 10.1016/j.ast.2024.109455 (piezo blade layout plus VSS-FxLMS). **[VERIFIED]**

3.6 **Chang D.-C. et al.**, "Feedforward active noise control with a new variable tap-length and step-size filtered-X LMS algorithm", *IEEE/ACM Trans. Audio Speech Lang. Process.* (2014), DOI 10.1109/taslp.2013.2297016. **[VERIFIED]**
- Variable L derived from an exponential-decay model of the controller.
- *Relevance:* makes L adaptive, which removes the second axis of the base paper's (μ, L) map.

3.7 **Song P., Zhao H.**, "Filtered-x least mean square/fourth (FXLMS/F) algorithm for active noise control", *Mech. Syst. Signal Process.* (2019), DOI 10.1016/j.ymssp.2018.10.009. 88 citations. **[PARTIAL: co-author]** Only "Pucha Song et al." shown.
- FXLMS/F acts as a VSS. A convex combination (C-FXLMS/F) is also proposed, with a stability condition.

3.8 **Jiang X. et al.**, "A game theory adaptive weighting convex combination FxLMS algorithm for active vibration control", *Meas. Control* (2026), DOI 10.1177/00202940261436521. **[VERIFIED]**
- Lyapunov-proven. Steady-state error **−23.9%** and control energy **−41.7%** against benchmarks.

**Other AVC VSS papers. These are incremental, and none is model-based:**
- Wu T. et al., *J. Vib. Eng. Technol.* (2023), DOI 10.1007/s42417-023-01044-x (error-weight FXLMS, plate). **[VERIFIED]**
- Ren S. et al., CCC 2024, DOI 10.23919/ccc63176.2024.10661723 (VSS-FxLMS with MFC on ZYNQ). **[VERIFIED]**
- Umar H.M. et al., *J. Low Freq. Noise* (2024), DOI 10.1177/14613484241295460 (MFC beam). **[VERIFIED]**
- Zhang F. et al., *Shock Vib.* (2022), DOI 10.1155/2022/9970106 (multichannel VSS, gears). **[VERIFIED]**
- Fang Y.-B. et al., *J. Vib. Control* (2021), DOI 10.1177/10775463211022885 (variable forgetting factor). **[VERIFIED]**
- Yang X.-S. et al., ASIM 2025, DOI 10.1109/asim67379.2025.11512924 (step size learned by DDPG). **[VERIFIED]**

### 3c. RLS, Kalman, momentum, proportionate, fractional and leaky variants

**RLS family:**
- Sun Y. et al., *Smart Mater. Struct.* (2020), DOI 10.1088/1361-665x/ab8c24 (Fx-RLS hybrid isolation strut, 41.84 dB). **[VERIFIED]**
- Hakvoort W. et al., *Mechatronics* (2023), DOI 10.1016/j.mechatronics.2022.102934 (filtered-error RLS, MIMO isolator, up to 40 dB). **[VERIFIED]**
- Beijen M. et al., IFAC-PapersOnLine (2019), DOI 10.1016/j.ifacol.2019.11.716. **[VERIFIED]**
- Liang X.-B. et al., *Sensors* (2022), DOI 10.3390/s22124566 (**proportionate FxRLS with momentum**; reduces sensitivity to the forgetting factor). **[VERIFIED]**
- Akhtar M.T., *Appl. Sci.* (2023), DOI 10.3390/app13042715 (robust FxRLS with a convex-combined step size). **[VERIFIED]**

**Momentum:**
- Song H.-S. et al., *J. Low Freq. Noise* (2025), DOI 10.1177/14613484251320198 (FxLMS spliced with Momentum-FxLMS). **[VERIFIED]**

**Fractional-order:**
- Aslam M.S. et al., *Signal Process.* (2015), DOI 10.1016/j.sigpro.2014.04.012 (fractional LMS for online SPM, 120 citations). **[VERIFIED]**
- Li T. et al., *Machines* (2022), DOI 10.3390/machines10080670 (multichannel fractional-order ANC). **[VERIFIED]**
- Wang L. et al., *Electronics* (2024), DOI 10.3390/electronics13101974 (fractional LMS for secondary-path identification plus MIMO-FxLMS). **[VERIFIED]**

**Leaky:**
- Tobias O.J. et al., *IEEE TSAP* (2005), DOI 10.1109/tsa.2005.852018 (stochastic analysis with secondary-path error). **[VERIFIED]**
- Tang Y. et al., *IEEE TCST* (2023), DOI 10.1109/tcst.2023.3242586 (frequency-weighted leaky FxLMS for guaranteed stability and limited output). **[VERIFIED]**
- Shi D.-Y. et al., *IEEE SPL* (2019), DOI 10.1109/lsp.2019.2903908 (optimal leak factor). **[VERIFIED]**

**Kalman-based:**
- Rashid S.M. et al., *J. Vib. Control* (2025), DOI 10.1177/10775463251356477 (NN-enhanced Kalman filter; −40.12% amplitude, simulation only). **[VERIFIED]**
- Feedback/Youla RLS: Li F. et al., *Appl. Sci.* (2021), DOI 10.3390/app11083338. More than 20 dB, with response time below 0.3 s to a sudden change. **[VERIFIED]**

### 3d. Robustness theory that turns "robustness" into numbers

3.9 **Snyder S.D., Hansen C.H.**, "The effect of transfer function estimation errors on the filtered-x LMS algorithm", *IEEE Trans. Signal Process.* (1994), DOI 10.1109/78.285659. 185 citations. **[PARTIAL: co-author]** Only "S. Snyder et al." shown.
- **±90° phase error** of the secondary-path estimate is the stability bound.
- Inside that region there is no simple relation between error and stability.

3.10 **Boucher C.C., Elliott S.J., Nelson P.A.**, "Effect of errors in the plant model on the performance of algorithms for adaptive feedforward control", *IEE Proc. F* (1991), DOI 10.1049/ip-f-2.1991.0042. 101 citations. **[PARTIAL: co-authors]** Only "C. C. Boucher et al." shown.
- Phase error is mapped to **maximum update coefficient and convergence time** for single-channel FxLMS.
- Gives a statistical analysis for the multichannel case.

3.11 **Elliott S.J.**, "Filtered reference and filtered error LMS algorithms for adaptive feedforward control", *Mech. Syst. Signal Process.* (1998), DOI 10.1006/mssp.1998.0178. **[VERIFIED]**
- Stability is expressed as a **bound on the unstructured multiplicative uncertainty** of the plant.
- *Relevance:* the bridge from FxLMS to the robust-control language the user already knows.

3.12 **Lopes P., Piedade M.**, "The behavior of the modified FX-LMS algorithm with secondary path modeling errors", *IEEE Signal Process. Lett.* (2004), DOI 10.1109/lsp.2003.821745. **[PARTIAL: co-author]**
- MFxLMS is stable for arbitrary delay if Re{Ŝ/S} > 1/2, i.e. a phase error below 60° when there is no amplitude error.

3.13 **Yang F.-R. et al.**, "Stochastic analysis of the filtered-x LMS algorithm for active noise control", *IEEE/ACM Trans. Audio Speech Lang. Process.* (2020), DOI 10.1109/taslp.2020.3012056. 86 citations. **[VERIFIED]**
- Mean and mean-square stability bound for an arbitrary reference and a general secondary path.

3.14 **Ardekani I.T., Abdulla W.H.**:
- "Theoretical convergence analysis of FxLMS algorithm", *Signal Process.* (2010), DOI 10.1016/j.sigpro.2010.05.009, 143 citations. **[VERIFIED]**
- "Effects of imperfect secondary path modeling on adaptive active noise control systems", *IEEE TCST* (2012), DOI 10.1109/tcst.2011.2161762, 75 citations. **[VERIFIED]**
- Finding: **intentional misadjustment of the secondary-path model can improve performance**.

3.15 **Gong C. et al.**, "Statistical analysis of multichannel FxLMS algorithm for narrowband active noise control", *Signal Process.* (2022), DOI 10.1016/j.sigpro.2022.108646. **[VERIFIED]**
- Closed-form mean and mean-square stability boundaries, including cross-path coupling.
- *Relevance:* an analytic counterpart to the base paper's 2-channel 85 Hz map.

3.16 **Fraanje R. et al.**, "Robustness of the filtered-X LMS algorithm—Part I", *IEEE Trans. Signal Process.* (2007), DOI 10.1109/tsp.2007.896083. **[PARTIAL: first author]** Consensus shows "M. Verhaegen et al.".
- Frequency-domain necessary and sufficient condition through the pseudospectrum of Toeplitz matrices.

3.17 **Costa M.H. et al.**, "Stochastic analysis of the filtered-X LMS algorithm in systems with nonlinear secondary paths", *IEEE Trans. Signal Process.* (2002), DOI 10.1109/tsp.2002.1003058. **[VERIFIED]**
- A small saturation nonlinearity has a significant effect.
- *Relevance:* MFC hysteresis and amplifier saturation are not modelled in the base paper.

**Synthesis for Q3.**
- About 20 VSS, nonlinear-step, convex-combination and RLS variants have been published for AVC since 2019. Typical claimed gains are 17–41% in steady-state error or convergence time.
- Almost all are tuned by hand or by a metaheuristic on a physical or FE model.
- None found evaluates the **size of the effective-convergence region** (the base paper's metric), or its robustness to a drifted secondary path.
- The robustness theory (3.9–3.16) is mature and quantitative. *Analysis:* it has not been combined with a data-driven identified path plus an uncertainty estimate.

---

## 4. Combining a data-driven model with robust or optimal control for plates

### 4a. μ-synthesis and H∞ with identified uncertainty

4.1 **Li P. et al.**, "Robust control of a vibrating plate using μ-synthesis approach", *Thin-Walled Struct.* (2003), DOI 10.1016/s0263-8231(03)00049-1. **[VERIFIED]**
- MIMO μ-controller on dSPACE DS1103. The disturbance-to-structure model is not needed.
- Small **added masses** emulate parameter variation. μ is **more robust than H∞** with the same weightings.
- *Relevance:* same journal as the base paper and same DS1103 hardware class; it ties directly to the user's μ skills.

4.2 **Li D.-W. et al.**, "Active vibration control design and experiment of the flexible plate with mixed uncertainty via μ-synthesis", WCICA 2008, DOI 10.1109/wcica.2008.4594053. **[VERIFIED]**
- Model identified by multivariable frequency-domain identification and ERA.
- Modal-parameter perturbations extracted by eigen-decomposition and placed in a linear fractional transformation (LFT).
- Result: attenuation time **40 s shorter** and random-noise reduction **about +4 dB**.
- *Relevance:* **identified model → uncertainty block → μ**, i.e. the pipeline a DMDc ensemble could feed.
- Related: Zhang L. (2010), J. Exp. Mech., actuator-force uncertainty, +3 dB. **[PARTIAL: DOI not shown]**

4.3 **Zhang S.-B. et al.**, "Robust active vibration control of flexible smart beam by μ-synthesis", *J. Sound Vib.* (2024), DOI 10.1016/j.jsv.2024.118737. **[VERIFIED]**
- The uncertain model is extracted from identification in magnitude and phase. Experiment shows robust stability and robust performance.

4.4 **Oomen T. et al.**, "Iterative data-driven H∞ norm estimation of multivariable systems with application to robust active vibration isolation", *IEEE TCST* (2014), DOI 10.1109/tcst.2014.2303047. **[VERIFIED]**
- Model-error modelling with fewer prior assumptions, which gives stronger guarantees for robust design.

4.5 **Gupta V. et al.**, "Non-parametric IQC multipliers in data-driven robust controller synthesis", *Automatica* (2026), DOI 10.1016/j.automatica.2025.112608. **[VERIFIED]**
- Convex robust synthesis directly from the frequency response, using an elliptical uncertainty set. Micro-vibration platform.

**Data-driven H∞ without identification:**
- Kucukdemiral I. et al., *ISA Trans.* (2023), DOI 10.1016/j.isatra.2023.01.030. **[VERIFIED]**
- Gormus B. et al., CoDIT 2025, DOI 10.1109/codit66093.2025.11321596 (12.7% lower H∞ norm with non-common Lyapunov matrices). **[VERIFIED]**
- Qin Z. et al., *CNSNS* (2023), DOI 10.1016/j.cnsns.2023.107397. **[VERIFIED]**

**Hybrid robust feedback plus adaptive feedforward:**
- Qin Y.-F. et al., *Shock Vib.* (2021), DOI 10.1155/2021/3153531. **[VERIFIED]** An H∞ mixed-sensitivity loop "adds active damping on the secondary path to weaken the adverse effects of its sudden changes"; the LMS feedforward then works on the equivalent path. Measured: 85% attenuation within 2.5 s.
- Song Y. et al., *IEEE TSAP* (2005), DOI 10.1109/tsa.2005.848878. **[VERIFIED]** An analog feedback loop reduces plant fluctuation for FxLMS.

### 4b. ADRC/ESO for plates

4.6 **Li S.-Q. et al.**, "Active disturbance rejection vibration control for an all-clamped piezoelectric plate with delay", *Control Eng. Pract.* (2021), DOI 10.1016/j.conengprac.2020.104719. 57 citations. **[VERIFIED]**
- Plate model identified with an auxiliary-variable method.
- Linear ADRC with a Smith predictor and an ESO for mode errors and harmonics. Experiment on NI PCIe-6343.

**Related ADRC/ESO papers:**
- Gu J.-H. et al., *Phys. Scr.* (2025), DOI 10.1088/1402-4896/ae221e (model-assisted ESO plus dynamic sliding mode; **15.1 dB → 0.2 dB**). **[VERIFIED]**
- Sun X. et al., *Eur. J. Mech. A/Solids* (2024), DOI 10.1016/j.euromechsol.2024.105420 (MIMO decoupling LADRC, multimodal beam). **[VERIFIED]**
- Ramírez-Neria M. et al., *J. Vib. Control* (2021), DOI 10.1177/10775463211009377 (building ADRC). **[VERIFIED]**

### 4c. Koopman/DMDc and MPC

4.7 **Korda M., Mezić I.**, "Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control", *Automatica*, DOI 10.1016/j.automatica.2018.03.046. 1371 citations. **[PARTIAL: year]** Consensus shows 2016, the preprint year, and "Milan Korda et al."; Mezić is named in the task prompt.
- Related: Arbabi H. et al., CDC 2018, DOI 10.1109/cdc.2018.8619720 (Koopman MPC for PDEs; sub-millisecond online evaluation). **[VERIFIED]**

4.8 **Jiang H.-K. et al.**, "Model predictive control of fluid–structure interaction via Koopman-based reduced-order model", *J. Fluid Mech.* (2026), DOI 10.1017/jfm.2025.11035. **[VERIFIED]**
- Kernel DMD plus residual DMD give a switched Koopman reduced-order model, used in MPC to suppress flapping of a flexible plate.
- "Matches the control performance of reinforcement learning while markedly reducing computational cost".

4.9 **Song D.-Y. et al.**, "Electromechanical coupling modeling and LQG active vibration control of CFRP cantilever plates using MFCs", *Modelling* (2026), DOI 10.3390/modelling7050177. **[VERIFIED]**
- FSDT with assumed modes. **Virtual springs** model the non-ideal clamp.
- First four frequencies within 2%. LQG validated experimentally.
- *Relevance:* same structure class (CFRP plate with MFC); a model-based alternative to the virtual testbed.

**Other MPC on beams and plates:**
- Alsahlani A. et al., *Open Eng.* (2024), DOI 10.1515/eng-2024-0001 (adaptive MPC with online identification, composite beam; settling 1.8–4 s depending on actuator position). **[VERIFIED]**
- Qiu Z.-C. et al., *J. Intell. Mater. Syst. Struct.* (2021), DOI 10.1177/1045389x20948597 (sliding-mode predictive control, piezo plate, **laser displacement sensors**). **[VERIFIED]**
- Dakova S. et al., *Eng. Struct.* (2025), DOI 10.1016/j.engstruct.2025.121317 (MPC on an adaptive building; data-based reduced-order model compared; +62% damping). **[VERIFIED]**

### 4d. Reinforcement learning on a data-driven or identified twin

4.10 **Qiu Z.-C. et al.**, "Reinforcement learning vibration control for a flexible hinged plate", *Aerosp. Sci. Technol.* (2021), DOI 10.1016/j.ast.2021.107056. **[VERIFIED]**
- FEM plus experimental identification gives the RL environment. DDPG with prioritised experience replay.
- Controllers transferred to the experiment by parameter transfer. Better than PD for small amplitudes.
- Series by the same group:
  - *MSSP* 2022, DOI 10.1016/j.ymssp.2022.109362. **[VERIFIED]**
  - *MSSP* 2023, DOI 10.1016/j.ymssp.2023.110502. **[VERIFIED]**
  - *MSSP* 2025, DOI 10.1016/j.ymssp.2024.112005. **[VERIFIED]**
  - *Aerosp. Sci. Technol.* 2022, DOI 10.1016/j.ast.2022.107801. **[VERIFIED]**
  - *Eng. Appl. Artif. Intell.* 2022, DOI 10.1016/j.engappai.2022.105002. **[VERIFIED]**
  - *Eng. Appl. Artif. Intell.* 2024, DOI 10.1016/j.engappai.2024.108630. **[VERIFIED]**

4.11 **Febvre M. et al.**, "Deep reinforcement learning for tuning active vibration control on a smart piezoelectric beam", *J. Intell. Mater. Syst. Struct.* (2024), DOI 10.1177/1045389x241260976. **[VERIFIED]**
- DRL **tunes the parameters of a fixed controller** on an experimental model, compared with H2/H∞ norm minimisation.
- *Relevance:* closest analogue to "tune (μ, L) on the identified twin".

4.12 **Sakamoto T., Okabayashi K.**, "Optimization of fluid control laws through deep reinforcement learning using dynamic mode decomposition as the environment", *AIP Adv.* (2024), DOI 10.1063/5.0237682. **[VERIFIED]**
- **DMDc model used as the RL environment.** Up to 3× saving over CFD.
- Fluid problem only. No structural equivalent was found.

**Other RL vibration control:**
- Nasiri K. et al., *MSSP* (2025), DOI 10.1016/j.ymssp.2024.112198 (SAC versus GA-optimised type-2 fuzzy for chatter in milling a plate with piezo patches; links to the user's previous Du et al. topic). **[VERIFIED]**
- Hu W.-S. et al., *J. Vib. Control* (2026), DOI 10.1177/10775463261483243 (constrained RL with vertex rollouts over a parametric-uncertainty set). **[VERIFIED]**

**Synthesis for Q4.**
- The robust-control route is well trodden on plates: identified model → μ/H∞ (4.1–4.3). So is ADRC (4.6).
- The uncertainty set has always come from **modal-parameter perturbation or frequency-response envelopes**. In no item found did it come from a statistical ensemble of data-driven (DMD) models.
- DMDc-based MPC and RL exist in fluids (4.8, 4.12). For structures, RL environments are FEM plus identification (4.10).
- **No structural AVC item uses a DMDc model as the MPC predictor or RL environment** (SciSpace check).

---

## 5. Uncertainty quantification (UQ) of DMD models

5.1 **Sashidhar D., Kutz J.N.**, "Bagging, optimized dynamic mode decomposition for robust, stable forecasting with spatial and temporal uncertainty quantification" (BOP-DMD), *Phil. Trans. R. Soc. A* (2022), DOI 10.1098/rsta.2021.0199. 97 citations. **[PARTIAL: co-author]** Only "Diya Sashidhar et al." shown; Kutz from memory.
- An ensemble of optimized-DMD models is fitted to bootstrap subsets of a single snapshot set.
- Gives **spatial and temporal UQ** and stabilised forecasts.
- Implemented in PyDMD (5.6).

5.2 **Palma G. et al.**, "Data-driven uncertainty-aware seakeeping prediction of the Delft 372 catamaran using ensemble Hankel dynamic mode decomposition", *J. Hydrodyn.* (2025), DOI 10.1007/s42241-026-0015-z. **[VERIFIED]**
- Compares a **Bayesian Hankel-DMDc** (stochastic hyperparameters) with a **frequentist Hankel-DMDc** (ensembles over data subsets).
- On experimental data, the frequentist version improved accuracy and gave robust UQ. The Bayesian version was "not found beneficial".
- *Relevance:* evidence on **which UQ flavour to use for DMDc**.

5.3 **Palma G. et al.**, "Model-free system identification of surface ships in waves via Hankel dynamic mode decomposition with control", arXiv 2502.15782 (2025), DOI 10.48550/arxiv.2502.15782. **[VERIFIED]**
- Bayesian Hankel-DMDc adds UQ and improves accuracy.
- Related: *Appl. Ocean Res.* (2025), DOI 10.1016/j.apor.2025.104863. **[VERIFIED]** Here "the uncertainty correlates well with the prediction error".

5.4 **Takeishi N. et al.**, "Bayesian dynamic mode decomposition", IJCAI 2017, DOI 10.24963/ijcai.2017/392. **[VERIFIED]**
- Variational version: Kawashima T. et al., AAAI 2021, DOI 10.1609/aaai.v35i9.16985. **[VERIFIED]**

5.5 **Algikar P. et al.**, "Statistical analysis and method to quantify the impact of measurement uncertainty on dynamic mode decomposition", CDC 2024, DOI 10.1109/cdc56724.2024.10886110. **[VERIFIED]**
- Random-matrix propagation of the first two moments through DMD gives **analytic confidence bounds**, checked against Monte Carlo.
- *Relevance:* a cheap alternative to bootstrapping.

5.6 **Ichinaga S.M. et al.**, "PyDMD: a Python package for robust dynamic mode decomposition", arXiv 2402.07463 (2024), URL https://github.com/PyDMD/PyDMD. **[VERIFIED]**
- Version 1.0 includes BOP-DMD and DMDc.
- *Relevance:* free tooling, consistent with the no-budget constraint.

**Other UQ and robustness items:**
- Askham T. et al., *SIAM J. Appl. Dyn. Syst.* (2018), DOI 10.1137/m1124176 (optimized DMD, less noise bias). **[PARTIAL: year]** Consensus shows 2017.
- Rains J. et al., *J. Comput. Phys.* (2023), DOI 10.1016/j.jcp.2023.112604 (constrained optimized DMDc that enforces stable eigenvalues; maximum relative error below 0.5%). **[VERIFIED]**
- Duke D. et al., *Exp. Fluids* (2012), DOI 10.1007/s00348-011-1235-7 (error bars on DMD growth rates). **[VERIFIED]**
- Gu M. et al., *Physica D* (2023), DOI 10.1016/j.physd.2023.133938 (DMD as the maximum-likelihood estimator of a linear state-space model, giving a generative model for UQ). **[VERIFIED]**
- Fasel U. et al., *Proc. R. Soc. A* (2022), DOI 10.1098/rspa.2021.0904 (ensemble-SINDy; ensemble statistics used for MPC). **[VERIFIED]**

**Synthesis for Q5.** Three UQ routes are available and free in Python:
- Bootstrap/bagging (5.1).
- Frequentist data-subset ensembles (5.2).
- Analytic moment propagation (5.5).

The project needs:
1. eigenvalue/frequency confidence intervals of the identified paths;
2. propagation through FxLMS to **confidence bands on convergence time and on the effective-convergence region**.

Item 1 is routine. **Item 2 was not found anywhere.**

---

## 6. Optimisation of (multichannel) FxLMS hyper-parameters

6.1 **Stenger D. et al.**, "Benchmark of Bayesian optimization and metaheuristics for control engineering tuning problems with crash constraints", arXiv 2211.02571 (2022), DOI 10.48550/arxiv.2211.02571. **[VERIFIED]**
- Compares 11 Bayesian-optimisation (BO) variants with 7 metaheuristics on 10 controller-tuning problems.
- Control tuning problems feature deterministic noise, low multimodality and large infeasible ("crash") regions.
- With a budget of 25·d evaluations: **pattern search was best for d = 2**, and **BADS was the most sample-efficient for 3 ≤ d ≤ 5**.
- Gain over random search: **6.6% on average, up to 16.1%**.
- *Relevance:* the base paper's (μ, L) is d = 2 and its 4-D problem is d = 4. FxLMS divergence is exactly a crash constraint.

6.2 **Stenger D. et al.**, "Early stopping Bayesian optimization for controller tuning", CDC 2024, DOI 10.1109/cdc56724.2024.10886051. **[VERIFIED]**
- Stops episodes that are clearly suboptimal.
- *Relevance:* abort diverging or slowly converging FxLMS runs early.

6.3 **Nobar M. et al.**, "Guided Bayesian optimization: data-efficient controller tuning with digital twin", *IEEE Trans. Autom. Sci. Eng.* (2024), DOI 10.1109/tase.2024.3454176. **[VERIFIED]**
- The digital twin is consulted when the Gaussian-process uncertainty is high.
- **57% and 46% fewer** hardware experiments than plain BO.
- *Relevance:* matches "DMDc twin plus a few real tests".

6.4 **Fujimoto Y. et al.**, "Controller tuning with Bayesian optimization and its acceleration: concept and experimental validation", *Asian J. Control* (2022), DOI 10.1002/asjc.2847. **[VERIFIED]**
- The **BO prior mean is built from a plant model**, which needs far fewer iterations.
- *Relevance:* use the DMDc-predicted convergence map as the prior.

6.5 **Neumann-Brosig M. et al.**, "Data-efficient autotuning with Bayesian optimization: an industrial control study", *IEEE TCST* (2019), DOI 10.1109/tcst.2018.2886159. 121 citations. **[VERIFIED]**

6.6 **Yao J. et al.**, "A novel adaptive step-size hybrid active noise control system", *Appl. Acoust.* (2021), DOI 10.1016/j.apacoust.2021.108285. **[VERIFIED]**
- Improved PSO whose **objective combines convergence rate and noise reduction**.

**Other PSO/GA tuning of FxLMS-type parameters:**
- Meng H. et al., *Appl. Acoust.* (2020), DOI 10.1016/j.apacoust.2020.107796 (PSO step-size factor). **[VERIFIED]**
- Jiang T. et al., *Appl. Sci.* (2023), DOI 10.3390/app13010225 (PSO; vehicle hardware-in-the-loop; 12 dB below 100 Hz). **[VERIFIED]**
- Long G. et al., *J. Vib. Control* (2021), DOI 10.1177/10775463211001625 (**GA for the multi-parameter design of multichannel delayless subband ANC before deployment**). **[VERIFIED]**
- Zhao T. et al., *IEEE TIE* (2017), DOI 10.1109/tie.2017.2682043 (GA convergence coefficient plus offline/online SPM; transformer, 8–15 dB). **[VERIFIED]**
- Gao Z.-Y. et al. 2021/2022: Jaya and DE tuning of the VSS and variable-tap parameters (3.5).
- Julai S. et al., *J. Low Freq. Noise* (2010), DOI 10.1260/0263-0923.29.4.293 (GA/PSO plate AVC). **[VERIFIED]**

**Data-driven step-size selection:**
- Li L. et al., *IEEE SPL* (2026), DOI 10.1109/lsp.2026.3671181 (modified MAML selects the FxLMS step size without extra online computation). **[VERIFIED]**

**Gradient-free control without a secondary-path model:**
- Rout N.K. et al., *IEEE TIM* (2012), DOI 10.1109/tim.2011.2169180 (PSO). **[VERIFIED]**
- Zhou Y. et al., *IEEE TIM* (2023), DOI 10.1109/tim.2023.3309366 (GA). **[VERIFIED]**

**Prior art to the base paper's "simulate, then pick (μ, L)" idea:**

6.7 **Li F., Yang T., Li W., Brennan M.J., Wu L., Li X.**, "Predicting the performance and optimizing the control parameters of an active vibration control system by pre-identifying the secondary path", *J. Sound Vib.* 568 (2024) 117953, DOI 10.1016/j.jsv.2023.117953. Author list and volume are from the base paper's ref [28]. **[VERIFIED]**
- The secondary path is split into actuator and electronics FRF, structure (identified by impact), and ADC/DAC (a half-sample phase shift).
- It is identified **before installing** the AVC hardware.
- FxLMS performance is predicted and its parameters optimised in simulation.
- *Relevance:* the closest prior art to the base paper's thesis. A new project cannot claim "simulation predicts (μ, L)" as new.

**Synthesis for Q6.**
- PSO/GA/Jaya/DE tuning of FxLMS-family parameters is common, with about 10 items.
- Model-based BO with priors (6.4), crash constraints (6.1) and digital twins (6.3) is established in control engineering. *Analysis:* it has not been applied to FxLMS step and filter-length tuning on a data-driven twin, nor to a **robust objective over an ensemble of identified models**.

---

## 7. Cross-cutting synthesis

1. **What the base paper claims and what the literature already contains.**
   - "Data-driven model → simulate FxLMS → choose (μ, L)" is anticipated in a model-based form by Li F. et al. 2024 (6.7) and Liu T. et al. 2024 (*MSSP*, DOI 10.1016/j.ymssp.2023.110876, MFC composite plate with FxLMS, 67 citations). **[VERIFIED]**
   - The base paper's genuine additions are the **purely data-driven** sd-DMDc identification and the **experimentally validated convergence-region maps**.
2. **Robustness is the open axis.** Every element needed for an uncertainty-aware version exists separately but has not been assembled:
   - DMD UQ (§5).
   - FxLMS phase-error and multiplicative-uncertainty bounds (3.9–3.15).
   - μ-synthesis from identified uncertainty (4.1–4.3).
   - Robust and crash-constrained BO (6.1–6.4).
3. **Time variation.**
   - Online DMD(c) algorithms are ready (1.1–1.3, 1.9), and online SPM for plates is demonstrated with FIR-LMS (2.10).
   - The coupling "online DMDc → secondary-path model inside FxLMS" was **not found**.
   - The physical drift magnitudes for CFRP+MFC are uncertain:
     - MFC actuation: tens of percent (2c).
     - CFRP frequency per °C: not verified.
     - So any study must be parametric.
4. **Advanced algorithms.** VSS/RLS/convex-combination papers report 17–41% gains, but on different structures and metrics. The base paper's effective-region metric (t < 5 s), applied to one identified twin, offers a **neutral benchmark** that the review in 3.1 says is missing.
5. **Numerical-only feasibility.** Every component listed in §8 below runs in Python:
   - PyDMD;
   - numpy/scipy for FxLMS;
   - python-control/slycot for H∞;
   - scikit-optimize or BADS (pybads) for BO;
   - pyswarms for PSO.

   The user's Chebyshev-Ritz CFRP+MFC plate is the "truth" that replaces the experiment. *Analysis:* the virtual-testbed approach is credible only if the Ritz plate is first calibrated to the base paper's published frequencies and convergence curves. That is the Phase 1 already completed in this project.

---

## 8. GAPS, ranked by suitability for a NUMERICAL-ONLY project

How to read the ranking:
- **Suitability (S)** combines laptop feasibility, independence from new experiments, fit with the user's skills (Python, Ritz plate, μ/D-K, PSO, ADRC), and the chance of a defensible novelty claim.
- **Confidence** is my subjective estimate that the gap is real as of 2026-10-16, i.e. that no published item fills it. It is not a statistic.
- **G0** (analytical closed-form (μ, L) convergence map from the identified paths) is treated in `lit_B_fxlms.md` and not repeated here. Several gaps below are cheaper if G0 succeeds.

| Rank | Gap | Evidence that it is open | S | Confidence |
|---|---|---|---|---|
| **1** | **Uncertainty-aware convergence maps.** Propagate the sd-DMDc identification uncertainty (BOP-DMD bagging, frequentist subset ensembles à la Palma 2025, or analytic moment propagation) into **probability maps P(t_conv < 5 s \| μ, L)** and confidence bands on convergence time | Base paper maps are deterministic. DMD UQ exists (5.1–5.5) but is never propagated to FxLMS. FxLMS robustness theory (3.9–3.15) is never fed with data-driven uncertainty. SciSpace and Consensus found no item | **High** | 0.8 |
| **2** | **Robust 4-D tuning (μ1, μ2, L1, L2) with crash constraints.** BO (BADS) or pattern search versus PSO on the DMDc twin. Objective: expected or worst-case convergence time over the ensemble from gap 1. Divergence as a crash constraint, with early stopping | Explicit future work of the base paper (lines 1411–1418). Benchmark 6.1 identifies the best optimiser class for d = 4. PSO/GA tuning of FxLMS exists (6.6, 3.5) but is nominal, not robust, and not on a DMDc twin | **High** | 0.7 for the robust version; 0.4 for a plain PSO/BO tuning (incremental) |
| **3** | **Time-varying plant on the virtual testbed, with online or windowed DMDc as the secondary-path tracker for FxLMS.** Temperature-scheduled E(T) and MFC gain(T), or added mass. Compare against (a) offline sd-DMDc, (b) Eriksson/Akhtar/Carini auxiliary-noise FIR SPM, (c) the noise-free Niu-type SPM | Online DMD(c) is used for LQR or MPC (1.4, 1.12) and for modal tracking (1.8, 1.9), but never as the FxLMS SPM engine. Plate online SPM uses FIR-LMS (2.10) | **Medium-high** | 0.75 |
| **4** | **Like-for-like benchmark of advanced adaptive algorithms on one identified twin, scored by the size and robustness of the effective region:** NFxLMS, two or three VSS families, leaky, MFxLMS, FxRLS, convex combination, plus a Kalman-type update | Review 3.1 says comparative studies are limited. The base paper's Table 1 compares only qualitatively | **Medium-high** (easy) | 0.6 (low intrinsic novelty) |
| **5** | **Hybrid μ-synthesis inner loop plus FxLMS outer loop on the DMDc model, with the uncertainty block built from the DMDc ensemble.** The inner loop "flattens" or damps the secondary path and enlarges the effective (μ, L) region | Hybrids exist (Qin 2021; Song 2005). μ from identified modal perturbations exists (4.2, 4.3). No item builds the uncertainty from a DMD ensemble, or measures the effect on FxLMS convergence regions | **Medium** (μ-synthesis in Python is heavier; D-K needs custom code or slycot) | 0.7 |
| **6** | **DMDc-model-assisted ESO/ADRC** as an alternative, or a complement, to FxLMS for tonal plus band-limited disturbances on the same plate | ADRC on identified plate models exists (4.6, Gu 2025). None uses a DMDc model in the ESO | Medium | 0.6 |
| **7** | **DMDc/Koopman-MPC or RL trained on the DMDc twin** for plate AVC | Exists in fluids only (4.8, 4.12). Structural RL uses FEM plus identification (4.10). Diverges from the FxLMS foundation, and real-time claims need hardware | Low-medium | 0.6 |

**Notes on gaps 1–3: falsifiable hypotheses, numbers needed and risks.**

**Gap 1.**
- *H1a:* across B = 100 bootstrap sd-DMDc models fitted to the virtual-plate data, the 90% band of the simulated convergence time contains the virtual-plate ("truth") convergence time at more than 80% of the (μ, L) grid points.
- *H1b:* the robust region where P(t < 5 s) ≥ 0.95 is at least 15% smaller in area than the nominal region. It shrinks most along the large-μ edge, where the base paper's simulation leaves the region earlier than the experiment.
- *Numbers to store:*
  - eigenfrequency CIs of the identified modes near 85, 120 and 280 Hz;
  - phase-error distribution of Ŝ at the control frequency, against the ±90° bound (3.9) and the 60° MFxLMS bound (3.12);
  - grid probability maps.
- *Risk:* bootstrap of time-series snapshots needs block resampling to respect the dynamics; BOP-DMD's subset sampling handles this.
- *Cost:* B × 120 grid points × one FxLMS run. With FxLMS at 10 kHz for 10 s, about 10^5 samples per run, roughly 12,000 runs. That is hours on a laptop in numpy, or minutes if G0 provides a closed form.

**Gap 2.**
- *H2:* under a ±3% frequency drift, the robust optimum loses less than 20% in convergence time, whereas the nominal 4-D optimum diverges in at least 10% of ensemble members.
- *Baselines:* the base paper's 3.48 s (simulation) and 3.81 s (experiment).
- *Budget:* compare BADS, pattern search and PSO at 25·d = 100 evaluations, as in 6.1.

**Gap 3.**
- *H3:* windowed DMDc with forgetting factor λ ∈ [0.98, 0.999] keeps the phase error of Ŝ below 60° during a 1–3% frequency ramp. Offline Ŝ exceeds 90° and diverges.
- *Main risk:* closed-loop identifiability. A tonal control signal poorly excites the secondary path, so online DMDc may still need auxiliary noise. That comparison is itself publishable.
- *Drift magnitudes must be swept, not assumed:*
  - CFRP frequency per °C is UNVERIFIED;
  - GFRP: −0.9% to −9% per 10 °C;
  - MFC actuation: −27.5% to +70% (2c).

---

## 9. The 10 most relevant references (for the orchestrator)

1. Zhang H., Rowley C.W., Deem E.A., Cattafesta L.N., "Online dynamic mode decomposition for time-varying systems", *SIAM J. Appl. Dyn. Syst.*, DOI 10.1137/18m1192329. [PARTIAL: year]
2. Deem E.A. et al., "Adaptive separation control of a laminar boundary layer using online dynamic mode decomposition", *J. Fluid Mech.* 2020, DOI 10.1017/jfm.2020.546. [VERIFIED]
3. Hamdan T.B. et al., "Tracking distributed parameters system dynamics with recursive DMD with control", *SIAM J. Appl. Dyn. Syst.* 2023, DOI 10.1137/22m1478665. [VERIFIED]
4. Pu Y.-X. et al., "Multi-channel adaptive active vibration control of piezoelectric smart plate with online secondary path modelling using PZT patches", *MSSP* 2019, DOI 10.1016/j.ymssp.2018.10.019. [VERIFIED]
5. Niu W. et al., "Adaptive vibration suppression of time-varying structures with enhanced FxLMS algorithm", *MSSP* 2019, DOI 10.1016/j.ymssp.2018.08.009. [VERIFIED]
6. Akhtar M.T., Abe M., Kawamata M., "A new variable step size LMS algorithm-based method for improved online secondary path modeling in ANC systems", *IEEE TASLP* 2006, DOI 10.1109/tsa.2005.855829. [VERIFIED]
7. Sashidhar D., Kutz J.N., "Bagging, optimized dynamic mode decomposition (BOP-DMD)", *Phil. Trans. R. Soc. A* 2022, DOI 10.1098/rsta.2021.0199. [PARTIAL: co-author]
8. Li F. et al., "Predicting the performance and optimizing the control parameters of an AVC system by pre-identifying the secondary path", *J. Sound Vib.* 568 (2024) 117953, DOI 10.1016/j.jsv.2023.117953. [VERIFIED]
9. Stenger D. et al., "Benchmark of Bayesian optimization and metaheuristics for control engineering tuning problems with crash constraints", arXiv 2211.02571, DOI 10.48550/arxiv.2211.02571. [VERIFIED]
10. Li P. et al., "Robust control of a vibrating plate using μ-synthesis approach", *Thin-Walled Struct.* 2003, DOI 10.1016/s0263-8231(03)00049-1. [VERIFIED]

**Runners-up:**
- Snyder & Hansen 1994, DOI 10.1109/78.285659 (±90° bound).
- Boucher, Elliott & Nelson 1991, DOI 10.1049/ip-f-2.1991.0042.
- Palma et al. 2025, DOI 10.1007/s42241-026-0015-z (ensemble DMDc UQ).
- Yang F.-R. et al. 2020, DOI 10.1109/taslp.2020.3012056.
- Akbar et al. 2026, DOI 10.1063/5.0308388.
- Liu T. et al. 2024, DOI 10.1016/j.ymssp.2023.110876.
- Hobeck J. et al. 2016, DOI 10.1063/1.4943947.
- Alfatlawi et al. 2020, DOI 10.3934/jcd.2020009.

---

## 10. Open items (UNVERIFIED or not found)

- **CFRP plate natural-frequency shift per 10 °C near room temperature:** UNVERIFIED. No abstract gave a number. Galea & White 1993 (DOI 10.1006/jsvi.1993.1225) and Galea's 1989 Southampton thesis (eprints.soton.ac.uk/52247, seen only as a WebSearch link) probably contain it.
- **MFC M-2814-P1 d33 temperature coefficient** from the manufacturer datasheet: not found. Only a generic piezo-fibre-composite value (−27.5% free strain from −15 to 40 °C, low-Tg matrix) and the qualitative trend (Lin 2017) were obtained.
- **Kuo & Vijayan 1997** co-author: not displayed by the tool.
- **Eriksson & Allie** year (1986 versus 1989): inconsistent.
- **Zhang Y. et al. 2025** (bolted composite plates under variable loads, *MSSP* 112496): abstract not shown, so its method is unknown. Important to read, because the base paper cites it as ref [29].
- **Ouyang et al. 2026, *Eng. Struct.*** (sd-DMDc participation matrices for PPF): abstract not shown.
- **Not found despite search:**
  - any use of online or recursive DMD(c) as the secondary-path estimator in FxLMS;
  - any propagation of DMD uncertainty to FxLMS convergence maps;
  - any structural AVC that uses a DMDc model as the MPC predictor or RL environment.
