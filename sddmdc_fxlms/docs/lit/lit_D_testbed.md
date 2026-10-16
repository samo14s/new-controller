# Literature sweep D: making the virtual testbed credible to reviewers, and free tools

Prepared 2026-10-16 for `sddmdc_fxlms` (base article: Ouyang, Dong, Liu & Zhou, *Thin-Walled Struct.* 228 (2026) 115090, doi:10.1016/j.tws.2026.115090). Scope: questions D1–D5 from the workflow. The main line (paper 1: damping budget, closed-form maps, re-analysis of the published data) is not revisited here. The testbed being judged is the one described in `docs/02_VIRTUAL_TESTBED.md`, `src/plate.py` and `src/testbed.py`.

## How this sweep was run (read first)

- **Tools that worked:** Consensus `search` (about 15 queries), SciSpace `search-papers` (3), WebSearch (about 15), the PubMed MCP for metadata and one PMC full text, and the PyPI JSON API. I downloaded wheels from files.pythonhosted.org and inspected their source with grep, without running them.
- **Tools that failed:**
  - WebFetch got a DNS failure on every host I tried (smart-material.com, mechano-transformer.com).
  - curl through the proxy got a 403 for mdpi.com, arxiv.org, doi.org, ncbi, openalex and readthedocs.
  - The `gh` API refused github.com repositories.
  - The Wiley Scholar Gateway free quota is used up until 2026-11-01.
- **What follows from those failures:**
  - Every manufacturer datasheet number below comes from search-engine snippets of the vendor pages, not from the PDF opened in full. They are tagged [PARTIAL-snippet] and must be re-checked against the PDF (`smart-material.com/wp-content/uploads/2025/02/MFC_V2.4-datasheet-web.pdf`) before submission.
  - Journal articles were judged on their abstracts only, except Sol et al. 2020, whose PMC full text I read (its tables were stripped).
- **Tags:**
  - **[VERIFIED]**: title, venue and DOI (or URL) all appeared in a tool result in this session.
  - **[PARTIAL]**: the item was seen, but one field is missing or the numeric value comes only from a snippet. The missing part is stated.
  - **[UNVERIFIED]**: the item was not seen in this session. It is listed only so it can be checked later.
- Statements marked *Analysis:* are my own calculations or reasoning. Calculations use the project's own code (`src/plate.py`, run in this session) or the formulas written out.

---

## 0. Executive summary

1. **MFC modelling (D1).** The testbed's MFC-P1 coupling, Λx = d33/Δ = 3.7e-10/0.5e-3 = 0.74 ppm/V, coincides with the datasheet low-field free strain per volt (~0.75 ppm/V).
   - *Analysis:* the blocked force it implies, 0.054 N/V over the 14 mm active width, is 73–85 % of the datasheet value (0.063–0.073 N/V).
   - Two items need documenting:
     - (a) MFC E2 = 30.34 GPa, copied from the paper's Table 2, against a datasheet E2 of 15.86 GPa.
     - (b) **Voltage headroom.** With g_A = 206.5, the ±4 V identification chirp maps to ±826 V. That breaks the P1 limit of −500 V unless a DC bias of at least +326 V is assumed, and it puts the actuator in the high-field regime (>1 kV/mm).
2. **Damping (D2).** CFRP laminates in the sources found span about 0.2–2 % modal damping. The low end is typical of fibre-dominated laminates at small strain in a rigid clamp.
   - The calibrated ζ = 0.24 % and the inverted 0.19–0.30 % are at the low end of that range but plausible.
   - Damping in polymer composites rises with amplitude (Li et al. 2018/2020), so µε-level AVC data sit at the floor.
   - The inverted ζ6 (torsion, 0.30 %) above ζ3 (bending, 0.19–0.20 %) agrees in direction with Adams–Bacon theory, in which shear-dominated modes dissipate more. This supports adding mode-dependent, strain-energy-based damping to the truth.
3. **Electronics (D3).** AVC simulation papers model the amplifier as a first-order lag (Gawryluk 2020) and split the secondary path into electronics, structure and an ADC/DAC term modelled as a half-sample phase shift (Li F. et al., JSV 2023).
   - Charge-amplifier noise models exist (Durdaut 2017; Ali 2024), but none of the AVC papers found uses them.
   - The testbed's 1 mV white noise reproduces the paper's 45 dB SNR (*Analysis:* 45.7 dB at 0.272 V peak).
   - Still missing from the testbed: a computational delay of one sample (4.3° at 120 Hz, 10.1° at 280 Hz), ADC quantization, the saturation block of the paper's Fig. 9, and non-white noise.
4. **Methodology (D4).** The inverse crime is defined in Wirgin 2004 and Kaipio & Somersalo 2007.
   - The testbed already avoids its plain form: truth model ≠ identified model, electronics and noise are added, and calibration and validation use separate sets.
   - Three residual criticisms remain:
     - (i) The truth, the "FEM-like" baseline and the noise model all come from the same code and the same assumptions.
     - (ii) The truth frequencies are pinned to Table 4.
     - (iii) The four ζ estimates are partly derived through the project's own FxLMS theory.
   - None of the purely numerical AVC papers in TWS, JSV or MSSP 2022–2026 that I found calibrates against published experiments. All of them validate code against code. So the project's protocol would be stronger than the norm, but it must be described in V&V language.
5. **Tools (D5).** All are free, and versions were checked on PyPI today:
   - python-control 0.10.2 has ERA (`eigensys_realization`) and Markov-parameter estimation (`markov`), plus h2syn/hinfsyn/mixsyn through slycot 0.7.0. It has no musyn.
   - PyDMD 2025.8.1 has `DMDc` and `BOPDMD` (eigenvalue std, eig constraints). BOPDMD takes no control input.
   - N4SID is available through `nfoursid` 1.0.2 (MIT) and through **`sippy-unipi`** 1.0.1 (LGPL-3), which also does MOESP, CVA and PARSIM. Beware: PyPI "sippy" is an unrelated SIP telephony stack.
   - scikit-fem 12.0.2 ships Morley, Argyris and BFS Kirchhoff-plate elements, which are enough for an independent second truth.
   - sdypy-EMA 0.31.0 does LSCF pole picking for a "virtual modal test".

---

## 1. D1. MFC M-2814-P1 (d33) and M-2814-P2 (d31): datasheet values and conversion into model coefficients

### 1.1 Datasheet values (Smart Material; all [PARTIAL-snippet] unless stated)

| Quantity | P1 (d33, IDE, elongator) | P2 (d31, contractor) | Source and notes |
|---|---|---|---|
| Operating voltage | −500 to +1500 V | −60 to +360 V | Vendor product pages and V2.4 datasheet (snippet). An older Quatek copy lists the P2 poling voltage as +450 V. |
| Free strain per volt (rod direction, low→high field) | ~0.75–0.9 ppm/V | ~1.1–1.3 ppm/V | V2.4 datasheet and product pages (snippet). |
| Piezo constants, low field (<1 kV/mm, unbiased) | d33 = 4.0e2 pC/N | d31 = −1.7e2 pC/N | V2.4 datasheet (snippet). |
| Piezo constants, high field (>1 kV/mm, biased) | d33 = 4.6e2 pC/N | d31 = −2.1e2 pC/N | V2.4 datasheet (snippet). |
| Max strain over the full range | up to ~1800 ppm | up to ~750 ppm | Product pages and distributor (snippet). |
| Elastic (active area) | E1 = 30.336 GPa, E2 = 15.857 GPa, ν12 = 0.31, G12 = 5.515 GPa, ρ = 5440 kg/m³ | same | Table in a 2024 open-access paper reproducing the datasheet (URL pmc.ncbi.nlm.nih.gov/articles/PMC11333884/table/tbl3) [PARTIAL: paper identity not confirmed]. |

**Model-specific rows:**
- **M-2814-P1** (active 28 × 14 mm, overall 38 × 20 mm):
  - **Capacitance:** 1.15 nF in the datasheet and 1.3 nF on the product page. A distributor lists 1.9 nF, and the older Quatek copy lists 0.61 nF, which is probably a transcription error. A 2025 study measured 1.176 nF (PMC12525912, snippet).
  - **Free strain:** 1160 ppm (older: 1550 ppm).
  - **Blocking force:** 146 N (older: 195 N).
- **M-2814-P2** (active 28 × 14 mm, overall 37 × 18 mm):
  - **Capacitance:** 43 nF (product page), 30.78 nF (V2.4 datasheet) and 25.7 nF (older copy).
  - **Free strain:** −630 ppm (older: −700 ppm).
  - **Blocking force:** −76 N (older: −85 N).

**Conclusions for the testbed:**
- The active area of 28 × 14 mm used in `plate.py` is confirmed (snippet, several sources).
- The "IDE pitch 0.5 mm" is not a datasheet entry in the snippets I saw. It enters only through Λx = d33/Δ, and that product matches the datasheet's 0.75 ppm/V. *Analysis:* document Λx as calibrated to the datasheet free strain per volt, not as d33 and Δ taken separately.
- The paper's Table 2 gives the MFC as in-plane isotropic (E1 = E2 = 30.34 GPa). The datasheet gives E2 = 15.86 GPa. *Analysis:* this affects only the patch stiffness, which is small next to a 3 mm CFRP plate (MFC thickness 0.18 mm). One sensitivity run with E2 = 15.86 GPa, reporting the change in the six frequencies and in |Gs| at 120 and 280 Hz, settles it.
- **Capacitance and amplifier current.** *Analysis:* I = 2πfCV.
  - Identification chirp at 300 Hz and 826 V: 2π·300·1.3e-9·826 ≈ 2.0 mA.
  - Converged control at 280 Hz and 457 V: ≈ 1.0 mA.
  - Both are far below any HV-amplifier current limit, so slew-rate and current saturation can be stated to be irrelevant.

### 1.2 How the literature converts datasheet values into plate-model coefficients

Six families of approach appear in the literature. All items below are [VERIFIED] unless tagged otherwise.

1. **Induced-strain (thermal-analogy) actuator with a homogeneous active layer.** The free strain Λ = d33_eff/Δ (P1) or d31·V/t (P2) is applied to an orthotropic layer. The equivalent stress resultants are M = Q·Λ·t·z_p, and the generalized force is f_i = −∫(Mx φ,xx + My φ,yy) dA. This is exactly what `plate.py` does.
   - Papers using it: Zhang S.-Q. & Schmidt 2015, *Compos. Struct.* (FE for laminated thin-walled structures with orthotropic MFC-d31 and MFC-d33, arbitrary fibre angle; doi:10.1016/j.compstruct.2015.02.051); Gohari et al. 2018, *SMS* (8-node FSDT shell element; doi:10.1088/1361-665x/aacc95); a 2018 SMO paper with an MFC "orthotropic layer with initial strain" model (doi:10.1007/s00158-018-2111-3).
   - Classical sources: Crawley & de Luis 1987 and Dimitriadis, Fuller & Rogers 1991 [UNVERIFIED, not seen in this session].
2. **Equivalent force applied along the actuator edges** of an FE plate. Dong et al. 2017, *SMS* validated it on a laminated composite plate with MFC actuators and impact tests (doi:10.1088/1361-665x/aa7bd0).
3. **Equivalent monolithic piezo with two end electrodes.** Emad et al. 2021, *Materials* (doi:10.3390/ma14154316) reports 98.6 % lower computational time than a detailed IDE model and was validated with rosette strain gauges.
4. **Homogenized properties (mixing rules, RVE, asymptotic homogenization):**
   - Deraemaeker & Nasser 2010, *IJSS* (doi:10.1016/j.ijsolstr.2010.08.006): agrees with mixing rules; local IDE effects are small.
   - Prasath & Arockiarajan 2014, *Sens. Actuators A* (doi:10.1016/j.sna.2014.04.008).
   - Biscani et al. 2011 (doi:10.1016/j.compositesb.2010.12.009).
   - Trindade & Benjeddou 2016, *Compos. Struct.* (doi:10.1016/j.compstruct.2015.10.011): the effective d and ε depend strongly on the electric field.
   - Kuang et al. 2019, *Compos. B* (doi:10.1016/j.compositesb.2018.09.068): equal-power equivalent properties for a d31 MFC; predicted actuation free strain **0.93 µε/V, "close to the manufacturer specification"**.
   - Camarena & Vel 2019, *SMS* (doi:10.1088/1361-665x/ab0b60): plane-stress homogenization **under-predicts the lateral modulus by 8 %** and over-predicts the transverse shear moduli by 100–300 %.
   - Steiger & Mokrý 2015, *SMS* (doi:10.1088/0964-1726/24/2/025026).
   - Li Y.-X. et al. 2016, *JIMSS* (doi:10.1177/1045389x16633763).
   - Ji et al. 2024, *Compos. Struct.* (doi:10.1016/j.compstruct.2024.118082): inhomogeneous poling.
   - Silva et al. 2022, *Compos. Struct.* (doi:10.1016/j.compstruct.2022.116292).
5. **Electric-field non-uniformity under the IDE.** Sharghi et al. 2021, *IJMS* (doi:10.1016/j.ijmecsci.2021.106864): a constant-field model **over-predicts capacitance by ~41 % and coupling by ~22 %** against a continuous-field model, consistent with the empirical correction factors used earlier. The continuous-field capacitance is within 2 % of the manufacturer value. Bowen et al. 2006 (doi:10.1007/s10832-006-9862-8): ~80 % of the maximum d33 strain is reached when electrode spacing / substrate thickness > 4. Bent & Hagood 1997 (doi:10.1177/1045389x9700801101). *Analysis:* calibrating Λ to the datasheet free strain per volt, as the project does, sidesteps this error. The paper should say so.
6. **Nonlinearity of MFC actuation:**
   - Williams et al. 2003 (doi:10.1115/imece2003-41205): d33 and d31 depend on stress and field.
   - Qu et al. 2019, *Ceram. Int.* (doi:10.1016/j.ceramint.2019.07.223): ellipse-like hysteresis loops whose size depends on peak-to-peak voltage, frequency and DC bias.
   - Umar et al. 2024 (doi:10.1177/14613484241295460): Bouc–Wen + ARX Hammerstein model for MFC in VSS-FxLMS.
   - Wang D. et al. 2025, *SMS* (doi:10.1088/1361-665x/ae2f85): modified Prandtl–Ishlinskii + dynamics; maximum RMSE rate 3.22 %.
   - Tan et al. 2018 (doi:10.1016/j.compstruct.2017.12.056).

**Sensing gain.** For a uniformly electroded d31 sensor at height z_s, the generated charge is Q ≈ e31·∫εx dA + e32·∫εy dA, with εx = −z_s w,xx.
- Voltage mode: V = Q/C_p, with a high-pass corner 1/(2π R_in C_p).
- Charge-amplifier mode: V = −Q/C_f, with a high-pass corner 1/(2π R_f C_f).
- Supporting literature:
  - Sodano, Park & Inman 2004, *MSSP* (doi:10.1016/s0888-3270(03)00081-5): MFC as sensor, self-sensing, impedance.
  - Shahab & Erturk 2017, *JIMSS* (doi:10.1177/1045389x16672732): two-way coupling with mixing rules.
  - Kuang et al. 2019 (above): the same equivalent properties serve actuation, sensing and harvesting.
  - Konka et al. 2013, *Sens. Actuators A* (doi:10.1016/j.sna.2012.12.039): sensitivity, linearity and noise calibration of MFC and PFC sensors.
- The "modal sensor" form (charge ∝ slope difference at the electrode edges) is classical: Lee & Moon 1990 [UNVERIFIED].

### 1.3 Numbers for the project's testbed (*Analysis:* computed in this session with `src/plate.py`)

- **Equivalent moment.** A1 and A2: Mx = 5.909e-3 N·m/m per V and My = −1.074e-3 N·m/m per V. The moment arm z_p = 1.543 mm.
  - Over the 14 mm active width: **8.3e-5 N·m/V**.
  - At the converged 120 Hz voltage (1.570 V × 206.5 = 324 V): 0.027 N·m.
- **Blocked-force check.** Mx/z_p × width = 3.83 N/m/V × 0.014 m = **0.054 N/V**.
  - Datasheet blocking force per volt: 146 N / 2000 V = 0.073 N/V (high-field, full range), or about 0.063 N/V scaled to low-field d33 (400/460).
  - So the testbed sits at 73–85 % of the datasheet value. That is consistent, given that the testbed uses the low-field Λ and a Q11 for E1 = E2.
- **Sensor gain check.** S_sens = 1e5 V/strain (0.1 V/µε).
  - |d31|·E1·A_active = (1.7–2.1)e-10 × 30.3e9 × 3.92e-4 m² ≈ 2.0–2.5 µC per unit strain, i.e. 2.0–2.5 nC/µε (ignoring the d32·εy term and bond shear lag).
  - In voltage mode with C_p = 30.8–43 nF this gives 0.047–0.081 V/µε.
  - With a charge amplifier, 0.1 V/µε needs C_f ≈ 20–25 nF (≈0.04–0.05 mV/pC). The 0.5 Hz high-pass then needs R_f ≈ 13–16 MΩ.
  - Conclusion: S_sens is physically realizable within a factor of about 2. The ECON MI-2004 range is UNVERIFIED (no datasheet found).
- **Voltage headroom (important).**
  - With g_A = 206.5, the ±4 V identification chirp gives ±826 V at the MFC: below the −500 V limit, and 1.65 kV/mm at the assumed Δ = 0.5 mm, which is high field.
  - The converged control voltages are 324 V peak (120 Hz), 240 V (280 Hz) and 457 V (two channels): within ±500 V, at 0.5–0.9 kV/mm.
  - Either the real rig used a DC bias (which the paper does not report), or the split of gain between g_A and S_sens (and g_E) differs from the testbed's split. Only the products are identifiable from the published data.
  - Recommendation: state this in `docs/02` §3. Then either (a) add a +350 V bias and a −500/+1500 V clip to the truth, or (b) report the gains as an identifiable product, with g_A ≤ 125 V/V as a physically admissible alternative split, and show that no conclusion changes.
- **CoreMorrow E01.A series** (vendor pages via WebSearch) [PARTIAL-snippet]: −500 to +1500 V output for MFC, small-signal bandwidth ~10 kHz, large-signal bandwidth 2–3 kHz, gain not stated.
  - *Analysis:* a first-order lag at 2–5 kHz gives 1.4–3.4° phase at 120 Hz and 3.2–8.0° at 280 Hz.
  - Compare the testbed's 5 kHz. Run one sensitivity at 2 kHz.

---

## 2. D2. Modal damping of CFRP cantilever plates: is ζ = 0.19–0.30 % plausible?

### 2.1 Values found (ζ unless stated; loss factor η ≈ 2ζ; specific damping capacity ψ ≈ 4πζ)

| Source | Specimen and test | Reported damping | Tag |
|---|---|---|---|
| Kim C.-J. 2021, *Crystals* 11:311 (doi:10.3390/cryst11030311) | Rectangular CFRP specimens at 0–90°, impact EMA | 0° specimen, 1st bending ~218 Hz: **~1.7 %** | [PARTIAL: value from search snippet of a table] |
| Kim C.-J. 2021, *Crystals* 11:781 (doi:10.3390/cryst11070781) | Same family, viscous-damping sensitivity | 30° specimen, 1st bending: **~0.39 %** | [PARTIAL: snippet; "should be checked"] |
| Zhang et al. 2020, *Shock Vib.* (URL onlinelibrary.wiley.com/doi/10.1155/2020/6021640) | CFRP motor frame | 1st mode **0.91 %** | [PARTIAL: snippet] |
| Wang W. et al. 2026, *Polymers* (doi:10.3390/polym18050578) | CFRP laminate, free-vibration log decrement | baseline **ζ = 2.1 %** (beam, epoxy E51/DICY) | [VERIFIED abstract] |
| Gong et al. 2022, *Compos. Sci. Technol.* (doi:10.1016/j.compscitech.2022.109309) | DMA of CFRP | η = 0.0345 at 1 Hz, 0.23 % strain (≈1.7 % ζ, but large strain and not modal) | [VERIFIED abstract] |
| Kayaaslan et al. 2022 (doi:10.1177/09673911221119669) | UD vs woven CFRP, free-free | woven SDC **+50 %** over UD | [VERIFIED abstract] |
| Rueppel et al. 2017, *CST* (doi:10.1016/j.compscitech.2017.04.011) | Angle-ply CFRP and flax | damping rises with angle-ply orientation below 300 Hz | [VERIFIED abstract] |
| Li H. et al. 2018, *Nonlinear Dyn.* (doi:10.1007/s11071-018-4486-5); Li H. et al. 2020, *Appl. Math. Model.* (doi:10.1016/j.apm.2019.11.048) | TC300 carbon/epoxy thin plates | **amplitude-dependent** damping; model errors on damping ≤ 9.6 % | [VERIFIED abstracts; numbers not shown] |
| Vignjevic et al. 2025, *Vibration* (doi:10.3390/vibration8040072) | UD CFRP 4.6 mm plates clamped on a shaker, half-power | loss factor linear in f for modes 1–2 | [VERIFIED abstract; numbers not shown] |
| Adams & Bacon 1973, *J. Compos. Mater.* (doi:10.1177/002199837300700401); Adams & Maheri 1994, *CST* (doi:10.1016/0266-3538(94)90058-2); Lin, Ni & Adams 1984 (doi:10.1177/002199838401800204) | Theory and measurement of flexural vs torsional damping against fibre angle; FE prediction of SDC of CFRP plates by mode | torsional (shear) damping > flexural | [VERIFIED abstracts] |
| Song et al. 2026, *Modelling* (doi:10.3390/modelling7050177) | CFRP cantilever plates with MFCs, LQG | "inherently low damping" (no number in abstract) | [VERIFIED abstract] |

Not found: a CFRP cantilever **plate** of 3 mm thickness with bonded MFCs and published per-mode ζ below 0.5 %. Liu T. et al. 2024, *MSSP* (doi:10.1016/j.ymssp.2023.110876) and Li H.-D. et al. 2026, *MSSP* (doi:10.1016/j.ymssp.2025.113666) probably contain such tables. Read them in full before writing the damping paragraph.

### 2.2 Verdict

- **Plausible, at the low end.**
  - The 0.1–0.3 % range is what fibre-dominated, autoclaved laminates give at µε-level strain in a stiff clamp.
  - The 1–2 % values above come from thin coupons, matrix-dominated or woven layups, hammer amplitudes, DMA at large strain, or include clamp and air losses.
  - The plate's E1 = E2 = 60 GPa suggests a woven or quasi-isotropic layup, which pushes damping upward (Kayaaslan: +50 % SDC). That is a counter-argument a reviewer may raise.
- **The project's evidence is internally consistent.** Five estimates agree:
  - 0.24 % from the Fig. 16c calibration;
  - 0.23 % from inverting map 18;
  - ~0.17 % from beat decay in Fig. 14a, which is model-light;
  - 0.19–0.20 % from the 85 Hz edge ratio;
  - ~0.21 % from the t_min·f_n ≈ 330-cycle regularity.
  - Caveat for reviewers: three of these five pass through the project's FxLMS theory. Present the beat-decay estimate, which uses no FxLMS theory, first.
- **Mode dependence.** The project's inverted ζ6 = 0.30 % (experiment, torsion) against ζ3 ≈ 0.19–0.20 % (bending) points in the direction predicted by Adams–Bacon and strain-energy theory. Shear-dominated modes draw on the larger in-plane shear loss factor.
  - *Analysis:* a strain-energy (MSE) damping model gives ζ_n = Σ_k η_k U_k,n /(2 U_n), with k ∈ {D11, D22, D12, D66}. It is computable directly from the Ritz matrices already in `plate.py`. With η_D66 > η_D11 it reproduces this ordering.
  - Calibrate η on two modes (3 and 4), predict the third (6), and report the prediction. This removes limitation 1 of `docs/02` §5 ("one damping for all modes") with a physically grounded, falsifiable model.

---

## 3. D3. Charge-amplifier and HV-amplifier dynamics and noise models used in AVC simulation

**What AVC simulation papers actually do (all [VERIFIED abstracts]):**
- Gawryluk et al. 2020, *Compos. Struct.* (doi:10.1016/j.compstruct.2020.111898): the control system, including the amplifier, is described by a **first-order inertial term** whose time constant is identified from experiments. This is the most common model and is what `testbed.py` uses (1st order, 5 kHz).
- Li F. et al. 2023, *JSV* 568:117953 (doi:10.1016/j.jsv.2023.117953): the secondary path is split into (1) actuator and electrical equipment, identified by FRF; (2) structure, by impact; (3) **ADC/DAC, modelled as a phase shift of half the sampling time**. This is the model a JSV reviewer will compare against.
  - *Analysis:* a half-sample delay at 10 kHz is 2.2° at 120 Hz and 5.0° at 280 Hz. A one-sample computational delay is 4.3° and 10.1°.
  - The paper says that the dSPACE task takes 10.5–17.4 µs and that sensor-to-actuator latency is "well below 20 µs". So the ZOH half-sample is the minimum, and a full sample applies if the output is written at the next tick. Document which applies.
- HV drivers for MFC and piezo:
  - Zhu, Qiu & Lei 2026, *Sensors* (doi:10.3390/s26165064): a −500/+1500 V MFC driver with a voltage–current dual loop for a capacitive load up to 1 kHz; the prototype swing was +499.9/−133.3 V.
  - Xu L.-S. et al. 2020, *Sensors* (doi:10.3390/s20226528): bandwidth ~57 kHz, ripple < 2 mV.
  - Xu L.-S. et al. 2021, *Electronics* (doi:10.3390/electronics10060720): 43–53 kHz, ripple 16 mV or < 2 mV.
  - *Analysis:* output ripple of 2–16 mV on a several-hundred-volt actuator scale is negligible next to sensor noise.
- Charge-amplifier noise:
  - Durdaut et al. 2017, *IEEE Sensors J.* (doi:10.1109/jsen.2017.2759000): 12-source JFET charge-amplifier noise model.
  - Ali et al. 2024, *Micromachines* (doi:10.3390/mi15020283): the feedback resistor dominates noise at low frequency.
  - Levinzon 2005 (doi:10.1109/jsen.2005.859256).
  - Alnasser 2020 (doi:10.1109/jsen.2020.2970839): example low-frequency cutoff of 0.2 Hz.
  - A commercial in-line charge amplifier (PCB 422E55 spec sheet, URL pcbpiezotronics.de/…/422E55_specs.pdf) lists 0.5 Hz–100 kHz, 33 µV broadband noise (1–10 kHz) and 9.8 µV/√Hz at 1 Hz [PARTIAL-snippet]. This is not the ECON unit used in the paper.
- Measurement chain of the base paper (from `docs/00`): Keyence LK-G30 lasers, measured "sequentially"; ECON MI-2004 charge amplifier; dSPACE MicroLabBox.
  - LK-G30: repeatability 0.05 µm (0.01 µm with 4096-point averaging), range ±5 mm, sampling down to 20 µs in the family [PARTIAL: keyence.com page snippet]. The testbed's 0.5 µm laser noise is therefore conservative, which is fine but should be stated.
  - MicroLabBox: 24 channels of 16-bit ADC and 8 channels of 14-bit ADC, input range from −10 V [PARTIAL: equipment-listing snippet].
  - *Analysis:* 16-bit at ±10 V gives LSB = 0.305 mV and σ_q = LSB/√12 = 0.088 mV, 21 dB below the testbed's 1 mV noise, so quantization is negligible on 16-bit channels. On the 14-bit channels σ_q = 0.35 mV, only 9 dB below. State which applies.

**Gaps in the testbed relative to this practice:**
- the computational delay;
- the ADC/DAC quantization;
- the saturation block that the paper's Fig. 9 shows, with an unpublished limit;
- noise that is not white (50 Hz harmonics, 1/f from R_f);
- the sequential laser measurement.

The last item matters most for credibility. If the 35 displacements were recorded in repeated runs, the identification data mix run-to-run variability (re-clamping, temperature) into one snapshot matrix. The testbed's simultaneous, noise-only data cannot reproduce that. *Analysis:* simulate it as K runs with f_n perturbed by ±0.5 % and ζ by ±30 %. These are the magnitudes already proposed in audit item M10.

---

## 4. D4. Methodology: the inverse crime, virtual and digital-twin testbeds, and purely numerical AVC papers

### 4.1 Inverse crime: definitions and remedies

- Wirgin 2004, arXiv:math-ph/0401050 (doi:10.48550/arxiv.math-ph/0401050) [VERIFIED].
  - Definition: the inverse crime occurs "when the same (or very nearly the same) theoretical ingredients are employed to synthesize as well as to invert data". Colton & Kress call it trivial and to be avoided; Wirgin questions whether it is always trivial.
  - Remedies cited by later papers: add noise, and use a different model to generate the data than to invert them.
- Kaipio & Somersalo 2007, *J. Comput. Appl. Math.* (doi:10.1016/j.cam.2005.09.027) [VERIFIED]: discretization and model-reduction error must be treated as part of the measurement-plus-modelling error. Neglecting it in ill-posed problems has serious consequences. The Bayesian **approximation-error** approach estimates its statistics.
- Calvetti et al. 2017, *Inverse Problems* (doi:10.1088/1361-6420/aaa34d) [VERIFIED]: iterative updating of the model-error distribution.
- Henderson et al. 2017 (doi:10.1007/s40295-016-0105-1) [VERIFIED]: a worked case in which committing the crime gives "misleadingly optimistic" estimates.
- Book: Kaipio & Somersalo 2005, *Statistical and Computational Inverse Problems*, Springer [UNVERIFIED]. Colton & Kress, *Inverse Acoustic and Electromagnetic Scattering Theory* [UNVERIFIED].

**Mapping to the testbed.** `docs/02` §1 already lists four deliberate differences: 40 modes against rank r, electronics, noise, and patch geometry. The remaining vulnerabilities a reviewer can name are:

1. **One code, one physics.** The truth, the "FEM-like" baseline (`control.py`) and the identification data share one Ritz basis, one modal-damping structure, and white Gaussian noise of known σ. Remedy: a second, independent truth (scikit-fem Kirchhoff FE; §5), with conclusions shown invariant across truths.
2. **Truth frequencies pinned to Table 4** (`f_override`). As a result the "0.02 % frequency error" of sd-DMDc measures identification on the truth, not agreement with the paper. Remedy: state it explicitly.
3. **Proportional, mode-independent damping.** An sd-DMDc model can represent it exactly. Remedy: MSE mode-dependent damping (§2.2), plus a small non-proportional term at the patches (bond-layer loss), which gives complex modes.
4. **The noise model equals the assumed noise model.** Remedy: coloured noise, mains harmonics, occasional outliers, and sequential runs (§3).
5. **Linear, time-invariant truth.** Remedy: an optional Bouc–Wen MFC block (Umar 2024) and an E(T) drift, both as sensitivity studies rather than as the baseline.

### 4.2 Virtual testbed versus digital twin: vocabulary

- The structural "digital twin" literature means a model coupled to a physical asset in both directions, with updating:
  - Chiachío et al. 2022, *Autom. Constr.* (doi:10.1016/j.autcon.2022.104333) [VERIFIED];
  - Oh et al. 2020 virtual sensing with FE updating (doi:10.5050/ksnve.2020.30.2.149) [VERIFIED].
- Zhang Y. et al. 2024, *Eng. Struct.* (doi:10.1016/j.engstruct.2024.118630) [VERIFIED] validates a DT engine on **"two virtual experiments"**. That is a precedent for virtual experiments as a validation device, but on a high-fidelity FE truth.
- I found **no AVC paper** that uses a calibrated twin as a stand-in for the experiment, with calibration and validation separated on published experimental data. That is a positioning point.
- *Recommendation:* call the testbed a "calibrated virtual testbed", or "surrogate experiment calibrated on published measurements", not a "digital twin". There is no physical asset in the loop.

### 4.3 Purely numerical AVC papers, 2022–2026, in the target journals (all [VERIFIED] title, venue and DOI from Consensus)

- **Thin-Walled Struct.:**
  - Su et al. 2024, piezo shunt on moving laminated plates (doi:10.1016/j.tws.2024.112340);
  - Jiang et al. 2023, PPF on a graphene-reinforced cantilever plate (doi:10.1016/j.tws.2023.110830);
  - Mahesh 2022, ACLD of sandwich plates (doi:10.1016/j.tws.2022.109547);
  - Zhen et al. 2023, ABH plate with active–passive piezo networks (doi:10.1016/j.tws.2023.110705);
  - Zhou K. et al. 2022, negative-capacitance shunt (doi:10.1016/j.tws.2022.109802);
  - Wang C. et al. 2024 (doi:10.1016/j.tws.2024.111656);
  - Cao et al. 2024 (doi:10.1016/j.tws.2024.112493);
  - Li P.-Q. et al. 2023, active flutter control (doi:10.1016/j.tws.2022.110323);
  - Gupta & Ray 2022, ACLD FE (doi:10.1016/j.tws.2021.108782);
  - Abbaspour et al. 2022 (doi:10.1016/j.tws.2022.110129).
- **J. Sound Vib.:**
  - Wang L. et al. 2023, uncertainty-oriented PID with interval method and Monte-Carlo check (doi:10.1016/j.jsv.2023.117588);
  - S. A. et al. 2022, frequency-adaptive optimal control, numerical (doi:10.1016/j.jsv.2022.117501).
- **Mech. Syst. Signal Process.:**
  - Yonezawa et al. 2022, model-free VCO + SPSA, "simulation studies" on five plants including time-varying ones (doi:10.1016/j.ymssp.2022.108801);
  - Zhao et al. 2022, semi-active MRE sandwich plate (doi:10.1016/j.ymssp.2022.108953).
- **Compos. Struct.:** Sharma 2021, porous FGPM velocity feedback (doi:10.1016/j.compstruct.2021.114815), just before 2022.
- **J. Vib. Control and Smart Mater. Struct.:** the FxLMS/AVC items found all contain experiments: Botti et al. 2024, JVC (doi:10.1177/10775463241305247); Luo et al. 2025, SMS (doi:10.1088/1361-665x/adbf7a). No numerical-only FxLMS-on-plate paper was found in these two journals (absence of evidence).

**Pattern** (*Analysis*):
- Numerical-only papers accepted in TWS, JSV and MSSP are mostly (a) new structural or material models with a classical controller, or (b) shunt or semi-active designs. They validate structure against structure (Ansys, literature benchmarks), rarely model electronics or noise, and none calibrates on published experiments.
- Adaptive-control (FxLMS) papers in MSSP and JSV almost always include an experiment: Liu T. 2024 (doi:10.1016/j.ymssp.2023.110876); Chen X.-H. 2025 (doi:10.1016/j.ymssp.2024.111959); Li F. 2023 (above).
- So a numerical FxLMS paper will be judged on how convincingly the virtual testbed substitutes for that experiment. Hence the V&V framing in §6.
- Useful for the results section: Forma et al. 2024, *JSV* (doi:10.1016/j.jsv.2024.118688), nine objective indicators for comparing active-control algorithms (transient speed, in-band reduction, out-of-band amplification) [VERIFIED].

---

## 5. D5. Free tools (versions and contents checked today on PyPI and in the wheel sources)

| Need | Package (PyPI name), version, last upload, licence | What it gives (checked in source) | Caveats |
|---|---|---|---|
| LTI analysis, ERA, robust synthesis | `control` 0.10.2, 2025-07-05, BSD-3 | `eigensys_realization(YY, r, m, n, dt)` (ERA from impulse-response or Markov data); `markov(Y, U, m)` (Markov parameters from I/O by least squares); `h2syn`, `hinfsyn`, `mixsyn` (need slycot); no `musyn` | `markov` assumes CA^kB = 0 beyond m. *Analysis:* at ζ = 0.24 % and f1 = 14 Hz the impulse response needs ln(100)/(ζω1) ≈ 22 s, i.e. ~2.2e5 samples at 10 kHz. Use free-decay tails, decimate to 1–2 kHz, or implement OKID (~50 lines). Ref: Fuller et al. 2021, IEEE CDC (doi:10.1109/cdc45484.2021.9683368) [VERIFIED] |
| SLICOT wrapper | `slycot` 0.7.0, 2026-06-16 | Riccati/Lyapunov solvers, balanced truncation, H∞ | Licence not shown in the PyPI JSON [UNVERIFIED; check GPL]. D–K iteration for μ must be custom |
| DMD / DMDc / BOP-DMD | `pydmd` 2025.8.1, 2025-08-01, MIT | `DMDc.fit(X, I, B=None)`; `BOPDMD.fit(X, t)` with `num_trials`, `trial_size`, `eig_constraints` ∈ {"stable", "imag", "conjugate_pairs"} and outputs `eigenvalues_std`, `modes_std`, `amplitudes_std`; also OptDMD, PiDMD, HankelDMD, SubspaceDMD, HAVOK, VarProDMD | **BOPDMD has no control input.** Apply it to the zero-input free-decay tails of each identification block for ζ_n ± std, or to delay-embedded data. Refs: Demo et al. 2018, JOSS (doi:10.21105/joss.00530) [VERIFIED]; Ichinaga et al. 2024, arXiv 2402.07463 [VERIFIED]; Sashidhar & Kutz 2022, Phil. Trans. A (doi:10.1098/rsta.2021.0199) [VERIFIED] |
| N4SID (light) | `nfoursid` 1.0.2, 2025-07-24, MIT | `NFourSID(dataframe, output_columns, input_columns, num_block_rows=2)`, Kalman filter | Single maintainer. Increase `num_block_rows` well above the default 2. Validate on a known system first |
| N4SID / MOESP / CVA / PARSIM, ARX/ARMAX | **`sippy-unipi`** 1.0.1, 2025-09-02, LGPL-3.0+ (repo CPCLAB-UNIPI/SIPPY) | `id_method` ∈ {"N4SID", "MOESP", "CVA"} plus the PARSIM family; depends on casadi and control ≥ 0.10 | **PyPI "sippy" 2.5.0 is an unrelated SIP/B2BUA telephony stack.** Do not `pip install sippy`. Reference paper Armenise et al. 2018 [UNVERIFIED] |
| Independent second truth (Kirchhoff plate FE) | `scikit-fem` 12.0.2, 2026-06-05, BSD | `ElementTriMorley`, `ElementTriArgyris`, 15-parameter plate triangle, `ElementQuadBFS` (C1 Bogner–Fox–Schmit) | You assemble the orthotropic plate and patch forms yourself (~200 lines); no ready plate model. Ref: Gustafsson & McBain 2020, JOSS (doi:10.21105/joss.02369) [VERIFIED] |
| Virtual modal test (ζ from simulated FRFs, as an experimentalist would) | `sdypy-EMA` 0.31.0 (alias `pyEMA`), 2026-09-29 | `get_poles(method='lscf' / 'lsce' / 'rfp')` + stabilization chart | Pairs with `pyFRF` 1.5.0 for H1/H2 estimators |
| Nonlinear ID (hysteresis studies) | `sysidentpy` 0.9.0 (NARMAX); `pysindy` 2.1.0 | NARX/NARMAX; SINDy-c | Optional |
| UQ and sensitivity | `SALib` 1.6.0; `chaospy` 4.3.21; `emcee` 3.1.6 | Sobol/Morris; PCE; MCMC | For Monte-Carlo bands on map edges |

**Classical references for the identification baselines:**
- ERA: Juang & Pappa 1985, *J. Guid. Control Dyn.* (doi:10.2514/3.20031) [VERIFIED].
- **Effects of noise on ERA, with Monte Carlo:** Juang & Pappa 1986 (doi:10.2514/3.20106) [VERIFIED]. This is the template for the SNR study.
- Consistent-mode indicator: Pappa et al. 1993 (doi:10.2514/3.21092) [VERIFIED].
- N4SID: Van Overschee & De Moor 1994, *Automatica* 30(1):75–93 [PARTIAL: citation printed in the nfoursid docstring; DOI not seen].

Practical note for the project (*Analysis*): the band of interest is ≤ 300 Hz, so decimate the 10 kHz identification data ×5–×10 (with an anti-alias FIR) before ERA, OKID or N4SID. This cuts Hankel sizes 25–100× and is standard practice. The controller still runs at 10 kHz.

---

## 6. What the testbed should add or document to satisfy reviewers (prioritized)

Effort: S < 1 day, M 1–3 days, L about 1 week. Items 1–6 matter for paper 1; 7–12 strengthen papers 2 and 3.

1. **A provenance table for every testbed parameter (S).** For each parameter give its value, source (paper table or figure, datasheet version, calibration step) and tag (published / datasheet / calibrated / assumed).
   - Include both the current and the older datasheet values for M2814-P1/P2 (§1.1).
   - Flag the MFC E2 choice: the paper's isotropic Table 2 against the datasheet's 15.86 GPa.
2. **An electromechanical cross-checks box in `docs/02` §2 (S):**
   - Λx = 0.74 ppm/V against the datasheet 0.75–0.9 ppm/V;
   - moment 8.3e-5 N·m/V;
   - blocked force 0.054 N/V against 0.063–0.073 N/V;
   - S_sens = 0.1 V/µε against 0.047–0.081 V/µε in voltage mode, or C_f ≈ 20–25 nF with a charge amplifier;
   - amplifier current ≤ 2 mA.
3. **The voltage-headroom statement (S, important).** ±4 V × 206.5 = ±826 V breaks the −500 V limit and enters the high-field regime. Either add a bias and clip to the truth, or present the gains as identifiable products, rerun with g_A ≤ 125 V/V (and S_sens and g_E rescaled to keep the published amplitudes), and show the conclusions unchanged.
4. **Mode-dependent damping (M).** Use the MSE/Adams–Bacon model with loss factors on D11, D22, D12 and D66. Calibrate on two modes, predict the third, and report it.
   - Cite the CFRP damping range (§2.1) and the amplitude dependence (Li H. 2018/2020).
   - Lead with the beat-decay ζ (0.17 %), which is independent of the FxLMS theory.
5. **Sampled-data realism (S):**
   - a one-sample computational delay, or a justification for half-sample only (Li F. 2023 JSV);
   - 16-bit ADC quantization (σ_q = 0.088 mV) and DAC quantization;
   - an actuator saturation block (Fig. 9).
   - Report the resulting phase change at 120 and 280 Hz (≤ 10°) and show that the map edges move less than one grid step.
6. **An inverse-crime paragraph naming the residual vulnerabilities and their tests (S).** Cite Wirgin 2004 and Kaipio & Somersalo 2007, and cover: same code, pinned frequencies, proportional damping, noise model = assumed model, LTI.
7. **A second independent truth (L).** Build a scikit-fem Morley or BFS Kirchhoff FE of the plate and patches. Recompute the G1–G4 gates and the map edges, and show that the conclusions are invariant. This is the strongest single answer to "the same code ran twice".
8. **Realistic noise and acquisition (M):**
   - coloured noise (1/f below ~10 Hz, 50/100/150 Hz lines);
   - sequential-laser acquisition as K runs with f ±0.5 % and ζ ±30 % run-to-run;
   - 0.5 µm laser noise justified against the LK-G30's 0.05 µm repeatability.
9. **Monte-Carlo bands (M).** Use ≥ 50 seeds × (noise, run-to-run, ζ, gain splits). Every reported number (t5 error, divergence μ, edge ratios) should carry a 90 % band. The Juang & Pappa 1986 Monte-Carlo template applies.
10. **Identification baselines (M; paper 2):**
    - ERA (python-control, with an OKID implementation or free-decay segments);
    - N4SID through `sippy-unipi` and `nfoursid`, on data decimated ×5–×10;
    - BOP-DMD ζ ± std on the zero-input tails.
11. **Optional nonlinear sensitivity (M):** a Bouc–Wen MFC block (loop width tuned to Qu 2019 / Umar 2024) and an E(T) drift. Report them as robustness, not as the baseline.
12. **Vocabulary and reproducibility (S):**
    - call it a "calibrated virtual testbed", not a "digital twin";
    - use V&V terms (code verification: Ritz convergence and exact-ZOH against fine-step integration; calibration on Fig. 16c; validation on the held-out published curves);
    - pin versions in `requirements.txt`;
    - archive the code and digitized data on Zenodo with a DOI;
    - give the testbed limitations their own section (already `docs/02` §5).
    - ASME V&V 10 / Oberkampf & Roy as the V&V vocabulary source [UNVERIFIED].

---

## 7. Items to verify before citing

- All Smart Material datasheet numbers, against the V2.4 PDF opened in full: capacitance 1.15 vs 1.3 nF (P1) and 30.78 vs 43 nF (P2); free-strain-per-volt ranges; low- and high-field d coefficients; E2 = 15.857 GPa. [PARTIAL-snippet]
- The CoreMorrow E01.A3 gain and bandwidth; the ECON MI-2004 range and noise; the PI E-501.00 module and gain (stated as ×10 in `testbed.py`); the MicroLabBox ADC resolution of the channels actually used. [PARTIAL or UNVERIFIED]
- The CFRP damping values from Kim 2021 (both Crystals papers) and Zhang 2020 *Shock Vib.*, from snippets only. Read Liu T. 2024 *MSSP* and Li H.-D. 2026 *MSSP* for per-mode ζ of CFRP plates with MFCs.
- Classical sources not seen in this session: Crawley & de Luis 1987; Dimitriadis, Fuller & Rogers 1991; Lee & Moon 1990; Kaipio & Somersalo 2005 (book); Colton & Kress; Van Overschee & De Moor 1994 DOI; Armenise et al. 2018 (SIPPY); ASME V&V 10 and Oberkampf & Roy. [UNVERIFIED]
- The slycot licence. [UNVERIFIED]

## 8. Files

- This synthesis: `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/research/lit_D_testbed.md`
- Downloaded wheels, inspected but not executed: `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/dl_wheels/`
