# Literature sweep B: FxLMS theory for tuning (mu, L) and the novelty of an analytical convergence map

Project: `/home/user/new-controller/sddmdc_fxlms` (base article: Ouyang, Dong, Liu, Zhou, *Thin-Walled Struct.* 228 (2026) 115090, doi:10.1016/j.tws.2026.115090)
Date of sweep: 2026-10-16. Language: English.

## How this sweep was done, and its limits

- Search tools: Consensus (semantic search, about 25 queries), SciSpace, Wiley Scholar Gateway (full-text passages, Wiley/IET/Hindawi content only) and WebSearch (snippets).
- **No full text could be opened.** The session's egress proxy refuses connections to publisher sites, arXiv, Crossref, OpenAlex and Semantic Scholar (403 / ENOTFOUND, logged in the proxy status). So most entries are checked at the level of their **bibliographic record and abstract**. A few are checked through quoted full-text passages (Scholar Gateway) or search snippets.
- Tag meanings:
  - **[VERIFIED]**: title, authors, year, venue and DOI were seen in a tool result during this session. Any content attributed to the work is limited to what its abstract or the quoted passage says.
  - **[UNVERIFIED]**: the record was not found, or the specific formula or claim was not seen in the source.
- Every formula in Section 1 was also **checked numerically** by a script written for this sweep (numpy only):
  - Script: `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/litB_tools/check_formulas.py`
  - Output: `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/litB_tools/check_formulas_out.txt`
- Notation:
  - Ouyang's update is W(k+1) = W(k) − 2μ v_s(k) r(k). The sign error in that pair is documented in audit S1. Here the correctly signed form W ← W + 2μ e r̂ is used, with the "2μ" convention.
  - A = amplitude of the reference sinusoid; ω = 2π f0/fs; fs = 10 kHz.
  - G = true secondary path at ω and Ĝ = its model (the "compensation path" G_c), with Δφ = arg Ĝ − arg G.
  - S(L, ω) = sin(Lω)/sin ω.

---

## 0. Executive summary

1. **Everything needed for a closed-form tonal (μ, L) theory is classical.** The theory rests on four results:
   - LMS/FxLMS with a sinusoidal reference is a linear time-invariant (LTI) system (Widrow 1975; Glover 1977; Elliott & Darlington 1985; Elliott, Stothers & Nelson 1987; Sievers & von Flotow 1992; Vicente & Masgrau 2002/2006).
   - The rank-2 eigenstructure of a sinusoidal autocorrelation matrix.
   - The delayed-LMS stability bound.
   - The ±90° phase-error rule (Boucher, Elliott & Nelson 1991; Snyder & Hansen 1994).
2. **The μ·L scaling is implicit in the classics but has not been used to collapse a 2-D (μ, L) map in active vibration control (AVC).**
   - In the classics it appears as the notch bandwidth, which depends on μ and the reference amplitude (Widrow 1975), and as NLMS normalisation, since ‖r‖² = L·A_r²/2.
   - Kuo et al. (1999, 2006) and Siswanto & Chang (2025, *JSV*) analyse how convergence depends on frequency and filter length in narrowband active noise control (ANC).
   - No paper was found that turns this into a closed-form prediction of the **whole (μ, L) convergence-time map**, or that tests it against measured AVC maps.
3. **Closest prior art for "predicting the effective step range from a model" — a novelty threat that must be cited:**
   - Li, Yang, Li, Brennan, Wu & Li (2024, *JSV* 568, 117953): pre-identified secondary path, then simulation-based prediction and tuning in AVC.
   - Zhang et al. (2022, *MSSP* 109831): an analytical interval of effective μ and an optimal μ for a multichannel delayed adaptive notch in vehicle ANC.
   - Ferrer et al. (2023, *Signal Process.*): a practical rule for choosing the convergence step.
   - Ouyang 2026 itself: simulation-based prediction.
4. **The pure-delay limit cannot explain Ouyang's data.**
   - For a tonal mode, β = μ L A² |Ĝ||G| cos Δφ / 2 must satisfy β < 2 sin(π/(2(2Δ+1))). The fastest achievable time constant is about Δ + 0.5 samples (checked numerically).
   - With Δ ≤ 100 samples (10 ms), the delay-limited 5 % convergence time is about 0.03 s. The observed t_min is 1.19–3.80 s.
   - So the active limit is the **lightly damped adjacent structural mode** (σ_n), as the audit (H4/M11) and the testbed (G4) found.
   - The idea that a structural resonance limits tonal LMS has precedent: Darlington & Elliott 1985 (IOA) and the Elliott et al. 1987 abstract, which says convergence takes "a timescale comparable to the response time of the system". No source found states the explicit **damping-budget law**: decay rate of the adaptation pole + decay rate of the adjacent mode = σ_n, independent of gain, so that α_div = σ_n and t_min ∝ 1/σ_n.
5. **Variable step-size (VSS) and normalised FxLMS (NFxLMS) do not remove tuning, they move it.**
   - NFxLMS replaces μ by a dimensionless α = μ‖r̂‖² ≈ μ L A_r²/2. This is exactly the collapse variable, so it removes the explicit L and amplitude dependence. But α_opt is still set by the plant (delay, phase error, σ_n).
   - VSS schemes add 2–5 hyperparameters (Ouyang's Table 1 counts 5 for its ref. [28]). Some papers then tune these with metaheuristics (Gao et al. 2021; Yao et al. 2021).
6. **Multichannel theory is classical.**
   - Modes follow the eigenvalues of Ĝᴴ G (Elliott et al. 1987, 1992; Boucher et al. 1991). Stability needs Re λ > 0 (Fraanje et al. 2007; Ferrer et al. 2023).
   - Ouyang's "two-channel" case is 2 actuators feeding 1 error sensor. Ĝᴴ G is then rank 1: one mode with gain Σᵢ Ĝᵢ* Gᵢ, plus one undamped null-space mode.
   - So equal (μ, L) on both channels **adds** the loop gains, and the region shrinks for that reason. "Cross-coupling" is not needed to explain it.

**Novelty verdict, in short:** an analytical (μ, L) map built only from classical eigenvalue and delay theory would be **incremental**. It becomes a defensible contribution only with three ingredients:
- (a) it is driven by a data-driven sd-DMDc secondary-path model;
- (b) it is validated quantitatively against the digitized **experimental** maps of the 2026 paper, with hypotheses declared before the runs;
- (c) it includes the **σ_n damping-budget law**, which explains the fast boundary and t_min. No closed form for that law was found in the literature this sweep could access.

The design rule that follows from the law is also new as a quantitative statement: added feedback damping should give t_min'/t_min = σ_n/σ_n'.

---

## 1. Key formulas, with sources and numerical checks

### F1. Eigenvalues of the (filtered-)reference autocorrelation matrix for one sinusoid and filter length L

For x(k) = A cos(ωk + θ), the matrix R_ij = (A²/2) cos(ω(i − j)) is L×L, symmetric Toeplitz, and of **rank 2**:

  λ± = (A²/4) · [ L ± sin(Lω)/sin(ω) ],  all other L − 2 eigenvalues = 0.

- For the filtered reference r = Ĝ * x, replace A by A|Ĝ(e^{jω})|.
- **Source:** Chassande-Mottin & Dhurandhar (2001; arXiv gr-qc/0003099) give the two non-zero eigenvalues as (N ± β)/2 with β = sin(Nδφ)/sin(δφ), in their normalisation. This was seen as a search snippet of that paper [VERIFIED snippet].
- Kuo, Tahernezhadi & Hao (1999) give bounds on λ_max and on the eigenvalue spread for narrowband direct-form FxLMS [VERIFIED abstract; the formula itself was not seen].
- Lovstedt et al. (2008, quoted passage) state that "the time constants for the modes of convergence … are determined by the eigenvalues of the autocorrelation matrix of the filtered-x signal" [VERIFIED passage].
- **Numerical check:** the computed and closed-form eigenvalues agree to 4 decimals in 7 cases (f0 ∈ {85, 120, 280} Hz, L ∈ {32, …, 128}). The third eigenvalue is below 1e-14. |S| = 8.84 at (120 Hz, L = 32), the same value as audit H3.
- **Consequence:** |S| ≤ 1/|sin ω|. At 120 Hz, 1/sin ω = 13.3, so at L = 128 the two eigenvalues differ from L·A²/4 by at most ±10 % (actually ±2.3 %, since |S| = 2.98). At L = 32 the split is ±28 %. This is the "small-L correction" to the μL law.

### F2. Mean convergence, time constants and the μ·L collapse (slow-adaptation limit)

- Mean recursion: E[W(k+1)] = (I − 2μ R̃) E[W(k)] + const, where R̃ = E[r̂(k) r_s(k)ᵀ]. Here r_s is the reference filtered by the true path G, and r̂ by the model Ĝ.
  - Stability with a perfect model: 0 < μ < 1/λ_max ("2μ" convention), quoted as Eqs. (26)–(28) in Yu, Cao & Li (2022) [VERIFIED passage].
  - Equivalently 0 < μ < 2/λ_max ("μ" convention), Eq. (2) in Lovstedt et al. (2008) [VERIFIED passage].
- Time constant of mode i: τ_i = 1/(2μ λ̃_i) samples. This is the standard LMS result, derived here.
- Tonal case with model error: the two dominant eigenvalues are λ̃± ≈ (A²|Ĝ||G|/4)(L ± S) cos Δφ.
  - Numerical check with |Ĝ| = |G| and L = 128: Re λ̃ / (L/4) = cos Δφ **exactly** for Δφ = 30°, 60°, 80°, 89°, and changes sign beyond 90°.
- **Convergence time to 5 % amplitude** (slow branch):

  t₅ ≈ ln 20 / (fs · 2μ λ̃) = 2 ln 20 / ( fs · μ (L ± S) · A² |Ĝ||G| cos Δφ ).

  This is the audit's H2 formula, confirmed on the paper's data (t·μ constant to ±2–6 %).
- **Collapse variable:** β ≡ μ L A² |Ĝ||G| cos Δφ / 2. Then t₅ depends on (μ, L) only through β, up to the L ± S split. Equivalently, the optimal step scales as μ_opt ∝ 1/L.
- **Precedents for the scaling:**
  - Widrow et al. (1975): with a periodic reference, the canceller "acts as a notch filter with narrow bandwidth … behaves as a linear, time-invariant system" [VERIFIED abstract].
    - A Fermilab report snippet says "the bandwidth of the filter is controlled by the noise reference amplitude C and the gain μ" [snippet; that report's citation is UNVERIFIED].
    - The textbook expression BW = μ N C²/T (N = number of weights) is **UNVERIFIED**: it was not seen in any tool result.
  - NLMS divides the update by the squared Euclidean norm of the tap-input vector (Lamb & Rouillard 2014, quoted passage [VERIFIED]).
    - For a sinusoid, ‖r‖² = L A_r²/2. Our numerical check gives ‖r‖² = 64.000, 15.999 and 32.001 for L = 128, 32 and 64 against L/2.
    - So a fixed NLMS step **is** a fixed β, for L ≫ |S|.

### F3. Maximum step size with a pure delay Δ in the secondary path

**(a) Broadband (white) reference** (Elliott & Nelson, quoted by Kajikawa, Gan & Kuo 2012):

  μ_max ≈ 2 / ( P_x′ (L + Δ) )   ("μ" convention), where P_x′ is the power of the filtered reference.

- Source: WebSearch summary of Kajikawa et al. (2012), doi:10.1017/atsip.2012.4 [VERIFIED snippet].
- A US patent snippet gives 1/(P_x′(L+Δ)), which is the same bound in the 2μ convention [snippet].
- The original derivation (attributed to Elliott & Nelson; probably J. Audio Eng. Soc. 37 (1989) 899–907) is **UNVERIFIED**.
- Ferrer et al. (2023) note that such bounds "only depend on the length of the adaptive filter, the power of the filtered reference signal, and the delay" and ignore the spectrum of the reference and the frequency response of the path [VERIFIED abstract/snippet].

**(b) Tonal reference** (the relevant case here):
- Each complex mode of the averaged LTI equivalent obeys the delayed-LMS scalar equation v(k+1) = v(k) − β v(k − Δ).
- The LTI equivalence itself is from Glover 1977, Elliott et al. 1987 and Vicente & Masgrau 2002/2006 [VERIFIED abstracts]. The scalar reduction is derived here.
- Its exact stability bound and fastest gain are:

  β_max = 2 sin( π / (2(2Δ+1)) )  ≈ π/(4Δ+2) for large Δ,
  β_opt = Δ^Δ / (Δ+1)^(Δ+1) ≈ 1/(e(Δ+1)),  τ_min ≈ Δ + 0.5 samples.

- Numerical check: bisection on the characteristic roots matches β_max to 6 decimals for Δ = 1…100; β_opt matches to 1e-5; τ_min = 1.44, 10.49 and 100.53 samples for Δ = 1, 10 and 100.
- Attribution of the closed form: Long, Ling & Proakis (1989; corrected 1992) derive "an upper bound for the step size that ensures the stability of the DLMS" [VERIFIED abstract]. Whether they state exactly 2 sin(π/(2(2Δ+1))) is **UNVERIFIED**.
- Ernst et al. (1995) give the exact bound as the roots of an order-(D+1) polynomial, which is the same characteristic equation [VERIFIED abstract].
- Vicente & Masgrau (2006) and Wen et al. (2021) give frequency-dependent μ_max for narrowband FxLMS with a pure-delay path [VERIFIED abstracts; formulas not seen].
- **Implication for Ouyang 2026:** Δ ≤ 100 samples gives τ_min ≤ 100.5 samples, i.e. t₅,min ≈ ln 20 × 10 ms ≈ 0.03 s. The observed t_min is 3.80, 2.90 and 1.19 s (85, 120, 280 Hz), 40–130 times slower. Delay is not the binding constraint, which agrees with audit M11.

### F4. Phase-error tolerance of the secondary-path model, and its link to stability

- **Slow adaptation (Δ → 0):** stability requires Re λ(R̃) > 0. For a single tone this means |Δφ(ω0)| < 90°, and the rate scales as cos Δφ. This follows from F2 and the numerical check.
  - Snyder & Hansen (1994): "a maximum phase error of ±90° is a bound for stability, [but] there is no simple relationship between error and stability within this region" [VERIFIED abstract].
  - Boucher, Elliott & Nelson (1991): "the effects of phase errors on the maximum update coefficient and convergence time of the filtered-x LMS algorithm … analytic results and computer simulations" [VERIFIED abstract].
  - Lovstedt et al. (2008): the algorithm "will converge (slowly) as long as phase errors are less than 90° … and phase errors less than 40° do not significantly affect convergence" [VERIFIED passage; its ref. [9] is not identified].
- **Phase error combined with delay** (derived and checked here, for a scalar delayed complex mode v(k+1) = v(k) − β e^{−jΔφ} v(k−Δ)):

  β_max(Δ, Δφ) = 2 sin( (π/2 − |Δφ|) / (2Δ+1) ).

  - This matches the numerical bisection to 4 decimals on a 5 × 6 grid (Δ ∈ {0, 1, 5, 20, 100}, Δφ ∈ {0°, …, 89°}).
  - For Δ = 0 it reduces to 2 cos Δφ.
  - So a large step size and a large delay both **shrink the usable phase-error margin**. At Δ = 20, a 60° error cuts β_max by a factor of 3.0 (0.0766 → 0.0255).
  - Whether this closed form appears in Boucher et al. (1991) or Elliott (2001) is **UNVERIFIED**.
- **General paths and long filters:**
  - Fraanje, Verhaegen & Elliott (2007, Part I): divergence occurs "when the eigenvalues of the cross-correlation matrix between the estimated filtered reference and the true filtered reference signals are not all located in the right half plane". The matrix is Toeplitz, and a frequency-domain condition is "sufficient and necessary for a large number of filter coefficients" [VERIFIED abstract].
  - Wang & Ren (1999): a strictly-positive-real (SPR) condition on the model error is sufficient [VERIFIED abstract].
  - Elliott (1998): the bound is expressed as a limit on unstructured multiplicative uncertainty [VERIFIED abstract].

### F5. Multichannel tonal FxLMS (K actuators, M sensors)

- In complex (phasor) form at ω0: u(k+1) = u(k) − α Ĝᴴ e(k), with e = d + G u.
- Modes lie along the eigenvectors of Ĝᴴ G. With no delay, each decays as (1 − α λ_m). Stability needs Re λ_m > 0 and α < 2 Re λ_m / |λ_m|².
- Small eigenvalues are slow modes. When Ĝ = G (Hermitian product), those modes also need large control effort.
- These per-mode formulas are derived here. The qualitative statements come from:
  - Elliott, Boucher & Nelson (1992): "some modes not only converge slowly but also require an excessive control effort … controlled by the proper choice of the cost function" [VERIFIED abstract].
  - Elliott, Stothers & Nelson (1987): with synchronous sampling, "the behavior … can be completely described by a matrix of linear, time invariant, transfer functions" [VERIFIED abstract].
- **Ouyang's 2-actuator / 1-sensor case** (derived here):
  - Ĝᴴ G = Ĝᴴ(1×2)ᴴ G(1×2) is rank 1.
  - It has λ₁ = Σᵢ Ĝᵢ* Gᵢ (the per-channel loop gains add) and λ₂ = 0, an undamped null-space direction. Without leakage this direction drifts under noise.
  - With μ₁ = μ₂ and L₁ = L₂, the effective β is the **sum** of the two single-channel β. This explains the narrower region and earlier divergence (audit M6/M7).

---

## 2. Annotated bibliography

### A. Foundations: LMS/FxLMS with a sinusoidal reference behaves as an LTI system

1. **B. Widrow, J.R. Glover Jr., et al., "Adaptive noise cancelling: Principles and applications", Proc. IEEE 63(12) (1975) 1692–1716.** doi:10.1109/proc.1975.10036 [VERIFIED; page range UNVERIFIED]
   - With a periodic reference, the canceller is an LTI notch filter that converges on a dynamic, not static, solution.
   - Relevance: the origin of "tonal LMS = LTI system" and of a bandwidth set by μ and the reference amplitude. The exact BW formula is [UNVERIFIED].
2. **J.R. Glover, "Adaptive noise canceling applied to sinusoidal interferences", IEEE Trans. ASSP 25(6) (1977) 484–491.** doi:10.1109/tassp.1977.1162997 [VERIFIED; page range UNVERIFIED]
   - A sum of sinusoids gives time-varying weights that implement a tunable notch.
   - The arXiv survey 2110.00531 credits Glover with modelling the adaptation as an LTI transfer function and using a root locus to obtain the upper step bound [snippet].
   - Relevance: the root-locus/LTI route to μ_max that our "averaged LTI equivalent" (testbed G2/G3) follows.
3. **S.J. Elliott, P. Darlington, "Adaptive cancellation of periodic, synchronously sampled interference", IEEE Trans. ASSP 33 (1985).** doi:10.1109/tassp.1985.1164596 [VERIFIED; issue/pages UNVERIFIED]
   - Synchronous sampling makes the LMS exactly an LTI comb filter.
4. **P. Darlington, S.J. Elliott, "Adaptive control of periodic disturbances in resonant systems", Proc. IOA Spring Conference 1985.** doi:10.25144/22466 [VERIFIED record]
   - Abstract: control of "a 2nd order network … a stability limit exists which is explained in terms of an analytical model of the adaptive system".
   - **Relevance: high.** This is the earliest precedent found for "a structural resonance limits tonal LMS stability", the mechanism behind σ_n. The full text was not accessible, so whether it contains a damping-budget result is [UNVERIFIED].
   - The related thesis, P. Darlington, "Applications of adaptive filters in active noise control" (1987), derives "Equivalent Transfer Function descriptors … exact specification of the stability limits imposed by the control loop filter" [VERIFIED abstract; no DOI].
5. **S.J. Elliott, I.M. Stothers, P.A. Nelson, "A multiple error LMS algorithm and its application to the active control of sound and vibration", IEEE Trans. ASSP 35(10) (1987) 1423–1434.** doi:10.1109/tassp.1987.1165044 [VERIFIED; issue/pages UNVERIFIED]
   - Abstract: the algorithm "converges on a timescale comparable to the response time of the system to be controlled"; with a synchronously sampled tone it is a matrix of LTI transfer functions.
   - **Relevance: high.** It is the multichannel base, and the "response time" remark anticipates the t_min ∝ 1/σ_n regularity.
6. **L. Sievers, A. von Flotow, "Comparison and extensions of control methods for narrow-band disturbance rejection", IEEE Trans. Signal Process. 40 (1992) 2377–2391.** doi:10.1109/78.157283 [VERIFIED; volume/pages from De Nicolao 1994 abstract]
   - FxLMS "is best analyzed from a classical linear time invariant feedback perspective". Stability bounds are given in terms of allowable model error.
   - Relevance: justifies Nyquist/root-locus analysis of the tonal loop with the identified G_s(z), and connects to the user's robust-control (μ-analysis) skills.
7. **W. Messner, M. Bodson, "Design of adaptive feedforward algorithms using internal model equivalence", Int. J. Adapt. Control Signal Process. 9(2) (1995) 199–212.** doi:10.1002/acs.4480090207 [VERIFIED; pages UNVERIFIED]
   - Adaptive feedforward cancellation (AFC) with sinusoidal regressors is equivalent to an internal-model-principle controller; the root locus is used to pick the regressor phase advance.
   - Related: M. Bodson (2005), Int. J. Adapt. Control Signal Process. 19, doi:10.1002/acs.844 (review); X. Guo, M. Bodson (2009), doi:10.1002/acs.1117 [all VERIFIED].
   - Relevance: the control-theory language for the "adaptation pole + structural pole" picture.
8. **D.R. Morgan, C. Sanford, "A control theory approach to the stability and transient analysis of the filtered-x LMS adaptive notch filter", IEEE Trans. Signal Process. (1992)** [UNVERIFIED]
   - Cited by a 2014 IFAC paper, according to a WebSearch result. No record or DOI was seen.
   - Relevance: if confirmed, this is the direct precedent for transient (convergence-time) analysis of tonal FxLMS by control theory. It must be checked before submission.
9. **L. Vicente, E. Masgrau, "Analysis of LMS algorithm with delayed coefficient adaptation for sinusoidal reference", EUSIPCO 2002.** doi:10.5281/zenodo.38065 [VERIFIED]
   - The system is exactly LTI for two in-phase/quadrature weights **and for a tapped delay line with a large number of weights**; an upper step bound is derived.
   - Relevance: justifies the exact 2-state reduction used in the testbed (G1, matched to 7e-13 V).
10. **L. Vicente, E. Masgrau, "Novel FxLMS convergence condition with deterministic reference", IEEE Trans. Signal Process. 54(10) (2006) 3768–3774.** doi:10.1109/tsp.2006.880205 [VERIFIED; volume/pages UNVERIFIED]
    - Upper step bound for sinusoidal and multi-sinusoidal references, derived without independence or slow-convergence assumptions.
    - According to the arXiv survey snippet it assumes a pure-delay path; Xiao et al. later extended it to FIR paths.
11. **Y. Hinamoto et al., "Analysis of the filtered-X LMS algorithm and a related new algorithm for active control of multitonal noise", IEEE Trans. ASLP 14 (2006).** doi:10.1109/tsa.2005.854089 [VERIFIED; volume UNVERIFIED]
    - "The algorithm is derived under the assumption of slow adaptation limit … exact analysis restricted to the case of one real sinusoid." Derives the characteristic polynomial of the equivalent linear system for any number of tones, and a stability limit.
12. **H.-X. Wen et al., "An analysis of the convergence condition for narrowband FxLMS", Electron. Lett. (2021/2022).** doi:10.1049/ell2.12378 [VERIFIED]
    - Also: Wen et al., ICSPCC 2021, doi:10.1109/icspcc52875.2021.9564582 [VERIFIED].
    - μ_max is estimated "as a function of the reference frequency" from root-locus critical points, for pure-delay and general paths.
13. **E. Chassande-Mottin, S.V. Dhurandhar, "Adaptive filtering techniques for gravitational wave interferometric data …", Phys. Rev. D 63 (2001) 042004; arXiv:gr-qc/0003099.** [VERIFIED via WebSearch]
    - Source for the rank-2 eigenvalues (N ± β)/2 with β = sin(Nδφ)/sin δφ (F1). It is outside the active-control literature, so worth citing for the closed form.

### B. Step-size bounds, delay, and analyses of convergence speed

14. **G. Long, F. Ling, J.G. Proakis, "The LMS algorithm with delayed coefficient adaptation", IEEE Trans. ASSP 37(9) (1989) 1397–1405.** doi:10.1109/29.31293 [VERIFIED]
    - Corrections: IEEE Trans. Signal Process. 40(1) (1992) 230–232, doi:10.1109/78.157202 [VERIFIED].
    - Upper step bound for delayed LMS (DLMS), plus the relation between step size, delay and convergence speed. The source formula behind F3(b) is [UNVERIFIED].
15. **T. Ernst et al., "Analysis of the LMS algorithm with delayed coefficient update", ISCAS 1995.** doi:10.1109/iscas.1995.520371 [VERIFIED]
    - The bound is given by the roots of a polynomial of order D+1, which is our F3(b) characteristic equation.
16. **P.L. Feintuch, N.J. Bershad, A.K. Lo, "A frequency domain model for 'filtered' LMS algorithms — stability analysis, design, and elimination of the training mode", IEEE Trans. Signal Process. 41(4) (1993) 1518–1531.** doi:10.1109/78.212728 [VERIFIED; pages UNVERIFIED]
    - Stability as a function of step size, input power and the loop transfer functions. For a band-limited input, a single delay can replace the full path model.
17. **E. Bjarnason, "Analysis of the filtered-X LMS algorithm", ICASSP 1993.** doi:10.1109/icassp.1993.319547 [VERIFIED]
    - Journal version (IEEE Trans. Speech Audio Process. 3(6) (1995) 504–514) [UNVERIFIED record]: a Consensus entry without venue, abstract "bounds and predicted dynamic behavior … correspond very well to simulation"; also covers off-line and on-line path estimation.
    - Relevance: classical stochastic analysis, but for spherically invariant (random) inputs, not tones.
18. **S.M. Kuo, D.R. Morgan, Active Noise Control Systems: Algorithms and DSP Implementations, Wiley (1996)**, and **"Active noise control: a tutorial review", Proc. IEEE 87(6) (1999) 943–973**, doi:10.1109/5.763310 [both VERIFIED; issue/pages UNVERIFIED]
    - Standard references for the FxLMS step bound with delay, narrowband ANC and normalised FxLMS. The exact equations are [UNVERIFIED] (the PDF was not reachable).
19. **Y. Kajikawa, W.-S. Gan, S.M. Kuo, "Recent advances on active noise control: open issues and innovative applications", APSIPA Trans. Signal Inf. Process. (2012).** doi:10.1017/atsip.2012.4 [VERIFIED; volume UNVERIFIED]
    - Quoted (search snippet) for μ_max = 2/(P_x′(L+Δ)), attributed to Elliott & Nelson (F3a).
20. **S.J. Elliott, P.A. Nelson, "Multiple-point equalization in a room using adaptive digital filters", J. Audio Eng. Soc. 37(11) (1989) 899–907.** [VERIFIED record; no DOI seen]
    - Probable origin of the white-noise delay bound [UNVERIFIED].
21. **O.J. Tobias, J.C.M. Bermudez, N.J. Bershad, "Mean weight behavior of the filtered-X LMS algorithm", ICASSP 1998.** doi:10.1109/icassp.1998.679637 [VERIFIED]
    - Also: "Second moment analysis of the filtered-X LMS algorithm", ICASSP 1999, doi:10.1109/icassp.1999.758288 [VERIFIED].
    - Mean weights for slow adaptation without independence theory. The steady state equals the Wiener solution only with an exact path model.
    - The journal version (IEEE TSP 2000) is [UNVERIFIED]. Relevance: stochastic (white Gaussian) theory, not tonal.
22. **I.T. Ardekani, W.H. Abdulla, "Theoretical convergence analysis of FxLMS algorithm", Signal Process. 90(12) (2010) 3046–3055.** doi:10.1016/j.sigpro.2010.05.009 [VERIFIED; pages UNVERIFIED]
    - Notes that earlier conditions "are only accurate for simplified cases with pure delay secondary paths or multi-sinusoidal input signals". Gives a necessary and sufficient condition and the fastest-convergence condition for moving-average paths with stochastic inputs.
    - Companion papers:
      - "On the convergence of real-time active noise control systems", Signal Process. 91 (2011), doi:10.1016/j.sigpro.2010.12.012 [VERIFIED]: step bound and fastest step for band-limited white noise.
      - "Effects of imperfect secondary path modeling on adaptive active noise control systems", IEEE Trans. Control Syst. Technol. 20(5) (2012), doi:10.1109/tcst.2011.2161762 [VERIFIED]: "intentional misadjustment of secondary path models can enhance performance".
      - "Stochastic modelling and analysis of filtered-x least-mean-square adaptation algorithm", IET Signal Process. 7(6) (2013) 486–496, doi:10.1049/iet-spr.2012.0090 [VERIFIED].
    - Relevance: optimal-step formulas exist, but **for stochastic inputs**.
23. **F.-R. Yang et al., "Stochastic analysis of the filtered-x LMS algorithm for active noise control", IEEE/ACM Trans. ASLP 28 (2020).** doi:10.1109/taslp.2020.3012056 [VERIFIED]
    - Mean and mean-square stability bounds for an arbitrary reference and a general path.
24. **S.M. Kuo, M. Tahernezhadi, W. Hao, "Convergence analysis of narrow-band active noise control system", IEEE Trans. Circuits Syst. II 46(2) (1999).** doi:10.1109/82.752958 [VERIFIED]
    - Bounds on λ_max and the eigenvalue spread for narrowband direct-form FxLMS. Sets reference amplitudes to the **inverse of |S(ω)|** to equalise convergence.
    - Relevance: amplitude normalisation equals the β normalisation of F2.
25. **S.M. Kuo, A.B. Puvvala, "Effects of frequency separation in periodic active noise control systems", IEEE Trans. ASLP 14 (2006).** doi:10.1109/TSA.2005.858524 [VERIFIED]
    - Also: Kuo, Puvvala & Gan, ICASSP 2006, doi:10.1109/ICASSP.2006.1661270 [VERIFIED].
    - "The length of the adaptive filter required for achieving the same convergence speed decreases with an increase in frequency separation."
    - Relevance: **an L-dependence of convergence speed derived analytically** (multi-tone ANC).
26. **A. Siswanto, C.-Y. Chang, "Frequency dependent convergence rate of FXLMS in narrowband active noise control systems", J. Sound Vib. 604 (2025) 118988.** doi:10.1016/j.jsv.2025.118988 [VERIFIED; abstract via SciSpace]
    - "Convergence rates for the given filter lengths are frequency dependent"; conditions for fast convergence; **simulation-only**, ANC.
    - **Relevance: the closest recent analytical work.** It is Ouyang's own ref. [31], cited there only in passing (audit M15). It must be cited and contrasted: no (μ, L) map, no structural plant, no experimental validation.
27. **J.K. Thomas, S.P. Lovstedt, J.D. Blotter, S.D. Sommerfeldt, "Eigenvalue equalization filtered-x algorithm …", J. Acoust. Soc. Am. 123 (2008).** doi:10.1121/1.2903857 [VERIFIED; author list partly UNVERIFIED]
    - Also: Lovstedt et al., "Genetic algorithm applied to the EE-FXLMS", Adv. Acoust. Vib. 2008, doi:10.1155/2008/791050 [VERIFIED passage].
    - FxLMS "exhibits frequency dependent convergence behavior". Flattening |Ĥ| while keeping its phase equalises the eigenvalues; experiments show 3.5–4.4 dB extra attenuation.
28. **A. Puri, S.V. Modak, K. Gupta, "A variable step-size filtered-x least mean square algorithm for continuously varying noise", Noise Control Eng. J. 64 (2016).** doi:10.3397/1/376394 [VERIFIED]
    - "The optimum value of the convergence coefficient is decided based upon the power and delay of the filtered reference signal and sampling frequency."
    - Relevance: a closed-form optimal μ from power + delay already exists (ANC, simulation).
29. **M. Ferrer et al., "Selection of the convergence step of the Fx-LMS algorithm", Signal Process. (2023) 109118.** doi:10.1016/j.sigpro.2023.109118 [VERIFIED]
    - Step choice "is usually carried out by trial and error, since the theoretical developments provide only indicative values". Proposes a low-cost selection method that is independent of reference type, delays and path.
    - **Relevance: a direct novelty threat for any "tuning rule"** claim; it must be cited.
30. **S. Zhang et al., "Active control of vehicle interior engine noise using a multi-channel delayed adaptive notch algorithm based on FxLMS structure", Mech. Syst. Signal Process. (2022) 109831.** doi:10.1016/j.ymssp.2022.109831 [VERIFIED]
    - Convergence condition plus models of steady-state error and **convergence speed**, matching simulation. "They help to theoretically indicate the interval range of the effective convergence coefficient μ and its optimal value". Vehicle experiments.
    - **Relevance: the strongest analytical-tuning precedent.** But it is ANC, uses a two-weight notch (no L dimension) and an acoustic plant (no lightly damped structural mode).
31. **Y. Xiao et al., "Stochastic analysis of the FXLMS-based narrowband active noise control system", IEEE Trans. ASLP 16 (2008).** doi:10.1109/tasl.2008.921756 [VERIFIED]
    - Also: Y. Xiao (2011), doi:10.1109/tasl.2010.2101598; Xiao et al. (2009), IEEE TSP, doi:10.1109/tsp.2009.2020766 [VERIFIED].
    - Closed-form mean/mean-square dynamics and a mean stability bound for discrete-Fourier-coefficient (two-weight) narrowband FxLMS.
32. **L. Lu et al., "A survey on active noise control in the past decade — Part I: Linear systems", Signal Process. 183 (2021) 108039.** doi:10.1016/j.sigpro.2021.108039 [VERIFIED; volume UNVERIFIED]
    - arXiv version 2110.00531: its snippet gives the lineage Glover → Elliott (multichannel narrowband) → Vicente & Masgrau → Xiao.
33. **D. Bismor, "Extension of LMS stability condition over a wide set of signals", Int. J. Adapt. Control Signal Process. 29(5) (2015; online 2014) 653–670.** doi:10.1002/acs.2500 [VERIFIED passage]
    - Small-step-size theory can be too restrictive or too loose for deterministic signals.
    - Relevance: a caution that eigenvalue bounds are not exact for tones; use the exact LTI equivalent.
34. **H.-J. Jeon, T.-G. Chang, S.M. Kuo, "Analysis of frequency mismatch in narrowband active noise control", IEEE Trans. ASLP 18 (2010).** doi:10.1109/tasl.2009.2038815 [VERIFIED]
    - Stability conditions in terms of frequency mismatch and phase error; optimum step for minimum excess MSE.

### C. Plant-model (phase) error and robustness

35. **C.C. Boucher, S.J. Elliott, P.A. Nelson, "Effect of errors in the plant model on the performance of algorithms for adaptive feedforward control", IEE Proc. F 138(4) (1991) 313–319.** doi:10.1049/ip-f-2.1991.0042 [VERIFIED]
    - Single channel: effect of phase errors on the maximum update coefficient and on the convergence time (analytic + simulation).
    - Multichannel: statistics with Gaussian-distributed errors; 32-microphone / 16-loudspeaker experiment.
    - **Relevance: high** (F4 and the FEM-versus-sd-DMDc question).
36. **S.D. Snyder, C.H. Hansen, "The effect of transfer function estimation errors on the filtered-x LMS algorithm", IEEE Trans. Signal Process. 42(4) (1994) 950–953.** doi:10.1109/78.285659 [VERIFIED; issue/pages UNVERIFIED]
    - ±90° is the stability bound, with "no simple relationship … within this region".
37. **R. Fraanje, M. Verhaegen, S.J. Elliott, "Robustness of the filtered-X LMS algorithm — Part I", IEEE Trans. Signal Process. 55 (2007).** doi:10.1109/tsp.2007.896083 [VERIFIED]
    - Also Part II (Elliott, Fraanje, Verhaegen), doi:10.1109/tsp.2007.896086 [VERIFIED].
    - Eigenvalues of the Toeplitz estimated-versus-true filtered-reference cross-correlation must lie in the right half plane. The frequency-domain condition is necessary and sufficient as L → ∞. Part II gives regularisation for norm-bounded uncertainty.
    - Relevance: the rigorous L-dependent version of the ±90° rule, and a robust-control angle for the user.
38. **A.K. Wang, W. Ren, "Convergence analysis of the multi-variable filtered-X LMS algorithm with application to active noise control", IEEE Trans. Signal Process. 47 (1999).** doi:10.1109/78.752618 [VERIFIED]
    - A multivariable SPR condition is sufficient for narrowband and broadband disturbances.
39. **S.J. Elliott, "Filtered reference and filtered error LMS algorithms for adaptive feedforward control", Mech. Syst. Signal Process. 12 (1998).** doi:10.1006/mssp.1998.0178 [VERIFIED]
    - Plant-model accuracy is expressed as a bound on unstructured multiplicative uncertainty.
40. **F.-Y. An, H. Sun, X. Li, "Novel convergence analysis of narrowband FXLMS-based algorithm", IET conf. (2012).** doi:10.1049/cp.2012.1248 [VERIFIED]
    - With a constant step, the speed is **maximised at a critical phase error** (narrowband noise/vibration).
41. **A. Omoto, S.J. Elliott, "The effect of structured uncertainty in the acoustic plant on multichannel feedforward control systems", IEEE Trans. Speech Audio Process. 7 (1999).** doi:10.1109/89.748125 [VERIFIED]
    - Tone at 88 Hz, 32 sensors × 16 sources; the SVD of the transfer matrix explains which uncertainties destabilise.

### D. Multichannel FxLMS convergence

42. **S.J. Elliott, C.C. Boucher, P.A. Nelson, "The behavior of a multiple channel active control system", IEEE Trans. Signal Process. 40(5) (1992) 1041–1052.** doi:10.1109/78.134467 [VERIFIED; pages UNVERIFIED]
    - Steepest descent on squared error plus effort; slow, effort-hungry ill-conditioned modes; prediction matches a 16 × 32 experiment.
    - **Relevance: high** for F5.
43. **S.D. Snyder, C.H. Hansen, "Convergence characteristics of the multiple input, multiple output LMS algorithm", J. Intell. Mater. Syst. Struct. 3 (1992).** doi:10.1177/1045389x9200300107 [VERIFIED]
    - The ratio of error sensors to control sources, the delays and the transfer functions all shape convergence.
44. **T. Sone et al., J. Sound Vib. (1996).** doi:10.1006/jsvi.1996.0417 [VERIFIED]
    - Per-frequency-bin convergence of multiple-error FxLMS; speed depends on the input spectra **and on the multiple secondary paths**.
45. **R.H. Cabell, "A principal component algorithm for feedforward active noise and vibration control" (1998).** [VERIFIED abstract; no DOI]
    - Rotation onto plant principal components at the tone allows **independent convergence factors per coordinate**.
    - Relevance: the principled answer to the "4-D (μ1, μ2, L1, L2) optimisation" the 2026 paper proposes as future work.
46. **C. Gong et al., "Statistical analysis of multichannel FxLMS algorithm for narrowband active noise control", Signal Process. (2022) 108646.** doi:10.1016/j.sigpro.2022.108646 [VERIFIED]
    - Mean and mean-square stability boundaries; coupling between cross secondary paths handled by vectorisation.
47. **M. Ferrer et al., "Assessment of stability of distributed FxLMS active noise control systems", Signal Process. (2023) 109087.** doi:10.1016/j.sigpro.2023.109087 [VERIFIED]
    - Stable if and only if all eigenvalues of the system matrix have positive real part; a fast test for this.
48. **G.-Y. Chen et al., J. Sound Vib. (2008).** doi:10.1016/j.jsv.2007.09.044 [VERIFIED]
    - Closed-form eigenvalues of the spectral-density matrix are used **to fix the step size**.
49. **S.J. Elliott, "Optimal controllers and adaptive controllers for multichannel feedforward control of stochastic disturbances", IEEE Trans. Signal Process. 48 (2000).** doi:10.1109/78.827539 [VERIFIED]
    - Convergence rate is limited by reference (cross-)correlations and by plant dynamics and coupling.

### E. Normalised and variable-step FxLMS (and variable filter length) for AVC, 2018–2026: do they remove μ tuning?

| # | Reference | DOI | What it does | Tuning still needed? |
|---|---|---|---|---|
| 50 | Y.-B. Fang et al., J. Low Freq. Noise Vib. Act. Control (2018) | 10.1177/1461348418812326 [VERIFIED] | VSS-LMS (Kwong type), piezo cantilever beam | yes (VSS constants) |
| 51 | W. Niu et al., MSSP 118 (2019) 93–107 | 10.1016/j.ymssp.2018.08.009 [VERIFIED] | bang-bang VSS plus online secondary-path modelling (SPM); Ouyang ref. [24] | yes (switching levels) |
| 52 | Y.-X. Pu et al., MSSP (2019) | 10.1016/j.ymssp.2018.10.019 [VERIFIED] | multichannel FxLMS with per-filter VSS and online SPM, piezo plate | yes |
| 53 | W. Li et al., Shock Vib. (2020) | 10.1155/2020/8832467 [VERIFIED] | VSS driven by the error derivative (DVSFxLMS), piezo stack beam | yes |
| 54 | W. Li et al., JSV 509 (2021) 116243 | 10.1016/j.jsv.2021.116243 [VERIFIED] | hybrid PID-FxLMS; Ouyang ref. [25], 5 parameters | yes (5) |
| 55 | E. Batista et al., IEEE/ASME Trans. Mechatron. (2021) | 10.1109/tmech.2021.3104307 [VERIFIED] | FxNLMS scheme against "vibrations introduced by the adaptive control algorithm itself … due to resonance phenomena" | yes; **relevant to σ_n** |
| 56 | Z.-Y. Gao et al., J. Vib. Eng. Technol. (2021) | 10.1007/s42417-021-00417-4 [VERIFIED] | variable step **and variable tap-length**; parameters tuned by DE-Jaya | tuning moved to a metaheuristic |
| 57 | D.-C. Chang, F.-T. Chu, IEEE SPL (2013) / IEEE/ACM TASLP (2014) | 10.1109/lsp.2013.2282396; 10.1109/taslp.2013.2297016 [VERIFIED] | variable tap-length and step-size FxLMS (ANC) | yes; L treated as adaptive |
| 58 | J. Jiang et al., J. Vib. Control (2024) | 10.1177/10775463241260110 [VERIFIED] | sech-VSS FxLMS, blade | yes |
| 59 | T. Liu, C. Liu, Z. Zhang, MSSP 206 (2024) 110876 | 10.1016/j.ymssp.2023.110876 [VERIFIED] | FxLMS on a composite plate with MFC (Ouyang ref. [26]) | fixed μ |
| 60 | X.-H. Chen et al., MSSP 224 (2025) 111959 | 10.1016/j.ymssp.2024.111959 [VERIFIED] | Fx-NLMS on a rotating composite beam; "filter orders and learning factor … studied" | yes (α, L) |
| 61 | J. Jiang et al., JLFNVAC (2026) | 10.1177/14613484261470208 [VERIFIED] | normalised tanh-VSS (VSS-FxtanhLMS), beam, mean/mean-square analysis | yes |
| 62 | X. Jiang et al., Meas. Control (2026) | 10.1177/00202940261436521 [VERIFIED] | convex combination of a VSS agent and a fixed small-step agent, game-theoretic weighting | yes |
| 63 | B.-Y. Huang et al., IEEE TASLP 21 (2013) | 10.1109/tasl.2012.2223673 [VERIFIED] | VSS-FXLMS for narrowband ANC, converging near FxRLS speed | yes |
| 64 | Y.-P. Ma et al., Signal Process. (2021) 108012 | 10.1016/j.sigpro.2021.108012 [VERIFIED] | simplified VSS-FXLMS for narrowband ANC with closed-form statistics | yes |

- Also: J. Yao et al., Appl. Acoust. (2021), doi:10.1016/j.apacoust.2021.108285 [VERIFIED]: VSS combined with modified normalised FxLMS, with parameters set by improved PSO.
- **Conclusion for Q4:** no paper in this set removes the need to tune. NLMS normalises out amplitude and L, since ‖r̂‖² ≈ L A_r²/2, but leaves α. VSS schemes add hyperparameters. One paper reduces the tuning load by normalising its forgetting factor (Jiang et al. 2025, CCSSTA, doi:10.1109/ieeeconf65522.2025.11137087 [VERIFIED]).

### F. AVC-specific prediction and tuning (direct context of the project)

65. **F. Li, T. Yang, W. Li, M.J. Brennan, L. Wu, X. Li, "Predicting the performance and optimizing the control parameters of an active vibration control system by pre-identifying the secondary path", J. Sound Vib. 568 (2024) 117953.** doi:10.1016/j.jsv.2023.117953 [VERIFIED]
    - The secondary path is split into three parts: actuator FRF; structure identified by pneumatic impact; ADC/DAC as a half-sample phase shift. "Combined with the simulation of the control system" it predicts performance and **optimises control parameters** without installing the AVC hardware.
    - **Relevance: the main AVC precedent** for "offline model → parameter prediction". It is numerical simulation, not analytical. Ouyang cites it only as a VSS example (ref. [28]; audit M15).
66. **Q. Ouyang et al. (co-authors UNVERIFIED), "Identification of electromechanical participation matrices via scaled-derivative DMDc for improved PPF active vibration control of thin-walled structures", Eng. Struct. (2026) 122232.** doi:10.1016/j.engstruct.2026.122232 [VERIFIED record; abstract not available]
    - Same group, 3 citations by October 2026. Relevance: the group is already pairing sd-DMDc with **PPF feedback**, which bears on the planned hybrid PPF + FxLMS kernel and must be checked by sweep A.
67. **Q. Ouyang et al., Thin-Walled Struct. 228 (2026) 115090.** doi:10.1016/j.tws.2026.115090 [VERIFIED]
    - Consensus shows 0 citations, and no follow-up was found (searched for "DMDc FxLMS", 2025–2026).
68. **K.B. Scribner, L.A. Sievers, A.H. von Flotow, "Active narrow-band vibration isolation of machinery noise from resonant substructures", J. Sound Vib. (1993).** doi:10.1006/jsvi.1993.1319 [VERIFIED]
    - Safely achievable performance is limited by "the level of passive damping present in the unmodeled resonant response" (modal overlap), by the pole-zero structure near the disturbance frequency, and by the disturbance bandwidth. This is about feedback.
    - **Relevance:** a qualitative precedent for "damping limits narrowband control". Our σ_n law is the adaptive-feedforward, quantitative version.
69. **J.S. Vipperman, R.A. Burdisso, C.R. Fuller, "Active control of broadband structural vibration using the LMS adaptive algorithm", J. Sound Vib. 166 (1993).** doi:10.1006/jsvi.1993.1297 [VERIFIED]
    - Classical AVC FxLMS with IIR path models on a beam.
70. **Q. Lu, P. Wang, C. Liu, Int. J. Mech. Sci. 232 (2022) 107634.** doi:10.1016/j.ijmecsci.2022.107634 [VERIFIED]
    - MFC + FxLMS on a sandwich beam (Ouyang ref. [21]).

### G. Books (standard references; formulas not seen)

- **S.J. Elliott, Signal Processing for Active Control, Academic Press (2001), ISBN 0-12-237085-6** [VERIFIED via book review doi:10.1109/mcs.2002.993318]. Ouyang ref. [56] (dated 2000 there). Ch. 3: single-channel feedforward for deterministic disturbances; Ch. 4: multichannel tonal control. Exact equations [UNVERIFIED].
- **P.A. Nelson, S.J. Elliott, Active Control of Sound, Academic Press (1992)** [VERIFIED record].
- **C.R. Fuller, S.J. Elliott, P.A. Nelson, Active Control of Vibration, Academic Press (1996)** [VERIFIED record].
- **S.M. Kuo, D.R. Morgan (1996)**, listed above.

### Not found

- "Wu, Zhou et al." (named in the task): no matching FxLMS-convergence paper was identified [UNVERIFIED].
- S.C. Douglas: only "An efficient implementation of the modified filtered-X LMS algorithm", IEEE SPL 4 (1997), doi:10.1109/97.633770 [VERIFIED]. It removes the output delay so that standard LMS applies, which bears on the delay limit, not on tonal (μ, L) tuning.

---

## 3. Synthesis by question

**Q1: convergence and stability for tonal references.**
- The tonal case is the one solved exactly. The adaptive filter plus plant is LTI (refs 1–5, 9–11).
- With L taps, the reference correlation has two non-zero eigenvalues (A²/4)(L ± sin Lω / sin ω) (F1). Time constants are τ = 1/(2μλ̃), and the convergence rate scales as μ L A² |Ĝ||G| cos Δφ (F2).
- With a pure delay Δ the bound is β < 2 sin(π/(2(2Δ+1))) and the fastest time constant is about Δ samples (F3b). For broadband inputs the classical approximation is μ_max ≈ 2/(P_x′(L+Δ)) (F3a). The two are different regimes; the tonal one is the relevant regime here.
- Stochastic theories (Bjarnason; Tobias/Bermudez; Ardekani & Abdulla; Yang 2020) target random references. Ardekani & Abdulla explicitly say earlier exact conditions held only for "pure delay … or multi-sinusoidal inputs".

**Q2: has anyone used μL (μ_opt ∝ 1/L) and built the full (μ, L) map?**
- *The scaling* is implicit in the notch bandwidth (Widrow 1975), in NLMS normalisation by ‖x‖² ∝ L, in Kuo et al. 1999 (amplitude equalisation), in Kuo & Puvvala 2006 (filter length needed for a given speed), and in Siswanto & Chang 2025 (frequency-dependent rates for given L).
- *A closed-form or semi-analytical prediction of the whole 2-D (μ, L) convergence-time map*, especially in AVC with a lightly damped structure and checked against measurements, **was not found**.
- The nearest approaches are:
  - Zhang 2022: ANC; analytical μ interval and optimum; two-weight notch, so no L.
  - Li et al. 2024: AVC; a simulation predictor from a pre-identified path.
  - Ouyang 2026: AVC; a simulation predictor from sd-DMDc. It notes that "as L increases the optimal … region shifts toward a smaller μ" but gives no law.
  - Puri 2016: closed-form μ_opt from power + delay; ANC.
  - Ferrer 2023: a step-selection rule; ANC.

**Q3: the ±90° rule.**
- For slow adaptation, stability requires |Δφ(ω0)| < 90°. The rate scales as cos Δφ, and the steady state is unaffected while stable (refs 35, 36, 27).
- With delay or a large step the margin shrinks: β_max = 2 sin((π/2 − |Δφ|)/(2Δ+1)) (F4, checked numerically).
- For long filters and general paths, the exact condition is that the Toeplitz cross-correlation spectrum lies in the right half plane (Fraanje 2007), with SPR as a sufficient condition (Wang & Ren 1999).
- For the FEM-versus-sd-DMDc comparison: FEM's narrower region should be explainable by its gain error and Δφ at ω0. Audit M7 already finds a 14 % gain error that accounts for the μ shift. This links the comparison to Boucher 1991 rather than to "model fidelity" in general.

**Q4: NFxLMS / VSS (2018–2026).** Table E. They do not remove tuning.
- NFxLMS turns (μ, L, A) into one α, which is the β of F2. A paper that tunes α on the sd-DMDc model would rediscover the μL collapse.
- VSS adds 2–5 hyperparameters (Ouyang Table 1).
- None predicts the convergence-time map.

**Q5: multichannel.**
- Modes follow the eigenvalues of Ĝᴴ G (F5). Ill-conditioned modes converge slowly and need large effort (Elliott 1992). Model error makes Ĝᴴ G non-Hermitian, and instability means some Re λ < 0 (Boucher 1991; Omoto & Elliott 1999; Ferrer 2023).
- Earlier divergence with equal per-channel (μ, L) follows from summed loop gain: λ_max(Ĝᴴ G) ≥ maxᵢ|Gᵢ|². In Ouyang's 2 × 1 layout λ₁ = Σᵢ Ĝᵢ* Gᵢ, and a zero-eigenvalue null-space mode needs leakage.
- Principal-component LMS (Cabell 1998) gives each mode its own step. That is a cheaper, principled alternative to the 4-D grid search announced in the 2026 paper.

---

## 4. Quantitative cross-check on the 2026 paper's own numbers (from the extracted text)

(μL)_opt from the two orthogonal sweeps of each case:

| Case | μ_opt at fixed L | (μL)_opt | L_opt at fixed μ | (μL)_opt |
|---|---|---|---|---|
| 120 Hz, S2 | 9e-6 at L = 128 | 1.15e-3 | 100 (exp) / 112 (sim) at 1e-5 | 1.00e-3 / 1.12e-3 |
| 280 Hz, S1 | 1.1e-5 at L = 128 | 1.41e-3 | 112 / 120 at 1.2e-5 | 1.34e-3 / 1.44e-3 |
| 85 Hz, 2-ch | 1.6e-5 (exp) / 1.3e-5 (sim) at L = 64 | 1.02e-3 / 0.83e-3 | 150 / 128 at 6e-6 | 0.90e-3 / 0.77e-3 |

- Agreement between the two sweeps is 3–13 %, as F2 predicts. It is the same table as audit H1, re-derived here from the text.
- At 120 Hz, 1/sin ω = 13.3, so for L ≥ 48 the eigenvalue split is ≤ ±28 %, and ≤ ±10 % at L = 128. The collapse should fail only at small L, which is what audit H3 finds (L = 32: slowdown 1.38–1.42, against the bound 1.38).

---

## 5. Novelty assessment

**The idea being assessed:** an analytical (μ, L) convergence-time map, computed from a data-driven (sd-DMDc) secondary-path model and validated against the digitized experimental maps of the 2026 paper.

### 5.1 Classical: cannot be claimed, must be cited

1. Tonal (Fx)LMS is LTI; root locus and Nyquist give the stability limit (Widrow 1975; Glover 1977; Elliott & Darlington 1985; Elliott et al. 1987; Sievers & von Flotow 1992; Messner & Bodson 1995; Vicente & Masgrau 2002/2006; Hinamoto 2006; Wen 2021).
2. Rank-2 eigenstructure, L ± sin Lω / sin ω, and time constants τ = 1/(2μλ).
3. μ·L (and amplitude) scaling of the tonal rate, which underlies NLMS and amplitude or eigenvalue equalisation (Kuo 1999; Thomas 2008). The filter-length and frequency dependence is studied by Kuo & Puvvala 2006 and Siswanto & Chang 2025.
4. Delay bounds (Long et al. 1989/1992; Ernst 1995; Elliott & Nelson via Kajikawa 2012). The ±90° phase rule and cos Δφ rate (Boucher 1991; Snyder & Hansen 1994; Fraanje 2007).
5. Multichannel eigen-analysis on Ĝᴴ G (Elliott 1987, 1992; Boucher 1991; Gong 2022).
6. Offline-model-based prediction and tuning of FxLMS parameters: Li et al. 2024 (AVC, simulation); Zhang 2022 (ANC, analytical μ interval and optimum); Puri 2016; Ferrer 2023; and Ouyang 2026 itself (AVC, simulation).

### 5.2 Partly anticipated: priority not established by this sweep

7. *A structural resonance limits tonal LMS stability and speed.*
   - Anticipated qualitatively by Darlington & Elliott 1985 (2nd-order network) and by the Elliott et al. 1987 abstract ("timescale comparable to the response time of the system").
   - Related: Batista 2021 (adaptive transients excite resonances) and Scribner 1993 (passive damping limits narrowband feedback).
   - The explicit **damping budget** — decay(adaptation pole) + decay(adjacent mode) = σ_n, independent of gain, hence α_div = σ_n and t_min ≈ c/σ_n, the "330 cycles" regularity — was **not found** in any abstract or passage seen.
   - It is a root-locus trace identity, so a reviewer may call it "known in principle". **UNVERIFIED negative.** The full texts of Darlington & Elliott 1985, Darlington's 1987 thesis, Elliott 2001 Ch. 3 and Morgan & Sanford 1992 must be read before claiming priority.

### 5.3 New, as far as this sweep can determine

8. **A closed-form or semi-analytical predictor of the complete 2-D (μ, L) t₅ map for tonal AVC.** It has three pieces:
   - slow branch: eigenvalues with the L ± S correction;
   - fast boundary and t_min: the σ_n budget or the LTI equivalent with the identified G_s(z);
   - phase and gain error: cos Δφ, |Ĝ|/|G|.

   It is fed by a **data-driven sd-DMDc model** rather than a measured FIR or FEM model. No paper predicts a 2-D (μ, L) convergence-time map analytically for a lightly damped structure.
9. **Quantitative validation against digitized published experimental maps** for three cases (120 Hz/S2, 280 Hz/S1, 85 Hz two-channel), with no new experiment. This is a re-analysis method that suits the no-funds constraint. Falsifiable hypotheses (audit HP1–HP6) must be declared before the runs to avoid the charge of curve-fitting.
10. **Showing, model-free and on the paper's own data, that the published 2-D maps collapse onto one variable** (μL, or μ·L_eff with L_eff = 2λ_min(Γ)).
    - Theoretical novelty is low: it follows from item 3.
    - Empirical and interpretive novelty is real: the 2026 paper attributes the effect of L to "degrees of freedom to model the secondary path" and to "computational delay", which this contradicts.
11. **Design law for hybrid control:** t_min'/t_min = σ_n/σ_n' when feedback damping (PPF / FOPID / ADRC) is added under FxLMS.
    - Hybrid schemes exist: PID-FxLMS (Li 2021), feedforward + H∞ (Qin et al. 2021, Shock Vib., doi:10.1155/2021/3153531 [VERIFIED]), and the Ouyang group's sd-DMDc + PPF (Eng. Struct. 2026).
    - No **quantitative** law linking t_min to added modal damping was found. New, and testable on the virtual testbed.

### 5.4 Verdict

- **An analytical (μ, L) map built from classical eigenvalue and delay formulas alone:** novelty **low to medium**. Reviewers would cite refs 1–5, 24–30 and 35–36 and call it textbook theory applied to a known dataset.
- **The full package:** novelty **medium to high**, publishable in JSV, MSSP or Thin-Walled Struct. The package is:
  - (i) data-driven sd-DMDc model → closed-form map;
  - (ii) proof that the 2-D experimental maps collapse to 1-D;
  - (iii) the σ_n damping-budget law explaining the fast boundary and t_min, which pure-delay theory misses by a factor of 40–130;
  - (iv) validation on three published experimental maps;
  - (v) the hybrid-damping design law.

  Its strongest unique element is (iii)+(v). It must survive a full-text check of Darlington & Elliott 1985 and of Elliott 2001 Ch. 3.
- **Recommended framing (one line):** *"The step-size/filter-length map of tonal FxLMS on lightly damped structures is one-dimensional and damping-limited: a closed-form, data-driven predictor validated against published experiments."* Position it against Li et al. 2024 (simulation predictor), Zhang 2022 (analytical μ range in ANC), Siswanto & Chang 2025 (frequency- and L-dependent rate in ANC) and Ouyang 2026 (simulation predictor).
- **Must-cite set (12):** Widrow 1975; Glover 1977; Elliott et al. 1987; Boucher et al. 1991; Elliott et al. 1992; Snyder & Hansen 1994; Sievers & von Flotow 1992; Vicente & Masgrau 2006; Fraanje et al. 2007; Siswanto & Chang 2025; Li et al. 2024; Zhang et al. 2022 (MSSP 109831). Add Darlington & Elliott 1985 if its content is confirmed.

---

## 6. Open items marked UNVERIFIED

- The BW = μ N C²/T expression of Widrow et al. (1975).
- The exact closed forms in Long et al. (1989/1992), Elliott (2001), Boucher et al. (1991) and Vicente & Masgrau (2006). Our F3(b) and F4 closed forms are verified **numerically only**.
- Morgan & Sanford (1992): no record found.
- Journal versions of Bjarnason (1995) and Tobias et al. (2000).
- The original source of μ_max = 2/(P_x′(L+Δ)) (Elliott & Nelson 1989?).
- The content of Darlington & Elliott (1985, IOA) and Darlington's 1987 thesis beyond their abstracts. This is the key priority check for the σ_n law.
- The abstract and content of Ouyang et al., Eng. Struct. 2026 (122232). This is the key check for the hybrid PPF + FxLMS kernel.
- "Wu, Zhou et al.", as named in the task.

## 7. Files

- `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/research/lit_B_fxlms.md` (this file)
- `/tmp/claude-0/-home-user-new-controller/c719504e-faf8-5f11-bb9b-685db8a8268c/scratchpad/litB_tools/check_formulas.py` and `check_formulas_out.txt`: numerical checks (a)–(e) of F1–F4.
