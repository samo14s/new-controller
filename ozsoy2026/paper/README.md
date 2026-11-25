# Manuscript: Beyond linear stability lobes — saturation-induced finite-amplitude instability and basin-aware gain scheduling in actively damped milling

Research article prepared for an Elsevier journal (*Mechanical Systems and Signal Processing* in the
`\journal{}` line; see "Journal" below) from the computational study in this repository, a reproduction
and extension of Ozsoy, Sims & Ozturk 2026, J. Manuf. Processes 171:247–257. This is the third version,
revised in response to two rounds of external pre-submission review (`notes/review2_response.md`,
`notes/review2_round2_response.md`).

## Files

| File | What it is |
|---|---|
| `main.pdf` | the manuscript (elsarticle preprint, 12 pt, line numbers) |
| `supplementary.pdf` | Supplementary Material (sections S3–S7) |
| `cover_letter.pdf` | cover letter to the editor (draft) |
| `highlights.txt` | the five highlights as a separate file (each ≤ 85 characters) |
| `submission.zip` | flat LaTeX source for Editorial Manager: `main.tex`, `main.bbl`, `refs.bib`, figure PDFs, highlights, and the PDFs above |
| `main.docx` | editable Word copy (numbers taken from the LaTeX build; the PDF is the reference) |
| `main.tex`, `sections/`, `figures/`, `refs.bib` | sources |
| `notes/review2_response.md` | point-by-point response to the external review, first round |
| `notes/review2_round2_response.md` | point-by-point response to the external review, second round (fair comparison, Monte Carlo of the chart, depth cap, interval proof) |
| `notes/internal_review.md`, `notes/review_triage.json` | the two internal referee-style reviews of the first draft and their triage |
| `notes/novelty.md` | literature memo: what is published and which claims the paper must not make |

Structure: 1 Introduction; 2 Model and experimental benchmark; 3 Actuator authority and the forced orbit;
4 Saturation-induced finite-amplitude instability; 5 Basin estimation (circle criterion, LMIs); 6
Basin-aware gain scheduling; 7 Delayed feedback and robustness (Monte Carlo); 8 Discussion and
conclusions; Appendices A (simulator), B (LMI settings), C (software and use of AI tools).

Rebuild everything with `tools/build_all.sh` (TeX Live with elsarticle, Python 3, pandoc for the DOCX);
it refuses to build while draft placeholders remain. The figures come from `figures/make_figures.py`,
which reads only the stored results in `../results` and `../verification`; the numbers added during the
revisions come from `../verification/paper/*.py`, each with its JSON output, and every number in the
LaTeX source has a `% src:` comment naming its file and key.

## To be completed by the authors before submission

1. Authors, affiliations, corresponding author and e-mail (`main.tex`, `cover_letter.tex`). With a full
   author list the front matter runs onto a second page in the preprint layout; this is normal.
2. CRediT statement, funding, acknowledgements, repository URL (data availability and Appendix C).
3. **Generative-AI declarations.** The code, the analyses and the text were produced with an AI assistant,
   named in both statements (Claude Code, Anthropic). Complete Appendix C with who designed the study and
   which steps were checked by hand, add any other tool used, and check the journal's current policy.
4. Suggested reviewers in the cover letter (choose experts without conflicts; verify affiliations).
5. **Second benchmark (done).** Ozsoy et al. 2025 (MSSP 224:111942) turned out to use the same rig; its
   Figs. 6 and 10/12 were digitised and compared with the model (Section 2.6, Supplement S6,
   `../verification/ozsoy2025/`). A data set from a different machine would still be needed to test
   generality.
6. **Journal.** MSSP states in its scope that theoretical nonlinear dynamics without experimental
   validation is out of scope, and this study is computational (validated only against published
   cutting tests): a desk rejection there is a real risk. Journal of Sound and Vibration, Nonlinear
   Dynamics, International Journal of Mechanical Sciences and Journal of Manufacturing Processes are the
   alternatives discussed in `notes/review2_response.md`. The `elsarticle` format serves the four Elsevier
   journals (change the `\journal{}` line and the cover letter); Nonlinear Dynamics (Springer) needs its
   own template. Check the journal's quartile and your institution's list separately.
7. Check every statement against your own understanding: all results are computational, and no
   experiment supports the bistability predictions yet (Discussion, last two paragraphs).
