# Manuscript: Actuator saturation and bistability in active chatter control of milling

Research article prepared for *Mechanical Systems and Signal Processing* (Elsevier) from the computational
study in this repository (a reproduction and extension of Ozsoy, Sims & Ozturk 2026, J. Manuf. Processes
171:247–257).

## Files

| File | What it is |
|---|---|
| `main.pdf` | the manuscript (elsarticle preprint, 12 pt, line numbers) |
| `supplementary.pdf` | Supplementary Material (sections S3–S6) |
| `cover_letter.pdf` | cover letter to the editor (draft) |
| `highlights.txt` | the five highlights as a separate file (each ≤ 85 characters) |
| `submission.zip` | flat LaTeX source for Editorial Manager: `main.tex`, `main.bbl`, `refs.bib`, figure PDFs, highlights, and the PDFs above |
| `main.docx` | editable Word copy (numbers taken from the LaTeX build; the PDF is the reference) |
| `main.tex`, `sections/`, `figures/`, `refs.bib` | sources |
| `notes/internal_review.md`, `notes/review_triage.json` | two internal referee-style reviews of the first draft and their triage |
| `notes/novelty.md` | literature memo: what is published and which claims the paper must not make |

Rebuild everything with `tools/build_all.sh` (TeX Live with elsarticle, Python 3, pandoc for the DOCX).
The figures come from `figures/make_figures.py`, which reads only the stored results in `../results`
and `../verification`; the numbers added during the revision come from `../verification/paper/*.py`.

## To be completed by the authors before submission

1. Authors, affiliations, corresponding author and e-mail (`main.tex`, `cover_letter.tex`). With a full
   author list the front matter runs onto a second page in the preprint layout; this is normal.
2. CRediT statement, funding, acknowledgements, repository URL (data availability and Appendix C).
3. **Generative-AI declarations.** The code, the analyses and the text were produced with an AI assistant.
   Name the tool and complete both statements: the declaration before the references (writing) and the
   paragraph in Appendix C (research use), as Elsevier's policy requires. Check the journal's current
   policy.
4. Suggested reviewers in the cover letter (choose experts without conflicts; verify affiliations).
5. Read the full text of Ozsoy et al. 2025 (MSSP 224:111942) and state in Section 4.5 whether its
   frequency-domain model already contains the forced command and a threshold amplitude.
6. Check every statement against your own understanding: all results are computational, and no
   experiment supports the bistability predictions yet (Discussion, last two paragraphs).
