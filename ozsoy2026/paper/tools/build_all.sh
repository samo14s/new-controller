#!/usr/bin/env bash
# Build every file of the submission from paper/:
#   main.pdf, supplementary.pdf, cover_letter.pdf, submission/ + submission.zip, main.docx,
#   notes/review2_round2_response.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
# refuse draft placeholders left from the revision (macros defined between DRAFT-PLACEHOLDERS markers)
if grep -n "DRAFT-PLACEHOLDERS\|\\PEND{\|\\NEW[A-Z]*\b" main.tex supplementary.tex sections/*.tex >/dev/null; then
  echo "draft placeholders remain:"; grep -n "DRAFT-PLACEHOLDERS\|\\PEND{\|\\NEW[A-Z]*\b" main.tex supplementary.tex sections/*.tex; exit 1
fi
if grep -n "\[basin-aware result\]\|\[contact loss\]\|\\PEND{" cover_letter.tex >/dev/null; then
  echo "cover letter placeholders remain"; exit 1
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error cover_letter.tex >/dev/null
python3 tools/make_submission.py
python3 tools/make_docx.py
# the response to the second review, rendered from its Markdown
pandoc notes/review2_round2_response.md -o notes/review2_round2_response.pdf --pdf-engine=pdflatex -V geometry:margin=0.8in -V colorlinks=true
for f in main supplementary; do
  grep -q "undefined" $f.log && { echo "undefined references in $f.log"; grep "undefined" $f.log | sort -u; exit 1; } || true
done
pdfinfo main.pdf | grep Pages; pdfinfo supplementary.pdf | grep Pages
