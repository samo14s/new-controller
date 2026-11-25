#!/usr/bin/env bash
# Build the standard-length version: main_std.pdf and supplementary_std.pdf (they cross-reference
# each other through xr-hyper, so both are compiled in turn until the references settle).
set -euo pipefail
cd "$(dirname "$0")/.."
export BIBINPUTS="..:${BIBINPUTS:-}"
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1.tex" > /dev/null || { echo "LaTeX error in $1.tex"; grep -n "^!" -A4 "$1.log" | head -20; exit 1; }; }
for f in main_std supplementary_std; do run $f; done
for f in main_std supplementary_std; do bibtex -terse $f > /dev/null || true; done
for i in 1 2; do for f in main_std supplementary_std; do run $f; done; done
for f in main_std supplementary_std; do
  if grep -q "There were undefined references\|Reference .* undefined\|Citation .* undefined" $f.log; then
    echo "undefined references in $f.log:"; grep "Reference .* undefined\|Citation .* undefined" $f.log | sed 's/ on input line.*//' | sort -u | head -40
  fi
  echo "$f: $(pdfinfo $f.pdf | awk '/Pages/{print $2}') pages, $(grep -c Overfull $f.log) overfull boxes"
done
echo "references in main_std: $(grep -c '\\bibitem' main_std.bbl)"
