#!/usr/bin/env bash
# Compile one section file of the standard version on its own, to catch LaTeX errors early.
# Usage: tools/check_section.sh sections/s4_bistability.tex [main|supp]
# Undefined references and citations are expected here (other sections are absent) and are not errors.
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
f="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
kind="${2:-main}"
master="$here/main_std.tex"; [ "$kind" = supp ] && master="$here/supplementary_std.tex"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/stdcheck.XXXXXX")"
python3 - "$master" "$f" "$tmp/check.tex" <<'PY'
import sys,re
m=open(sys.argv[1]).read()
pre=m.split('\\begin{document}')[0]
pre=re.sub(r'\\externaldocument\[[^\]]*\]\{[^}]*\}','',pre)
open(sys.argv[3],'w').write(pre+'\\begin{document}\n\\input{'+sys.argv[2][:-4]+'}\n\\end{document}\n')
PY
cp -r "$here/sections" "$tmp/" 2>/dev/null
( cd "$here" && TEXINPUTS="$here:$here/..:" pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$tmp" "$tmp/check.tex" > "$tmp/out.txt" 2>&1 ); rc=$?
grep -n "^!" -A3 "$tmp/check.log" | head -20
grep -c "Overfull" "$tmp/check.log" | sed 's/^/overfull boxes: /'
[ $rc -eq 0 ] && echo "COMPILES (pages: $(pdfinfo "$tmp/check.pdf" 2>/dev/null | awk '/Pages/{print $2}'))" || echo "FAILED (log: $tmp/check.log)"
