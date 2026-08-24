# Paper draft

`draft.tex` — manuscript draft (plain `article` class; switching to a
journal class such as `elsarticle` is mechanical). Figures in `figs/`
are generated from the stored result artifacts (`results/*.npz`) by the
session scripts — regenerate rather than edit.

Build: `pdflatex draft.tex` (twice for references). No BibTeX needed —
the bibliography is inline.
