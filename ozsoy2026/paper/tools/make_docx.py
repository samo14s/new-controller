"""Convert the LaTeX manuscript to an editable Word file (main.docx).

The PDF built from main.tex is the reference; the DOCX is a convenience copy
for co-authors who edit in Word.  Pandoc does not number equations, figures,
tables or sections the way LaTeX does, so every number is taken from the
.aux file of the LaTeX build: \\ref and \\eqref are replaced by their numbers,
equations get \\tag{n}, captions get "Fig. n." / "Table n.".  Citations are
rendered by pandoc's citeproc with the Elsevier numeric style
(tools/elsevier-with-titles.csl).  The TikZ diagram (Fig. 1) is compiled
separately and embedded as a PNG; the data figures use their PNG previews.

Usage (from paper/, after latexmk -pdf main.tex):  python3 tools/make_docx.py
Output: main.docx
"""
import os, re, shutil, subprocess, sys, tempfile, unicodedata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper/
TOOLS = os.path.join(HERE, "tools")


def read(p):
    return open(p, encoding="utf-8").read()


def inline(text, depth=0):
    def repl(m):
        f = os.path.join(HERE, m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return inline(read(f), depth + 1).rstrip("\n")
    return re.sub(r"(?m)^(?!%)[ \t]*\\input\{([^}]+)\}", repl, text)


def labels_from_aux(aux):
    """label -> printed number, from \\newlabel{key}{{num}{page}...}."""
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{((?:[^{}]|\{[^{}]*\})*)\}", read(aux)):
        out[m.group(1)] = re.sub(r"\\[a-zA-Z]+\s*", "", m.group(2)).replace("{", "").replace("}", "").replace(" ~", "~")
    return out


def braced(s, i):
    """Return (content, end) of the brace group starting at s[i] == '{'."""
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def env_blocks(s, name):
    """Yield (start, end) of every \\begin{name}...\\end{name}."""
    for m in re.finditer(r"\\begin\{%s\}" % re.escape(name), s):
        e = s.index(r"\end{%s}" % name, m.end())
        yield m.start(), e + len(r"\end{%s}" % name)


def bib_names(bib):
    """key -> 'Surname', 'Surname and Surname' or 'Surname et al.' (for \\citet)."""
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", read(bib), re.S):
        a = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", m.group(2), re.S)
        if not a:
            continue
        people = [p.strip() for p in re.split(r"\s+and\s+", a.group(1).replace("\n", " "))]
        def surname(p):
            p = p.strip("{}")
            last = p.split(",")[0] if "," in p else p.split()[-1]
            acc = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303", "v": "\u030c", "c": "\u0327"}
            last = re.sub(r"\\([\'`\"^~vc])\s*\{?(\w)\}?", lambda q: q.group(2) + acc[q.group(1)], last)
            return unicodedata.normalize("NFC", last.replace("{", "").replace("}", ""))
        sn = [surname(p) for p in people]
        out[m.group(1).strip()] = sn[0] if len(sn) == 1 else (" and ".join(sn) if len(sn) == 2 else sn[0] + " et al.")
    return out


def tikz_png(work, preamble_macros):
    src = read(os.path.join(HERE, "sections", "fig_model.tex"))
    pic = src[src.index(r"\begin{tikzpicture}"):src.index(r"\end{tikzpicture}") + len(r"\end{tikzpicture}")]
    doc = (r"\documentclass[border=4pt]{standalone}" "\n"
           r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}\usepackage{siunitx}"
           r"\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,calc}" "\n"
           + preamble_macros + "\n\\begin{document}\n" + pic + "\n\\end{document}\n")
    open(os.path.join(work, "fig01_model.tex"), "w").write(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig01_model.tex"], cwd=work,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("TikZ figure failed:\n" + r.stdout[-2000:])
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "fig01_model.pdf", "fig01_model"],
                   cwd=work, check=True)
    return os.path.join(work, "fig01_model.png")


def main():
    aux = os.path.join(HERE, "main.aux")
    if not os.path.exists(aux):
        sys.exit("build the PDF first (latexmk -pdf main.tex): main.aux missing")
    lab = labels_from_aux(aux)
    main_tex = read(os.path.join(HERE, "main.tex"))
    macros = "\n".join(l for l in main_tex.splitlines()
                       if l.startswith(r"\newcommand") or l.startswith(r"\DeclareSIUnit"))
    docx_macros = re.sub(r"_\{\\(lim|max|min)\}", r"_{\\mathrm{\1}}", macros)
    work = tempfile.mkdtemp(prefix="docx_")

    # front matter
    title = braced(main_tex, main_tex.index(r"\title{") + len(r"\title"))[0]
    hl = main_tex[main_tex.index(r"\begin{highlights}") + 18:main_tex.index(r"\end{highlights}")]
    kw = main_tex[main_tex.index(r"\begin{keyword}") + 15:main_tex.index(r"\end{keyword}")].replace(r" \sep", ";")
    body = main_tex[main_tex.index(r"\linenumbers") + len(r"\linenumbers"):main_tex.index(r"\bibliographystyle")]
    body = inline(body)
    abstract = inline(read(os.path.join(HERE, "sections", "s0_abstract.tex")))

    # figures: TikZ -> PNG; data figures -> PNG previews
    png1 = tikz_png(work, macros)
    i = body.index(r"\begin{tikzpicture}")
    j = body.index(r"\end{tikzpicture}") + len(r"\end{tikzpicture}")
    body = body[:i] + r"\includegraphics[width=\textwidth]{%s}" % png1 + body[j:]
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}.]+)(\.pdf)?\}",
                  lambda m: r"\includegraphics[width=\textwidth]{%s}" % os.path.join(HERE, "figures", m.group(2) + ".png"),
                  body)

    # numbered captions for figures and tables
    for env, word in (("figure", "Fig."), ("table", "Table")):
        out, last = [], 0
        for a, b in env_blocks(body, env):
            blk = body[a:b]
            m = re.search(r"\\label\{([^}]+)\}", blk)
            if m and m.group(1) in lab and r"\caption{" in blk:
                k = blk.index(r"\caption{") + len(r"\caption{")
                blk = blk[:k] + r"\textbf{%s %s.} " % (word, lab[m.group(1)]) + blk[k:]
            out.append(body[last:a]); out.append(blk); last = b
        out.append(body[last:])
        body = "".join(out)

    # equation numbers: \label{eq:..} -> \tag{n}.  texmath wants the tag at the
    # end, so in an equation environment it is moved before \end{equation};
    # in align/gather the labels already sit at the line ends.
    out, last = [], 0
    for a, b in env_blocks(body, "equation"):
        blk = body[a:b]
        keys = re.findall(r"\\label\{(eq:[^}]+)\}", blk)
        if r"\begin{aligned}" in blk:                 # texmath: no \tag after an aligned block
            blk = (blk.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
                      .replace(r"\begin{equation}", r"\begin{align}").replace(r"\end{equation}", r"\end{align}"))
        if keys:
            blk = re.sub(r"\\label\{eq:[^}]+\}", "", blk)
            k = blk.rindex(r"\end{align}" if r"\end{align}" in blk else r"\end{equation}")
            blk = blk[:k].rstrip() + r" \tag{%s}" % lab.get(keys[0], "?") + "\n" + blk[k:]
        out.append(body[last:a]); out.append(blk); last = b
    out.append(body[last:])
    body = "".join(out)

    def tag(m):
        k = m.group(1)
        return r"\tag{%s}" % lab[k] if k in lab else ""
    body = re.sub(r"\\label\{(eq:[^}]+)\}", tag, body)
    # \citet: numeric CSL styles print only the number, so write the names here
    names = bib_names(os.path.join(HERE, "refs.bib"))
    body = re.sub(r"\\citet\{([^}]+)\}",
                  lambda m: ", ".join(names.get(k.strip(), k) for k in m.group(1).split(","))
                  + r"~\citep{" + m.group(1) + "}", body)
    # tabular column specs: drop @{} and >{...} that the pandoc reader rejects
    def spec(m):
        sp = re.sub(r">\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", m.group(1))
        return r"\begin{tabular}{" + sp.replace("@{}", "") + "}"
    body = re.sub(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}", spec, body)
    body = re.sub(r"\\multicolumn\{(\d+)\}\{@\{\}([lcr])\}", r"\\multicolumn{\1}{\2}", body)
    # cross references
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(%s)" % lab.get(m.group(1), "??"), body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: lab.get(m.group(1), "??"), body)
    # sections: print their numbers (pandoc would renumber appendices)
    out, pos = [], 0
    for m in re.finditer(r"\\((?:sub)?section)(\*?)\{", body):
        t, end = braced(body, m.end() - 1)
        out.append(body[pos:m.start()])
        rest = body[end:end + 200]
        k = re.match(r"\s*\\label\{([^}]+)\}", rest)
        if m.group(2) or not k:
            out.append(body[m.start():end]); pos = end; continue
        n = lab.get(k.group(1), "")
        prefix = ("%s. " % n.replace(" ~", "~")) if k.group(1).startswith("app:") else ("%s " % n)
        out.append(r"\%s*{%s%s}" % (m.group(1), prefix, t))
        pos = end + k.end()
    out.append(body[pos:])
    body = "".join(out)
    body = body.replace(r"\appendix", "")
    body = re.sub(r"(?m)^[ \t]*%.*\n", "", body)                          # comment lines
    body = re.sub(r"\{\\rpm\}", "{rpm}", body)                            # siunitx unit pandoc does not know
    body = re.sub(r"\\parbox\{[^{}]*\}\{", "{", body)                      # pandoc fails on \parbox (Table 3 note) and then drops Table 2
    body = re.sub(r"\\[Bb]igg?[lr](?=[^a-zA-Z])", "", body)              # texmath rejects \Bigl( in some places
    body = re.sub(r"\\SI\{(-?)e(\d+)\}", r"\\SI{\g<1>1e\2}", body)          # \SI{-e5}: pandoc needs a mantissa
    body = body.replace(r"\lvert", "|").replace(r"\rvert", "|").replace(r"\lVert", r"\|").replace(r"\rVert", r"\|")
    body = re.sub(r"_\{\\(lim|max|min)\}", r"_{\\mathrm{\1}}", body)       # a_{\lim}: an operator in OMML
    body = re.sub(r"\\tag\{([^}]*)\}", r"\\qquad(\1)", body)                 # Word shows no \tag
    # top-level & in align/gather blocks: OMML alignment markers render badly
    for env in ("align", "gather"):
        out, last = [], 0
        for a, b in env_blocks(body, env):
            blk, res, depth, i = body[a:b], [], 0, 0
            while i < len(blk):
                if blk.startswith(r"\begin{", i) and not blk.startswith(r"\begin{" + env + "}", i):
                    depth += 1
                elif blk.startswith(r"\end{", i) and not blk.startswith(r"\end{" + env + "}", i):
                    depth -= 1
                if blk[i] == "&" and depth == 0 and blk[i - 1] != "\\":
                    i += 1
                    continue
                res.append(blk[i]); i += 1
            out.append(body[last:a]); out.append("".join(res)); last = b
        out.append(body[last:])
        body = "".join(out)

    doc = (r"\documentclass{article}" "\n" r"\usepackage{amsmath,amssymb,bm,siunitx,graphicx,booktabs}" "\n"
           + docx_macros + "\n" r"\title{" + title + "}\n" r"\author{[Author name]}" "\n" r"\date{}" "\n"
           r"\begin{document}" "\n" r"\maketitle" "\n"
           r"\section*{Highlights}" "\n" r"\begin{itemize}" + hl + r"\end{itemize}" "\n"
           r"\section*{Abstract}" "\n" + abstract + "\n\n" r"\textbf{Keywords:} " + kw.strip() + "\n"
           + body + "\n" r"\section*{References}" "\n" r"\end{document}" "\n")
    tex = os.path.join(work, "docx_source.tex")
    open(tex, "w", encoding="utf-8").write(doc)
    out_docx = os.path.join(HERE, "main.docx")
    r = subprocess.run(["pandoc", tex, "-f", "latex", "-t", "docx", "--citeproc",
                        "--bibliography", os.path.join(HERE, "refs.bib"),
                        "--csl", os.path.join(TOOLS, "elsevier-with-titles.csl"),
                        "--resource-path", HERE + ":" + work, "-o", out_docx],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("pandoc failed:\n" + r.stderr[-3000:])
    if r.stderr.strip():
        print("pandoc warnings:\n" + r.stderr[-3000:])
    print("ok:", out_docx, "(work dir %s)" % work)


if __name__ == "__main__":
    main()
