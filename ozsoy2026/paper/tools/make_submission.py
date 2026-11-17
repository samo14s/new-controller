"""Build a flat, self-contained LaTeX source package for Editorial Manager.

Editorial Manager places all uploaded files in one directory, so the
submission version inlines every \\input, strips directory prefixes from
\\includegraphics, drops the internal provenance comments ("% src: ..."),
and ships the .bbl next to refs.bib.  The package is test-compiled in a
clean directory before it is zipped.

Usage (from paper/):  python3 tools/make_submission.py
Output: submission/ (flat sources + PDF) and submission.zip
"""
import os, re, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper/
OUT = os.path.join(HERE, "submission")


def inline(path, depth=0):
    if depth > 5:
        raise RuntimeError("\\input nesting too deep at " + path)
    text = open(path, encoding="utf-8").read()

    def repl(m):
        name = m.group(1)
        f = os.path.join(HERE, name if name.endswith(".tex") else name + ".tex")
        return inline(f, depth + 1).rstrip("\n")

    # only \input lines that are not commented out
    return re.sub(r"(?m)^(?!%)[ \t]*\\input\{([^}]+)\}", repl, text)


def main():
    if not os.path.exists(os.path.join(HERE, "main.bbl")):
        sys.exit("run latexmk -pdf main.tex first (main.bbl missing)")
    flat = inline(os.path.join(HERE, "main.tex"))
    flat = re.sub(r"(?m)^[ \t]*% src:.*\n", "", flat)                 # internal provenance notes
    figs = []

    def fig(m):
        opts, name = m.group(1) or "", m.group(2)
        base = os.path.basename(name)
        figs.append(name)
        return "\\includegraphics" + opts + "{" + base + "}"

    flat = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", fig, flat)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(flat)
    for name in figs:
        src = os.path.join(HERE, name)
        if not os.path.splitext(src)[1]:
            for ext in (".pdf", ".png", ".jpg", ".eps"):
                if os.path.exists(src + ext):
                    src += ext
                    break
        shutil.copy(src, OUT)
    for f in ("refs.bib", "main.bbl", "highlights.txt"):
        shutil.copy(os.path.join(HERE, f), OUT)
    # test-compile in the clean directory (the .bbl is used as is)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode:
            sys.exit("flat build failed:\n" + r.stdout[-3000:])
    log = open(os.path.join(OUT, "main.log"), encoding="latin-1").read()
    bad = re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined", log)
    if bad:
        sys.exit("undefined references in the flat build: " + "; ".join(sorted(set(bad))[:10]))
    for f in os.listdir(OUT):                                           # keep sources + PDF only
        if f.split(".")[-1] in ("aux", "log", "out", "spl", "fls", "fdb_latexmk", "blg"):
            os.remove(os.path.join(OUT, f))
    zpath = os.path.join(HERE, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), f)
    print("ok:", len(figs), "figures;", zpath)


if __name__ == "__main__":
    main()
