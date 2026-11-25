"""Words of prose in LaTeX section files (comments, figures, tables and display maths excluded).
Usage: python3 tools/wordcount.py sections/s4_bistability.tex [...]"""
import re, sys
def words(path):
    s = open(path).read()
    s = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('%'))
    s = re.sub(r'(?<!\\)%.*', '', s)
    s = re.sub(r'\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}', '', s, flags=re.S)
    s = re.sub(r'\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}', '', s, flags=re.S)
    s = re.sub(r'\\\[.*?\\\]', '', s, flags=re.S)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', '', s)
    s = re.sub(r'[{}$]', ' ', s)
    return len(s.split())
tot = 0
for p in sys.argv[1:]:
    w = words(p); tot += w
    print(f"{p}: {w}")
if len(sys.argv) > 2:
    print("total:", tot)
