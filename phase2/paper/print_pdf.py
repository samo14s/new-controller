"""Paginate the built HTML into a PDF.

Chromium does the page breaking, honouring the @page rules and the
break-inside/break-after hints the build script sets on headings, tables and
display maths.  The page number goes in the footer, which Chromium renders in
its own context -- so it is kept to Latin digits, which need no embedded font.

    node phase2/paper/build_pdf.mjs out.html
    python phase2/paper/print_pdf.py out.html paper_ar.pdf
"""
import os
import sys

from playwright.sync_api import sync_playwright

CHROMIUM = '/opt/pw-browsers/chromium'

FOOTER = """
<div style="width:100%;font-family:Georgia,serif;font-size:8pt;color:#5B6A70;
            padding:0 18mm;display:flex;justify-content:center;">
  <span class="pageNumber"></span>
</div>
"""

EMPTY = '<span></span>'


def main(src, out):
    src = os.path.abspath(src)
    out = os.path.abspath(out)
    exe = CHROMIUM if os.path.exists(CHROMIUM) else None
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=exe)
        page = browser.new_page()
        page.goto(f'file://{src}')
        page.wait_for_load_state('networkidle')
        page.evaluate('document.fonts.ready')
        page.pdf(path=out, format='A4', print_background=True,
                 display_header_footer=True,
                 header_template=EMPTY, footer_template=FOOTER,
                 margin=dict(top='20mm', bottom='18mm',
                             left='18mm', right='18mm'))
        browser.close()
    print(f'{out}  ({os.path.getsize(out) / 1024:.0f} KB)')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
