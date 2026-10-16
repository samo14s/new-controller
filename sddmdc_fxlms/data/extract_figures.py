"""
Extract the raster figures and the text of the article PDF, so that every
digitisation script in data/ can be re-run.  The PDF itself is not stored in
the repository (publisher copyright).

    pip install pymupdf
    python extract_figures.py path/to/S0263823126006142.pdf out_dir

Writes out_dir/figs/pNN_imgK.png (NN = PDF page) and out_dir/paper_text.txt.
Fig. 16 is p13_img1, Fig. 17 p15_img0, Fig. 18 p15_img1, Fig. 19 p16_img0,
Fig. 20 p16_img1, Fig. 21 p17_img0, Fig. 22 p18_img0, Fig. 23 p18_img1.
"""
import os
import sys

import pymupdf


def main(pdf, out):
    os.makedirs(os.path.join(out, 'figs'), exist_ok=True)
    doc = pymupdf.open(pdf)
    for i, page in enumerate(doc):
        for j, im in enumerate(page.get_images(full=True)):
            pix = pymupdf.Pixmap(doc, im[0])
            if pix.width < 500:
                continue
            if pix.n - pix.alpha >= 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            pix.save(os.path.join(out, 'figs', f'p{i + 1:02d}_img{j}.png'))
    with open(os.path.join(out, 'paper_text.txt'), 'w') as f:
        f.write('\n\n=====PAGE=====\n\n'.join(p.get_text() for p in doc))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
