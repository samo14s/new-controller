"""Digitize marker centres (exp/sim) in a convergence-time panel.
y calibration: dashed 5-s line row and a second reference row (given) ; x: returns pixel columns."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
img = sys.argv[1]; x0, x1, y0, y1 = map(int, sys.argv[2:6])
im = np.array(Image.open(img).convert('RGB')).astype(int)[y0:y1, x0:x1]
R, G, B = im[..., 0], im[..., 1], im[..., 2]
masks = {
 'blue': (B > 170) & (R < 90) & (G < 120),
 'red': (R > 190) & (G < 80) & (B < 80),
 'purple': (R > 90) & (R < 170) & (G < 90) & (B > 100),
 'green': (G > 130) & (R < 160) & (R > 60) & (B < 90),
}
dark = (R < 60) & (G < 60) & (B < 60)
# dashed line: row with many dark pixels in short runs
rows = dark.sum(1)
for name in sys.argv[6].split(','):
    m = masks[name]
    op = ndi.binary_opening(m, structure=np.ones((5, 5)))
    lab, n = ndi.label(op)
    cs = ndi.center_of_mass(op, lab, range(1, n+1))
    sz = ndi.sum(op, lab, range(1, n+1))
    pts = sorted([(c[1]+x0, c[0]+y0, s) for c, s in zip(cs, sz) if s > 15])
    print(name, len(pts), ' '.join(f'({x:.0f},{y:.1f})' for x, y, s in pts))
print('dark-row candidates (row:count):', ' '.join(f'{i+y0}:{c}' for i, c in enumerate(rows) if c > 0.25*(x1-x0)))
