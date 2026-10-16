"""Locate y ticks (left axis) and the dashed threshold line of a panel; print threshold value.
args: img xf_lo xf_hi y0 y1 tick_values(comma, top->bottom)"""
import sys, numpy as np
from PIL import Image
img = sys.argv[1]; xl, xh, y0, y1 = map(int, sys.argv[2:6]); ticks = [float(v) for v in sys.argv[6].split(',')]
im = np.array(Image.open(img).convert('RGB')).astype(int)
d = (im[..., 0] < 90) & (im[..., 1] < 90) & (im[..., 2] < 90)
fc = xl + int(np.argmax(d[y0:y1, xl:xh].sum(0)))
rows = [y0+i for i in range(y1-y0) if d[y0+i, fc+3:fc+9].all()]
# group consecutive
grp = []
for r in rows:
    if grp and r - grp[-1][-1] <= 1: grp[-1].append(r)
    else: grp.append([r])
cent = [np.mean(g) for g in grp]
# dashed line: row with many dark pixels across panel width, in short runs
W = 500
cnt = [(y0+i, d[y0+i, fc+20:fc+20+W].sum()) for i in range(y1-y0)]
print('frame col', fc, 'tick-like rows', [round(c, 1) for c in cent])
cand = [r for r, c in cnt if 0.3*W < c < 0.9*W]
print('dashed candidates', cand)
