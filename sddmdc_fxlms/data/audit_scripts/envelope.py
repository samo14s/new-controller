"""Extract upper/lower envelope of a filled time-signal plot from a raster figure.
usage: python envelope.py img y0 y1 color amp_ref t_on_col_guess
Calibrates time by onset (t=5 s) and control-on (t=25 s) columns, amplitude by steady level."""
import sys, numpy as np
from PIL import Image
img, y0, y1, color = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
amp_ref = float(sys.argv[5])
im = np.array(Image.open(img).convert('RGB')).astype(int)[y0:y1]
R, G, B = im[..., 0], im[..., 1], im[..., 2]
if color == 'blue':
    m = (B > 170) & (R < 90) & (G < 90)
elif color == 'red':
    m = (R > 190) & (G < 70) & (B < 70)
cols = np.where(m.any(0))[0]
top = np.full(im.shape[1], np.nan); bot = np.full(im.shape[1], np.nan)
for c in cols:
    r = np.where(m[:, c])[0]
    top[c], bot[c] = r.min(), r.max()
h = bot - top
# onset: first column with height > 10 px
on = np.where(h > 10)[0][0]
# zero line: median of center before onset
zc = np.nanmedian((top[on-30:on-5] + bot[on-30:on-5]) / 2)
# steady: median height over columns on+150..on+250 approx (paper: 10-20 s)
steady = np.nanmedian(h[on+120:on+250])
# control-on: first column after on+250 where h < 0.97*steady
idx = np.where(h[on+250:] < 0.97 * steady)[0]
con = on + 250 + idx[0]
pxs = (con - on) / 20.0
print(f'onset col {on}, control-on col {con}, px/s {pxs:.2f}, steady height {steady:.1f}px, zero row {zc:.1f}')
V = amp_ref / (steady / 2)
t = (np.arange(im.shape[1]) - on) / pxs + 5
amp = h / 2 * V
np.save(img.replace('.png', f'_{color}_{y0}_env.npy'), np.vstack([t, amp]))
for c in range(on, on + int(6 * pxs), max(1, int(pxs / 8))):
    print(f'{t[c]:6.2f} s  amp {amp[c]:.3f} V')
