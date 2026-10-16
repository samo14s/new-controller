"""Estimate modal damping from the beating transient at excitation onset in a raster time plot.
args: img y0 y1 color amp_ref t_left t_right x_left x_right  (axis calibration: x_left px <-> t_left s)
Prints beat period, ripple amplitudes per beat, fitted decay rate sigma."""
import sys, numpy as np
from PIL import Image
img, y0, y1, color = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
amp_ref = float(sys.argv[5]); tl, tr, xl, xr = map(float, sys.argv[6:10])
im = np.array(Image.open(img).convert('RGB')).astype(int)[y0:y1]
R, G, B = im[..., 0], im[..., 1], im[..., 2]
m = ((B > 170) & (R < 90) & (G < 90)) if color == 'blue' else ((R > 190) & (G < 70) & (B < 70))
W = im.shape[1]
h = np.full(W, np.nan)
for c in range(W):
    r = np.where(m[:, c])[0]
    if r.size: h[c] = r.max() - r.min()
t = tl + (np.arange(W) - xl) * (tr - tl) / (xr - xl)
on = np.where(h > 10)[0][0]
sel = (t > 12) & (t < 19.5)
steady = np.nanmedian(h[sel])
amp = h / 2 * amp_ref / (steady / 2)
print(f'onset at t={t[on]:.2f}s; px/s={(xr-xl)/(tr-tl):.2f}')
# ripple about steady
seg = slice(on + 2, on + int(7 * (xr - xl) / (tr - tl)))
tt, aa = t[seg], amp[seg]
# local extrema
ext = []
for i in range(1, len(aa) - 1):
    if np.isnan(aa[i-1:i+2]).any(): continue
    if aa[i] >= aa[i-1] and aa[i] > aa[i+1]: ext.append((tt[i], aa[i], 'max'))
    if aa[i] <= aa[i-1] and aa[i] < aa[i+1]: ext.append((tt[i], aa[i], 'min'))
maxs = [(a, b) for a, b, k in ext if k == 'max']
mins = [(a, b) for a, b, k in ext if k == 'min']
print('maxima:', ' '.join(f'{a:.2f}:{b:.3f}' for a, b in maxs))
print('minima:', ' '.join(f'{a:.2f}:{b:.3f}' for a, b in mins))
# ripple = max - amp_ref at maxima (excess over steady)
ex = np.array([(a, b - amp_ref) for a, b in maxs if b - amp_ref > 0.004])
if len(ex) >= 3:
    p = np.polyfit(ex[:, 0], np.log(ex[:, 1]), 1)
    print(f'excess-at-maxima fit: sigma={-p[0]:.3f} 1/s using {len(ex)} points; beat period ~ {np.median(np.diff(ex[:,0])):.3f}s')
