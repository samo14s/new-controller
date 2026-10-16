import sys
from PIL import Image
import numpy as np
path,panel=sys.argv[1],sys.argv[2]
lef,rig,top,bot=[float(v) for v in sys.argv[3].split(',')]
xlim=[float(v) for v in sys.argv[4].split(',')]; ylim=[float(v) for v in sys.argv[5].split(',')]
Ls=[int(v) for v in sys.argv[6].split(',')]
rgb=np.array(Image.open(path).convert('RGB')).astype(int)
H,W,_=rgb.shape; x0=0 if panel=='left' else W//2
s=rgb[H//2:, x0:x0+W//2]
sat=(s.max(axis=2)-s.min(axis=2))>40
fx=lambda x: xlim[0]+(x-lef)/(rig-lef)*(xlim[1]-xlim[0])
fy=lambda L: bot-(L-ylim[0])/(ylim[1]-ylim[0])*(bot-top)
# also get min-time colour? report region
for L in Ls:
    y=int(round(fy(L))); y=int(min(max(y,top+3),bot-3))
    xs=[x for x in range(int(lef)+3,int(rig)-2) if sat[y,x]]
    if not xs: print('L=%3d none'%L); continue
    segs=[];cur=[xs[0]]
    for x in xs[1:]:
        if x-cur[-1]>3: segs.append(cur);cur=[x]
        else: cur.append(x)
    segs.append(cur)
    print('L=%3d '%L+'  '.join('[%.2f, %.2f]'%(fx(a[0]),fx(a[-1])) for a in segs))
