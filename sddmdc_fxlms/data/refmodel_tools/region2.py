import sys
from PIL import Image
import numpy as np
def analyse(path, panel, xlim, ylim, Ls):
    rgb=np.array(Image.open(path).convert('RGB')).astype(int)
    im=rgb.mean(axis=2)
    H,W=im.shape
    y0p = H//2; x0p = 0 if panel=='left' else W//2
    sub=im[y0p:, x0p:x0p+W//2]; srgb=rgb[y0p:, x0p:x0p+W//2]
    dark=sub<90
    h,w=sub.shape
    rows=[y for y in range(h) if dark[y].sum()>0.6*w*0.7]
    cols=[x for x in range(w) if dark[:,x].sum()>0.6*h*0.8]
    top,bot=min(rows),max(rows)
    cols=[c for c in cols]
    lef=min(cols); 
    # right frame: the dark column closest to lef + plausible (first big gap)
    cand=[c for c in cols if c-lef>200]
    rig=min(cand)
    print('frame lef,rig,top,bot',lef,rig,top,bot, 'all cols',cols)
    fx=lambda x: xlim[0]+(x-lef)/(rig-lef)*(xlim[1]-xlim[0])
    fyinv=lambda L: bot-(L-ylim[0])/(ylim[1]-ylim[0])*(bot-top)
    sat=(np.max(srgb,axis=2)-np.min(srgb,axis=2))>40
    for L in Ls:
        y=int(round(fyinv(L)))
        y=min(max(y,top+3),bot-3)
        xs=[x for x in range(lef+3,rig-2) if sat[y,x]]
        if xs:
            # split into segments
            segs=[];cur=[xs[0]]
            for x in xs[1:]:
                if x-cur[-1]>3: segs.append(cur);cur=[x]
                else: cur.append(x)
            segs.append(cur)
            print('L=%3d '%L + '  '.join('[%.2f, %.2f]'%(fx(s[0]),fx(s[-1])) for s in segs))
        else: print('L=%3d none'%L)
if __name__=='__main__':
    path=sys.argv[1]; panel=sys.argv[2]
    xlim=[float(v) for v in sys.argv[3].split(',')]; ylim=[float(v) for v in sys.argv[4].split(',')]
    Ls=[int(v) for v in sys.argv[5].split(',')]
    analyse(path,panel,xlim,ylim,Ls)
