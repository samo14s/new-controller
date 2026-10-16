"""Semi-automatic digitizer for the 1-D convergence-time sweep panels.
usage: digit.py img x0,y0,x1,y1  (panel crop)  -> prints frame + label blobs"""
import sys, json
from PIL import Image
import numpy as np
from scipy import ndimage
def load(path, box):
    rgb=np.array(Image.open(path).convert('RGB')).astype(int)
    x0,y0,x1,y1=box
    return rgb[y0:y1, x0:x1]
def frame(s):
    blk=(s.max(axis=2)<100)&((s.max(axis=2)-s.min(axis=2))<40)
    h,w=blk.shape
    rows=[y for y in range(h) if blk[y].sum()>0.5*w]
    cols=[x for x in range(w) if blk[:,x].sum()>0.5*h]
    return blk, rows, cols
def blobs(mask, minpix=8):
    lab,n=ndimage.label(mask)
    out=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        if len(xs)>=minpix: out.append((xs.min(),xs.max(),ys.min(),ys.max(),len(xs)))
    return out
if __name__=='__main__':
    path=sys.argv[1]; box=[int(v) for v in sys.argv[2].split(',')]
    s=load(path,box); blk,rows,cols=frame(s)
    print('rows',rows); print('cols',cols)
    top,bot=min(rows),max(rows); lef,rig=min(cols),max(cols)
    # y labels left of frame
    yb=blobs(blk[top-15:bot+15, :max(lef-3,1)])
    print('ylabel blobs (x0,x1,y0,y1):',[(a,b,c+top-15,d+top-15) for a,b,c,d,n in yb])
    xb=blobs(blk[bot+3:bot+45, lef-15:rig+15])
    print('xlabel blobs:',[(a+lef-15,b+lef-15,c,d) for a,b,c,d,n in xb])
