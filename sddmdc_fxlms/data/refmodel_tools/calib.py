import sys
from PIL import Image
import numpy as np
from scipy import ndimage
def grp(vals,tol):
    vals=sorted(vals); g=[];cur=[]
    for v in vals:
        if cur and v-cur[-1]>tol: g.append(cur);cur=[]
        cur.append(v)
    if cur: g.append(cur)
    return [round((min(a)+max(a))/2,1) for a in g]
path=sys.argv[1]; x0,y0,x1,y1=[int(v) for v in sys.argv[2].split(',')]
rgb=np.array(Image.open(path).convert('RGB')).astype(int); s=rgb[y0:y1,x0:x1]
blk=(s.max(axis=2)<100)&((s.max(axis=2)-s.min(axis=2))<40)
h,w=blk.shape
rows=[y for y in range(h) if blk[y].sum()>0.55*w]; cols=[x for x in range(w) if blk[:,x].sum()>0.55*h]
print('rows',rows,'cols',cols)
top,bot=min(rows),max(rows); lef,rig=min(cols),max(cols)
lab,n=ndimage.label(blk[top-12:bot+12,:max(lef-4,1)])
yc=[]
for i in range(1,n+1):
    ys,xs=np.where(lab==i)
    if len(xs)>=6 and xs.max()>lef-4-45: yc+= [ (ys.min()+ys.max())/2+top-12 ]
print('ylabel centers',grp(yc,8))
lab,n=ndimage.label(blk[bot+4:bot+40,max(lef-25,0):rig+25])
xc=[]
for i in range(1,n+1):
    ys,xs=np.where(lab==i)
    if len(xs)>=6 and ys.min()<12: xc+=[ (xs.min()+xs.max())/2+max(lef-25,0) ]
print('xlabel centers',grp(xc,14))
