import sys
from PIL import Image
import numpy as np
path=sys.argv[1]; panel=sys.argv[2]
rgb=np.array(Image.open(path).convert('RGB')).astype(int); im=rgb.mean(axis=2)
H,W=im.shape; y0p=H//2; x0p=0 if panel=='left' else W//2
sub=im[y0p:, x0p:x0p+W//2]; dark=sub<90; h,w=sub.shape
rows=[y for y in range(h) if dark[y].sum()>0.6*w*0.7]
cols=[x for x in range(w) if dark[:,x].sum()>0.6*h*0.8]
top,bot=min(rows),max(rows); lef=min(cols); rig=min([c for c in cols if c-lef>200])
print('frame',lef,rig,top,bot)
# x labels: dark blobs in band below bottom frame
from scipy import ndimage
band=dark[bot+5:bot+40, lef-20:rig+20]
lab,n=ndimage.label(band)
cents=[]
for i in range(1,n+1):
    ys,xs=np.where(lab==i)
    if len(xs)>15: cents.append((xs.min()+lef-20, xs.max()+lef-20))
print('x label blobs (xmin,xmax):',cents)
band=dark[top-10:bot+10, max(lef-60,0):lef-4]
lab,n=ndimage.label(band)
c2=[]
for i in range(1,n+1):
    ys,xs=np.where(lab==i)
    if len(xs)>15: c2.append(round(ys.mean()+top-10,1))
print('y label blob centers:',sorted(c2))
