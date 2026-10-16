import sys
from PIL import Image
import numpy as np
def groups(idx):
    g=[];cur=[]
    for c in idx:
        if cur and c-cur[-1]>2: g.append(cur);cur=[]
        cur.append(c)
    if cur: g.append(cur)
    return [np.mean(a) for a in g]
def analyse(path, panel, xt, yt):
    rgb=np.array(Image.open(path).convert('RGB')).astype(int)
    im=rgb.mean(axis=2)
    H,W=im.shape
    y0p = H//2; x0p = 0 if panel=='left' else W//2
    sub=im[y0p:, x0p:x0p+W//2]; srgb=rgb[y0p:, x0p:x0p+W//2]
    dark=sub<80
    rows=[y for y in range(sub.shape[0]) if dark[y].sum()>250]
    cols=[x for x in range(sub.shape[1]) if dark[:,x].sum()>250]
    top,bot=min(rows),max(rows); lef,rig=min(cols),max(cols)
    # x ticks: dark pixels in rows bot-8..bot-2
    tx=[x for x in range(lef+3,rig-3) if dark[bot-7:bot-2,x].sum()>=4]
    ty=[y for y in range(top+3,bot-3) if dark[y,lef+2:lef+7].sum()>=4]
    gx=groups(tx); gy=groups(ty)
    print('frame',lef,rig,top,bot,'xticks px',np.round(gx,1),'yticks px',np.round(gy,1))
    # calibrate with provided tick values (xt list matches gx, yt list from top to bottom)
    ax=np.polyfit(gx[:len(xt)],xt,1); ay=np.polyfit(gy[:len(yt)],yt,1)
    print('x lims', np.polyval(ax,[lef,rig]), 'y lims', np.polyval(ay,[bot,top]))
    # coloured (non-white, non-grey) mask
    R,G,B=srgb[...,0],srgb[...,1],srgb[...,2]
    sat=(np.max(srgb,axis=2)-np.min(srgb,axis=2))>40
    for L in [40,48,60,64,80,96,100,112,120,128,140,150,160]:
        ypx=np.polyval(np.polyfit(yt,gy[:len(yt)],1),L)
        y=int(round(ypx))
        if y<top+2 or y>bot-2: continue
        xs=[x for x in range(lef+2,rig-1) if sat[y,x]]
        if xs:
            mu=np.polyval(ax,[xs[0],xs[-1]])
            print('L=%3d  mu in [%.2f, %.2f] e-6'%(L,mu[0]*1,mu[1]*1))
        else: print('L=%3d none'%L)
if __name__=='__main__':
    path=sys.argv[1]; panel=sys.argv[2]
    xt=[float(v) for v in sys.argv[3].split(',')]; yt=[float(v) for v in sys.argv[4].split(',')]
    analyse(path,panel,xt,yt)
