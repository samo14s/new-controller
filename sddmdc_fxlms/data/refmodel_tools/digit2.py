import sys, json
from PIL import Image
import numpy as np
COL={'blue':lambda R,G,B:(B>150)&(R<110)&(G<110),
     'red':lambda R,G,B:(R>170)&(G<90)&(B<90),
     'purple':lambda R,G,B:(R>90)&(R<170)&(G<80)&(B>100)&(B<190),
     'green':lambda R,G,B:(G>140)&(R>90)&(R<170)&(B<90)}
def run(path, box, xmap, ymap, xs, color, inset=None, bandw=1):
    rgb=np.array(Image.open(path).convert('RGB')).astype(int)
    x0,y0,x1,y1=box; s=rgb[y0:y1,x0:x1]
    R,G,B=s[...,0],s[...,1],s[...,2]
    m=COL[color](R,G,B)
    (px1,v1),(px2,v2)=xmap; (py1,w1),(py2,w2)=ymap
    out=[]
    for xv in xs:
        px=px1+(xv-v1)*(px2-px1)/(v2-v1)
        c=int(round(px))
        ys=np.where(m[:,c-bandw:c+bandw+1].any(axis=1))[0]
        runs=[];cur=[]
        for y in ys:
            if cur and y-cur[-1]>2: runs.append(cur);cur=[]
            cur.append(y)
        if cur: runs.append(cur)
        cand=[(r[0],r[-1]) for r in runs if 5<=r[-1]-r[0]<=24]
        if not cand: cand=[(r[0],r[-1]) for r in runs]
        if not cand: out.append((xv,None,[])); continue
        if inset and inset[0]<=xv<=inset[1]:
            pick=max(cand,key=lambda r:r[1])
        else:
            pick=max(cand,key=lambda r:r[1]-r[0])
        yc=(pick[0]+pick[1])/2
        val=w1+(yc-py1)*(w2-w1)/(py2-py1)
        out.append((xv,round(val,2),[(round(w1+((a+b)/2-py1)*(w2-w1)/(py2-py1),2)) for a,b in cand]))
    return out
if __name__=='__main__':
    cfg=json.loads(sys.argv[1])
    for color in cfg['colors']:
        res=run(cfg['img'],cfg['box'],cfg['xmap'],cfg['ymap'],cfg['xs'],color,cfg.get('inset'),cfg.get('bandw',1))
        print(color)
        for r in res: print('  ',r)
