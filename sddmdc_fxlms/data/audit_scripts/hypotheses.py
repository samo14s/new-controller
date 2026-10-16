"""Compute audit hypothesis tables from digitized paper data (research/01_audit_digitized.csv)."""
import numpy as np, csv, io, sys
path = sys.argv[1]
rows = [r for r in csv.reader(l for l in open(path) if not l.startswith('#'))]
data = {}
for fig, case, sweep, fixed, x, te, ts in rows:
    data.setdefault((fig, case), []).append((float(x), float(te) if te else np.nan, float(ts) if ts else np.nan))
fs = 1e4
def interp_mu(fig, case, mu, col):
    a = np.array(data[(fig, case)]); i = np.argsort(a[:, 0])
    # interpolate in t*mu (smooth) on log axis
    return np.interp(mu, a[i, 0], a[i, col])
print('== H2 slow branch t*mu (x1e-6 s) ==')
for (fig, case), L in [(('16c', '120Hz-S2'), 128), (('19c', '280Hz-S1'), 128), (('21c', '85Hz-S2-2ch'), 64)]:
    a = np.array(data[(fig, case)])
    for x, te, ts in a:
        print(f'{case:12s} mu={x:.1e}  t*mu exp={te*x*1e6:6.2f}  sim={ts*x*1e6:6.2f}')
print('== H1 collapse: L-sweep vs mu-sweep at equal mu*L ==')
w = lambda f: 2*np.pi*f/fs
for (figL, case, muL_fixed, Lref, figmu, f) in [('16d', '120Hz-S2', 1e-5, 128, '16c', 120), ('19d', '280Hz-S1', 1.2e-5, 128, '19c', 280), ('21d', '85Hz-S2-2ch', 6e-6, 64, '21c', 85)]:
    for L, te, ts in data[(figL, case)]:
        mu_eq = muL_fixed*L/Lref
        a = np.array(data[(figmu, case)]); lo, hi = a[:, 0].min(), a[:, 0].max()
        S = abs(np.sin(L*w(f))/np.sin(w(f)))
        if lo <= mu_eq <= hi:
            ce, cs = interp_mu(figmu, case, mu_eq, 1), interp_mu(figmu, case, mu_eq, 2)
            print(f'{case:12s} L={L:4.0f} muL={muL_fixed*L:.2e} tL(exp/sim)={te:5.2f}/{ts:5.2f} tmu(exp/sim)={ce:5.2f}/{cs:5.2f} ratio={te/ce:4.2f}/{ts/cs:4.2f}  split-bounds=[{L/(L+S):.2f},{L/max(L-S,1e-9):.2f}]')
        else:
            print(f'{case:12s} L={L:4.0f} muL={muL_fixed*L:.2e} (outside mu-sweep range)  split-bounds=[{L/(L+S):.2f},{L/max(L-S,1e-9):.2f}]')
