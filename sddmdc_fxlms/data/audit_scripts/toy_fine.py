import numpy as np, sys
sys.path.insert(0, '.')
from toy_fxlms import setup, run, t5, fs
A = 1.0
for zeta, fn in [(0.0017, 117.07), (0.0034, 117.07), (0.0068, 117.07)]:
    x, bp, ap, bs, as_, k_on, gs, w = setup(zeta, fn=fn, T_post=30.0)
    sigma = zeta*2*np.pi*fn
    best = (np.inf, None); amax = None
    out = []
    for r in np.linspace(0.4, 2.0, 17):
        alpha = r*sigma/fs
        mu = 2*alpha/(128*A**2*gs**2)
        e = run(x, bp, ap, bs, as_, 128, mu, k_on, -1.0, 1.0)
        tf, ts = t5(e, k_on, 120.0)
        out.append((r, tf, ts))
        if ts < best[0]: best = (ts, r)
        if not np.isfinite(ts) and amax is None and r > 0.6: amax = r
    print(f'zeta={zeta}: sigma={sigma:.3f}  alpha_opt/sigma={best[1]:.2f}  t5min(stay)={best[0]:.3f}s  t5min*sigma/ln20={best[0]*sigma/np.log(20):.2f}  first non-converging alpha/sigma={amax}')
    print('   ' + ' '.join(f'{r:.1f}:{tf:.2f}/{ts:.2f}' for r, tf, ts in out))
