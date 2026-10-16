"""Toy check of audit hypotheses H1-H3 on tonal FxLMS with a single lightly damped mode near f0.
H1: slow branch t5 = 2 ln20 / (fs mu L A^2 |Gs|^2)   (exact model Gc=Gs)
H2: small-L factor in [L/(L+|S|), L/(L-|S|)], |S| = |sin(L w)/sin w|
H3: optimal adaptation rate alpha_opt*fs ~ sigma_n = zeta*wn and t5_min ~ c*ln20/sigma_n
Not the paper's plant: a synthetic 1-mode plant with fn=117.07 Hz, f0=120 Hz, fs=10 kHz."""
import numpy as np, numba, sys
from scipy import signal

fs = 1e4

def mode_tf(fn, zeta, gain):
    wn = 2*np.pi*fn
    num, den = [gain*wn**2], [1, 2*zeta*wn, wn**2]
    bd, ad, _ = signal.cont2discrete((num, den), 1/fs, method='bilinear')
    return np.ravel(bd), np.ravel(ad)

@numba.njit(cache=True)
def run(x, bp, ap, bs, as_, L, mu, k_on, sgn_update, sgn_junction):
    n = x.size
    e = np.zeros(n); W = np.zeros(L); xb = np.zeros(L); rb = np.zeros(L)
    # IIR states (direct form, order 2)
    dp_x = np.zeros(3); dp_y = np.zeros(3)
    s_x = np.zeros(3); s_y = np.zeros(3)
    c_x = np.zeros(3); c_y = np.zeros(3)
    for k in range(n):
        # primary path
        dp_x[2] = dp_x[1]; dp_x[1] = dp_x[0]; dp_x[0] = x[k]
        d = (bp[0]*dp_x[0] + bp[1]*dp_x[1] + bp[2]*dp_x[2] - ap[1]*dp_y[0] - ap[2]*dp_y[1]) / ap[0]
        dp_y[1] = dp_y[0]; dp_y[0] = d
        # reference buffer
        for i in range(L-1, 0, -1):
            xb[i] = xb[i-1]; rb[i] = rb[i-1]
        xb[0] = x[k]
        # filtered reference r = Gc*x (Gc = Gs exact)
        c_x[2] = c_x[1]; c_x[1] = c_x[0]; c_x[0] = x[k]
        r = (bs[0]*c_x[0] + bs[1]*c_x[1] + bs[2]*c_x[2] - as_[1]*c_y[0] - as_[2]*c_y[1]) / as_[0]
        c_y[1] = c_y[0]; c_y[0] = r
        rb[0] = r
        y = 0.0
        if k >= k_on:
            for i in range(L): y += W[i]*xb[i]
        s_x[2] = s_x[1]; s_x[1] = s_x[0]; s_x[0] = y
        ys = (bs[0]*s_x[0] + bs[1]*s_x[1] + bs[2]*s_x[2] - as_[1]*s_y[0] - as_[2]*s_y[1]) / as_[0]
        s_y[1] = s_y[0]; s_y[0] = ys
        ek = d + sgn_junction*ys
        e[k] = ek
        if k >= k_on:
            for i in range(L): W[i] += sgn_update*2*mu*ek*rb[i]
            if abs(W[0]) > 1e6:
                for j in range(k, n): e[j] = np.inf
                break
    return e

def t5(e, k_on, f0):
    blk = int(round(fs/f0))  # 1 period
    nb = (e.size - k_on)//blk
    env = np.array([np.abs(e[k_on+i*blk:k_on+(i+1)*blk]).max() for i in range(nb)])
    a0 = np.abs(e[k_on-blk:k_on]).max()
    below = env < 0.05*a0
    if not np.isfinite(env).all(): return np.inf, np.inf
    if not below.any(): return np.inf, np.inf
    first = np.argmax(below)
    # last crossing above
    above = np.where(~below)[0]
    stay = (above.max()+1) if above.size and above.max()+1 < nb else (np.inf if above.size and above.max()+1 >= nb else 0)
    return (first+1)*blk/fs, (stay+1)*blk/fs if np.isfinite(stay) else np.inf

def setup(zeta, fn=117.07, f0=120.0, T_pre=20.0, T_post=20.0, A=1.0):
    bp, ap = mode_tf(fn, zeta, 1.0)
    bs, as_ = mode_tf(fn, zeta, 0.6)
    w = 2*np.pi*f0/fs
    Gs = np.polyval(bs[::-1], np.exp(-1j*w)) / np.polyval(as_[::-1], np.exp(-1j*w))
    n = int((T_pre+T_post)*fs); k_on = int(T_pre*fs)
    x = A*np.cos(w*np.arange(n))
    return x, bp, ap, bs, as_, k_on, abs(Gs), w

if __name__ == '__main__':
    f0 = 120.0
    # sign test (H0): paper's literal Eq.(18) '-' junction with Eq.(20) '-' update
    x, bp, ap, bs, as_, k_on, gs, w = setup(0.0017)
    for sj, su, lab in [(+1, -1, "'+' junction, '-' update (Elliott)"), (-1, -1, "'-' junction (Eq.18) + '-' update (Eq.20) literal"), (-1, +1, "'-' junction, '+' update (Widrow/Kuo)")]:
        e = run(x, bp, ap, bs, as_, 128, 2e-4/(128*gs**2), k_on, su, sj)
        a0 = np.abs(e[k_on-400:k_on]).max(); a1 = np.abs(e[-4000:]).max()
        print(f'SIGN {lab:55s}: final/initial amplitude = {a1/a0:.3g}')
    print('|Gs(w0)|=', gs)
    A = 1.0
    for zeta in [0.0017, 0.0034, 0.0068]:
        x, bp, ap, bs, as_, k_on, gs, w = setup(zeta)
        sigma = zeta*2*np.pi*117.07
        rows = []
        for a_fs in np.geomspace(0.1*sigma, 8*sigma, 26):
            alpha = a_fs/fs
            for L in [128]:
                mu = 2*alpha/(L*A**2*gs**2)
                e = run(x, bp, ap, bs, as_, L, mu, k_on, -1.0, 1.0)
                tf, ts = t5(e, k_on, f0)
                rows.append((a_fs, tf, ts, np.log(20)/a_fs))
        rows = np.array(rows)
        i = np.nanargmin(np.where(np.isfinite(rows[:,1]), rows[:,1], np.nan))
        print(f'zeta={zeta:.4f} sigma={sigma:.3f}/s : alpha_opt*fs={rows[i,0]:.3f} (ratio to sigma {rows[i,0]/sigma:.2f}), t5_min={rows[i,1]:.3f}s, t5_min*sigma/ln20={rows[i,1]*sigma/np.log(20):.2f}')
        for r in rows[::3]:
            print(f'   alpha*fs={r[0]:7.3f}  t5={r[1]:7.3f}  t5(stay)={r[2]:7.3f}  H1-pred={r[3]:7.3f}')
    # H2: small-L split
    x, bp, ap, bs, as_, k_on, gs, w = setup(0.0017)
    sigma = 0.0017*2*np.pi*117.07
    a_fs = 0.3*sigma
    x2, _, _, _, _, k_on2, _, _ = setup(0.0017, T_post=60.0)
    print('H2 small-L test at alpha*fs =', a_fs)
    tref = None
    for L in [128, 64, 48, 40, 32, 24, 16]:
        mu = 2*(a_fs/fs)/(L*A**2*gs**2)
        e = run(x2, bp, ap, bs, as_, L, mu, k_on2, -1.0, 1.0)
        tf, ts = t5(e, k_on2, f0)
        S = abs(np.sin(L*w)/np.sin(w))
        if tref is None: tref = tf
        print(f'  L={L:4d} t5={tf:.3f}  ratio={tf/tref:.3f}  bounds=[{L/(L+S):.3f},{L/max(L-S,1e-9):.3f}]')
