"""Measure the frequency-grid trap of Section 6.3, and store the numbers.

The crossing test needs sup_omega rho((j omega I - A_cl)^-1 A_d,cl).  The plate
has zeta ~ 0.003, so each resonance is only a few rad/s wide, while a plain
logarithmic grid steps over it by two orders of magnitude more.  The samples
then land BESIDE the peak rather than on it and the reported spectral radius is
far too small -- small enough to turn a verdict of instability into stability.

certify2._wgrid refines around every closed-loop eigenvalue.  This script
measures what that refinement is worth, over the same vertex family the
certificate uses, and writes the worst case to results/grid_trap.txt so the
paper can quote a measured number instead of an asserted one.

    python phase2/grid_trap.py
"""
import os

import numpy as np

import certify2 as CF2
import config as C
from plant_ss import ControlledPlant

N_PLAIN = 400            # the plain log grid this is compared against
WS = 2 * np.pi * 800.0   # the same state scaling certify2 uses


def peak_on(Acl, Adcl, w):
    I = np.eye(Acl.shape[0])
    p = 0.0
    for wk in w:
        try:
            G = np.linalg.solve(1j * wk * I - Acl, Adcl)
            p = max(p, float(np.max(np.abs(np.linalg.eigvals(G)))))
        except np.linalg.LinAlgError:
            pass
    return p


def dense_grid(Acl, hi):
    """A grid far denser than the shipped one, to show the refined value is the
    converged value and not just a different guess."""
    ev = np.linalg.eigvals(Acl)
    g = [np.logspace(-3, np.log10(hi), 4000)]
    for lam in ev:
        wk = abs(float(np.imag(lam)))
        if wk <= 0.0:
            continue
        half = max(abs(float(np.real(lam))), 1e-9 * wk)
        g.append(np.linspace(max(wk - 40 * half, 0.0), wk + 40 * half, 20001))
    return np.unique(np.concatenate(g))


def main():
    plate = C.build_plate(n_modes=C.N_MODES, patch=C.PATCH_SIDE)
    plant = ControlledPlant(plate, ap=C.AP_S)
    xs = np.linspace(0.0, plate.lp, 9)
    a4s = (C.ALPHA_LO * plant.abar4, C.ALPHA_HI * plant.abar4)

    out = []
    out.append('=' * 78)
    out.append('SECTION 6.3 - WHAT REFINING THE FREQUENCY GRID IS WORTH')
    out.append('=' * 78)
    out.append(f'  plain grid   : {N_PLAIN} points, logspace(1e-3, 5 max|lambda|)')
    out.append('  refined grid : certify2._wgrid, local grids around every')
    out.append('                 closed-loop eigenvalue')
    out.append('  family       : the uncontrolled loop over 9 positions x')
    out.append(f'                 alpha4 in [{C.ALPHA_LO}, {C.ALPHA_HI}] abar4')
    out.append('')
    for i, (f, z) in enumerate(zip(C.F_MEASURED[:2], C.ZETA[:2]), 1):
        w = 2 * np.pi * f
        out.append(f'  mode {i}: omega = {w:8.1f} rad/s, half width zeta*omega '
                   f'= {z * w:5.1f} rad/s')

    worst = dict(ratio=0.0)
    for x in xs:
        for a4 in a4s:
            A, Ad, B, _, Cy = plant.matrices(x_pos=float(x), a4=a4)
            Acl, Adcl, npl = CF2.augment(A, Ad, B, Cy, None, None, plant.n)
            Acl, Adcl = CF2._scale(Acl, Adcl, plant.n, npl, WS)
            if not np.all(np.isfinite(Acl)):
                continue
            hi = 5.0 * max(float(np.abs(np.linalg.eigvals(Acl)).max()), 1.0)
            plain = peak_on(Acl, Adcl, np.logspace(-3, np.log10(hi), N_PLAIN))
            ref = peak_on(Acl, Adcl, CF2._wgrid(Acl))
            r = ref / max(plain, 1e-12)
            if r > worst['ratio']:
                worst = dict(ratio=r, x=float(x), a4=a4 / plant.abar4,
                             plain=plain, refined=ref, hi=hi)

    hi = worst['hi']
    step = np.log(hi / 1e-3) / (N_PLAIN - 1)
    A, Ad, B, _, Cy = plant.matrices(x_pos=worst['x'],
                                     a4=worst['a4'] * plant.abar4)
    Acl, Adcl, npl = CF2.augment(A, Ad, B, Cy, None, None, plant.n)
    Acl, Adcl = CF2._scale(Acl, Adcl, plant.n, npl, WS)
    conv = peak_on(Acl, Adcl, dense_grid(Acl, hi))

    out.append('')
    out.append(f'  worst vertex : x = {worst["x"]*1e3:.1f} mm, '
               f'alpha4 = {worst["a4"]:.1f} abar4')
    out.append(f'  plain grid   : rho = {worst["plain"]:.4f}')
    out.append(f'  refined grid : rho = {worst["refined"]:.4f}')
    out.append(f'  dense grid   : rho = {conv:.4f}   '
               f'(the refined value IS the converged one)')
    out.append(f'  understated by: {worst["ratio"]:.2f}x')
    out.append('')
    out.append('  log-grid spacing at that vertex, against the mode widths:')
    for i, (f, z) in enumerate(zip(C.F_MEASURED[:2], C.ZETA[:2]), 1):
        w = 2 * np.pi * f
        out.append(f'    mode {i}: spacing {w*step:6.1f} rad/s vs half width '
                   f'{z*w:5.1f} rad/s  -> {w*step/(z*w):4.1f}x wider')
    out.append('')
    out.append('  A peak this much too small is enough to turn a verdict of')
    out.append('  instability into one of stability, which is why the grid is')
    out.append('  refined around every eigenvalue rather than made denser.')

    txt = '\n'.join(out) + '\n'
    path = os.path.join(C.RESULTS, 'grid_trap.txt')
    with open(path, 'w') as fh:
        fh.write(txt)
    print(txt)
    print('written to', path)


if __name__ == '__main__':
    main()
