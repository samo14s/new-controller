"""
test_phys_lft.py — does the physics-based LFT actually contain the physics?
============================================================================
For every point of the physical set an EXPLICIT witness delta is constructed
(by inverting each parameterisation), it is checked that |delta| <= 1, and the
LFT state matrix is compared with the true one.  This is containment with a
witness, not a sampled bound.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.set_printoptions(precision=4, suppress=True, linewidth=140)

import config as C
from plate_model import build_plate
from uncertain_phys import PhysUncertainSystem


def witness(U, eta, xi, x_tool, a_mult, z):
    """delta reproducing the physical point (eta, xi, x, a, z)."""
    h = U.eta_max / 2.0 if U.one_sided else U.eta_max
    e0 = h if U.one_sided else 0.0
    d_eta = (eta - e0) / h
    d_xi = 1.0 - 2.0 * xi        # xi = 1 -> -1 (fit branch), xi = 0 -> +1
    D = U.plate.D_row(x_tool, U.plate.hp)[:U.n]
    Y = D - U.d_c
    B = np.column_stack([U.l1, U.l2])
    c = np.linalg.solve(B, Y)                    # D = d_c + c1 l1 + c2 l2
    d_a = (U.sign * a_mult * U.abar4 - U.a0) / U.La
    d_z = (z - 1.0) / U.zeta_pert
    return dict(d_eta=d_eta, d_xi=d_xi, d_D=c[0], d_r=c[1], d_a=d_a,
                d_z1=d_z, d_z2=d_z)


def main():
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    U = PhysUncertainSystem(plate, C.RPM_S, C.AP_S, ae=C.AE, sign=C.SIGN)

    print('=' * 74)
    print('PHYSICS-BASED LFT — structure and containment')
    print('=' * 74)
    print(f'  parameters   : {[b[0] for b in U.blocks_used]}')
    print(f'  multiplicity : {[b[1] for b in U.blocks_used]}  '
          f'-> {U.n_par} channels for {len(U.blocks_used)} real parameters')
    print(f'  eta in [0, {U.eta_max:.4f}], one-sided = {U.one_sided}')
    print('\n  quadratic fit of the Ritz integrals in eta:')
    for k, r in U.fit_report.items():
        print(f'    {k}: max residual {r["resid"]:.3e} = '
              f'{100*r["rel_amp"]:.3f} % of the perturbation amplitude, '
              f'{100*r["rel_nom"]:.4f} % of nominal')
    print(f'\n  D(x) locus: singular values {U.geom["sv"]}, '
          f'second/first = {U.geom["cover"]:.4f}')
    print(f'    d_c = {U.d_c},  l1 = {U.l1},  l2 = {U.l2}')
    print('\n  nominal (delta = 0) plate:')
    Minv = np.linalg.inv(U.M_c)
    w = np.sqrt(np.linalg.eigvals(Minv @ U.K_c).real)
    print(f'    M_c = diag{np.diag(U.M_c)},  f_c = {np.sort(w)/(2*np.pi)} Hz')
    print(f'    amplitudes  |M_1| = {np.abs(np.diag(U.M_1))}, '
          f'|K_1|/K_c = {np.abs(np.diag(U.K_1)/np.diag(U.K_c))}, '
          f'|C_1|/C_c = {np.abs(np.diag(U.C_1)/np.diag(U.C_c))}')
    print(f'    |M_x| = {np.abs(np.diag(U.M_x))}, |K_x|/K_c = '
          f'{np.abs(np.diag(U.K_x)/np.diag(U.K_c))}')

    # ---------------------------------------------------------------- witness
    print('\n  containment with an explicit witness:')
    worst_d, worst_err = 0.0, 0.0
    rows = []
    for eta in np.linspace(0.0, U.eta_max, 7):
        for xi in (0.0, 0.5, 1.0):
            for x in np.linspace(0.0, plate.lp, 9):
                for am in (0.3, 1.6, 2.9):
                    for z in (0.8, 1.0, 1.2):
                        d = witness(U, eta, xi, x, am, z)
                        dm = max(abs(v) for v in d.values())
                        st = U.sample(eta, xi, x, am, z)
                        A_true, Bf_true = st[0], st[1]
                        A_lft = U.lft_sample(d)
                        err = (np.abs(A_true - A_lft).max()
                               / np.abs(A_true).max())
                        Bf_lft = U.lft_Bf(d)
                        err = max(err, np.abs(Bf_true - Bf_lft).max()
                                  / np.abs(Bf_true).max())
                        worst_d = max(worst_d, dm)
                        worst_err = max(worst_err, err)
                        rows.append((eta, xi, x, am, z, dm, err))
    print(f'    {len(rows)} physical points')
    print(f'    max |delta|_inf over the witnesses = {worst_d:.4f}  '
          f'({"INSIDE the unit box" if worst_d <= 1.0 + 1e-9 else "OUTSIDE"})')
    print(f'    max relative error on [A, B_f] = {worst_err:.3e}')

    # per-parameter extremes
    arr = np.array(rows)
    print('\n    per-parameter range of the witnesses:')
    for eta in (0.0, U.eta_max):
        d = witness(U, eta, 1.0, plate.lp, 1.6, 1.0)
        print(f'      eta = {eta:.4f} -> d_eta = {d["d_eta"]:+.3f}')
    for x in (0.0, plate.lp / 2, plate.lp):
        d = witness(U, 0.0, 1.0, x, 1.6, 1.0)
        print(f'      x/l = {x/plate.lp:.2f} -> d_D = {d["d_D"]:+.3f}, '
              f'd_r = {d["d_r"]:+.3f}')

    # ------------------------------------------------- size of the two sets
    print('\n  size of the reachable set of state matrices (relative to |A|):')
    import uncertain_plant as up
    Upap = up.MillingUncertainSystem(plate, C.RPM_S, C.AP_S, C.AE, n_modes=2,
                                     alpha_coupling='paper', sign=C.SIGN)
    A0 = Upap.ss_plant['A']
    spp = Upap.ss_plant
    rng = np.random.default_rng(0)
    nu_p = spp['Bunc'].shape[1]
    rad_p = 0.0
    for _ in range(4000):
        dv = rng.choice([-1.0, 1.0], nu_p)
        Dl = np.diag(dv)
        Ap = spp['A'] + spp['Bunc'] @ np.linalg.solve(
            np.eye(nu_p) - Dl @ spp['Dunc_unc'], Dl @ spp['Cunc'])
        rad_p = max(rad_p, np.abs(Ap - A0).max())
    rad_f = 0.0
    A0f = U.ss_plant['A']
    for _ in range(4000):
        d = {nm: rng.choice([-1.0, 1.0]) for nm, _ in U.blocks_used}
        rad_f = max(rad_f, np.abs(U.lft_sample(d) - A0f).max())
    print(f'    paper Eqs. (22)-(25) : max |A(delta) - A(0)| = {rad_p:.4e}')
    print(f'    physics-based        : max |A(delta) - A(0)| = {rad_f:.4e}')
    print(f'    ratio = {rad_f/rad_p:.3f}')

    # generalized plant sanity
    P = U.generalized_plant()
    print(f'\n  generalized plant: A {P["A"].shape}, B {P["B"].shape}, '
          f'C {P["C"].shape}, D {P["D"].shape}')
    print(f'    blocks = {P["blocks"]}')
    print(f'    dscale_init = {P["dscale_init"]}')
    print(f'    conditioning of [C D] rows: '
          f'{np.abs(P["C"]).max(1).max()/max(np.abs(P["C"]).max(1).min(),1e-30):.3e}')


if __name__ == '__main__':
    main()
