"""
musyn.py — controller 4: the paper's own mu-synthesis controller.
==================================================================
Wraps the D-K iteration archived in baseline/control_old (Eqs. 26-29 of Du et
al. 2024) so that it can be scored by exactly the same objective as the other
four structures.

Two variants are produced:

    'paper' : the uncertainty set exactly as published -- Eq. (25) with the
              cross terms dropped, and the flat 10 % / 20 % box;
    'exact' : the same synthesis machinery on the CORRECTED envelope
              (alpha_coupling='exact'), i.e. Eq. (25) with the two cross terms
              restored.  The difference between the two is the price of the
              algebraic slip, measured rather than asserted.
"""
import numpy as np

import config as C
from fopid import rolloff_ss, series


def design(plate, rpm=None, ap=None, alpha_coupling='paper', n_iter=3,
           reduce_to=12, verbose=False, mass_pert=0.10, stiff_pert=0.10,
           damp_pert=0.20):
    from robust_design import design_robust
    import uncertain_plant as up

    rpm = C.RPM_S if rpm is None else rpm
    ap = C.AP_S if ap is None else ap
    # the perturbation sizes live in uncertain_plant.nominal_and_perturbations;
    # patch the defaults so the same machinery can be driven with the
    # physics-based bounds instead of the published flat ones
    orig = up.nominal_and_perturbations

    def patched(pl, r, a, ae=C.AE, n_modes=2, alpha_coupling='paper',
                mass_pert=mass_pert, stiff_pert=stiff_pert,
                damp_pert=damp_pert, n_pos=201):
        return orig(pl, r, a, ae, n_modes, alpha_coupling, mass_pert,
                    stiff_pert, damp_pert, n_pos)

    up.nominal_and_perturbations = patched
    try:
        res = design_robust(plate, rpm, ap, ae=C.AE,
                            alpha_coupling=alpha_coupling, n_iter=n_iter,
                            orders=(0, 1, 1, 1), reduce_to=reduce_to,
                            verbose=verbose, sign=C.SIGN)
    finally:
        up.nominal_and_perturbations = orig
    ss = series(res['ss'], rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
    return dict(ss=ss, mu=float(res['mu']), gamma=float(res['gamma']),
                order=ss[0].shape[0], raw=res)


# ---------------------------------------------------------------------------
def design_phys(plate, rpm=None, ap=None, n_iter=3, reduce_to=12,
                verbose=False, f_grid=None, **set_kw):
    """Same D-K iteration (Eqs. 28-29), physics-based uncertainty set.

    Everything the paper fixes is kept byte-for-byte: the additive weights of
    Eqs. (18)-(19), the synthesis weights, the spectral shift alpha = 12 rad/s,
    the plant scaling, the frequency grid, the D-scale orders and the balanced
    controller reduction.  `set_kw` selects the uncertainty description:

        one_sided=True/False   eta in [0, eta_max]  vs  a symmetric box
        correlated=True/False  ONE parameter for M, C, K  vs  three
        rank_one=True/False    W = a d d^T  vs  four free entries (Eq. 25)

    so each structural fact can be switched off on its own and its contribution
    measured, rather than asserted.
    """
    import numpy as _np
    from dk_synthesis import (prepare_plant, unscale_controller, dk_iteration,
                              reduce_controller)
    from robust_design import freq_grid
    from uncertain_phys import PhysUncertainSystem

    rpm = C.RPM_S if rpm is None else rpm
    ap = C.AP_S if ap is None else ap
    U = PhysUncertainSystem(plate, rpm, ap, ae=C.AE, n_modes=C.N_MODES_DESIGN,
                            sign=C.SIGN, **set_kw)
    P = U.generalized_plant()
    Ps, info = prepare_plant(P, alpha_shift=12.0)
    f = freq_grid() if f_grid is None else _np.asarray(f_grid, float)
    res = dk_iteration(Ps, P['blocks'], 2 * _np.pi * f / info['w0'],
                       n_iter=n_iter, orders=(0, 1, 1, 1), verbose=verbose)
    K_full = unscale_controller(res['K'], info)
    K = reduce_controller(K_full, reduce_to) if reduce_to else K_full
    ss = series((_np.array(K.A), _np.array(K.B), _np.array(K.C),
                 _np.array(K.D)), rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER))
    return dict(ss=ss, mu=float(res['mu']), gamma=float(res['gamma']),
                order=ss[0].shape[0], U=U, P=P, info=info, f=f,
                mus=res['mus'], history=res['history'], K=K)
