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
