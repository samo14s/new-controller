"""stage_common.py — rebuild the designed controllers from disk."""
import os
import pickle

import numpy as np

import config as C
import ctrl2 as K2

ORDER = ('open', 'fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac',
         'ps_ac_eta', 'ps_ac_obs', 'ps_ac_full')
LABEL = dict(open='no control', fopid='FOPID', lqg='LQG',
             mu_tdc='mu-TDC (Du 2024)',
             mu_phys_tdc='mu-TDC on the physics set',
             ps_ac='PS-AC (proposed)',
             ps_ac_eta='PS-AC + eta', ps_ac_obs='PS-AC obs only',
             ps_ac_full='PS-AC K+obs', ps_tdc='PS-TDC (frozen base)',
             ps_tdc_j='PS-TDC (joint)',
             ps_ac_r='PS-AC-R (envelope design)',
             ps_tdc_r='PS-TDC-R (envelope + pair)',
             ps_rob='PS-ROB (floored observer)',
             ps_rob_tdc='PS-ROB-TDC (+ pair)')


def load_mu_ss(tag='mu_paper'):
    """State space of a mu-synthesis controller, by uncertainty description."""
    fname = ('musyn_phase2.npz' if tag in ('mu_paper', 'mu_exact')
             else 'musyn_phys.npz')
    p = os.path.join(C.RESULTS, fname)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if f'{tag}_ss0' not in d.files:
        return None
    return tuple(d[f'{tag}_ss{i}'] for i in range(4))


def load_controllers(plate, plant_ref=None, include_open=True):
    """{name: factory(plant) -> Ctrl}.

    A factory rather than an object, because a controller must be rebuilt for
    each plant (the design depends on a_p through alpha_40).
    """
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'rb') as f:
        store = pickle.load(f)
    ss = dict(mu_tdc=load_mu_ss('mu_paper'),
              mu_phys_tdc=load_mu_ss('mu_phys'))
    if 'ps_ac' in store:            # ps_tdc rides on the stored ps_ac gains
        ss['ps_tdc'] = dict(store['ps_ac']['params'])
    if 'ps_ac_r' in store:          # ps_tdc_r on the stored envelope base
        ss['ps_tdc_r'] = dict(store['ps_ac_r']['params'])
    out = {}
    if include_open:
        out['open'] = lambda p: K2.Ctrl('open', 0)
    for kind in ('fopid', 'lqg', 'mu_tdc', 'mu_phys_tdc', 'ps_ac',
                 'ps_ac_eta', 'ps_ac_obs', 'ps_ac_full', 'ps_tdc',
                 'ps_tdc_j', 'ps_ac_r', 'ps_tdc_r'):
        if kind not in store:
            continue
        if kind in ss and ss[kind] is None:
            continue
        u = dict(store[kind]['params'])

        def mk(p, kind=kind, u=u, s=ss.get(kind)):
            return K2.build(kind, p, u, s)

        out[kind] = mk
    return out


def n_params(name):
    return dict(open=0, fopid=5, lqg=4, mu_tdc=6, mu_phys_tdc=6, ps_ac=4,
                ps_ac_eta=5, ps_ac_obs=4, ps_ac_full=4,
                ps_tdc=6, ps_tdc_j=6, ps_ac_r=5,
                ps_tdc_r=7, ps_rob=6, ps_rob_tdc=8).get(name, 0)
