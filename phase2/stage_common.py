"""stage_common.py — rebuild the designed controllers from disk."""
import os
import pickle

import numpy as np

import config as C
import ctrl2 as K2

ORDER = ('open', 'fopid', 'lqg', 'mu_tdc', 'ps_ac', 'ps_ac_eta',
         'ps_ac_obs', 'ps_ac_full')
LABEL = dict(open='no control', fopid='FOPID', lqg='LQG',
             mu_tdc='mu-TDC (Du 2024)', ps_ac='PS-AC (proposed)',
             ps_ac_eta='PS-AC + eta', ps_ac_obs='PS-AC obs only',
             ps_ac_full='PS-AC K+obs')


def load_mu_ss():
    p = os.path.join(C.RESULTS, 'musyn_phase2.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if 'mu_paper_ss0' not in d.files:
        return None
    return tuple(d[f'mu_paper_ss{i}'] for i in range(4))


def load_controllers(plate, plant_ref=None, include_open=True):
    """{name: factory(plant) -> Ctrl}.

    A factory rather than an object, because a controller must be rebuilt for
    each plant (the design depends on a_p through alpha_40).
    """
    with open(os.path.join(C.RESULTS, 'stage3_controllers.pkl'), 'rb') as f:
        store = pickle.load(f)
    ss_mu = load_mu_ss()
    out = {}
    if include_open:
        out['open'] = lambda p: K2.Ctrl('open', 0)
    for kind in ('fopid', 'lqg', 'mu_tdc', 'ps_ac', 'ps_ac_eta',
                 'ps_ac_obs', 'ps_ac_full'):
        if kind not in store:
            continue
        u = dict(store[kind]['params'])

        def mk(p, kind=kind, u=u):
            return K2.build(kind, p, u, ss_mu)

        out[kind] = mk
    return out


def n_params(name):
    return dict(open=0, fopid=5, lqg=4, mu_tdc=6, ps_ac=4, ps_ac_eta=5,
                ps_ac_obs=4, ps_ac_full=4).get(name, 0)
