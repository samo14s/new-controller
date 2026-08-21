"""
run_stage78.py — STAGES 7 and 8: comparison and stress tests.
==============================================================
Identical conditions for every controller: same plant, actuator, sensor, sign,
cutting parameters, initial conditions, horizon, time step and disturbance.
Only the controller changes.

    python run_stage78.py [1 2 3 4]
Writes results/stage78.pkl and results/log_stage78.txt
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from eval2 import limits
from plate_model import build_plate
from plant_ss import ControlledPlant
from scenarios import _perturbed, mode_scale_from_removal, paper_box_scale
from simulate import MillingSimulation
from sim_ctrl2 import ScheduledLTI
from stage_common import load_controllers, LABEL, n_params
from uncertainty import RemovalFamily

OUT = C.RESULTS
LOG = open(os.path.join(OUT, 'log_stage78.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


# ---------------------------------------------------------------------------
def time_run(plate, plant, ctrl, ap=None, mode_scale=None, zeta_scale=1.0,
             T=None, moving=True):
    ap = C.AP_S if ap is None else ap
    sim = MillingSimulation(plate, C.RPM_S, ap, ae=C.AE, fz=C.FZ, sign=C.SIGN,
                            n_modes=C.N_MODES, n_sub=C.N_SUB_TIME,
                            mode_scale=mode_scale, zeta_scale=zeta_scale)
    ss, _ = ctrl.at(0.0)
    if ss is None:
        c = None
    else:
        c = ScheduledLTI(ctrl, sim.dt, plate.lp, tau=sim.tau,
                         feed=plant.feed_speed(), moving=moving)
    r = sim.run(controller=c, T=T, moving=moving)
    y = np.asarray(r['y_obs'], float)
    u = np.asarray(r['u'], float)
    return dict(A_max=float(np.max(np.abs(y))) * 1e6,
                A_rms=float(np.sqrt(np.mean(y ** 2))) * 1e6,
                E_u=float(np.sum(u ** 2) * r['dt']),
                u_max=float(np.max(np.abs(u))),
                u_mean=float(np.mean(np.abs(u[int(0.05 * u.size):]))),
                diverged=bool(r['diverged']), t_div=r['t_div'])


def settling(plate, plant, ctrl, y0_um=10.0, T=0.4, band=0.05):
    """Settling time of the FREE response: a milling pass is forced
    continuously, so the settling time of the cut is not a controller property.
    """
    sim = MillingSimulation(plate, C.RPM_S, 1e-9, ae=C.AE, fz=0.0,
                            sign=C.SIGN, n_modes=C.N_MODES,
                            n_sub=C.N_SUB_TIME)
    ss, _ = ctrl.at(0.0)
    c = None if ss is None else ScheduledLTI(ctrl, sim.dt, plate.lp,
                                             tau=sim.tau, moving=False)
    n, dt = sim.n, sim.dt
    D = sim.D_obs
    q = D / float(D @ D) * (y0_um * 1e-6)
    qd = np.zeros(n)
    qdd = np.zeros(n)
    g, b = sim.g, sim.b
    nstep = int(round(T / dt)) + 1
    y = np.zeros(nstep)
    us = np.zeros(nstep)
    for k in range(1, nstep):
        yk = float(D @ q)
        ydk = float(D @ qd)
        y[k - 1] = yk
        u = 0.0 if c is None else float(c(y=yk, yd=ydk, t=k * dt, k=k))
        us[k] = u
        qdp = qd + (1 - g) * dt * qdd
        qp = q + dt * qd + (0.5 - b) * dt ** 2 * qdd
        qdd = sim.S0inv @ (sim.H * u - sim.C @ qdp - sim.K @ qp)
        qd = qdp + g * dt * qdd
        q = qp + b * dt ** 2 * qdd
    y[-1] = float(D @ q)
    a = np.abs(y)
    idx = np.where(a > band * float(a.max()))[0]
    return dict(T_s=float(idx[-1] * dt) if idx.size else 0.0,
                E_u=float(np.sum(us ** 2) * dt))


# ---------------------------------------------------------------------------
def s1(plate, plant, ctrls, store):
    log('\n' + '=' * 96)
    log(f'STAGE 7 - NOMINAL COMPARISON  ({C.RPM_S} rpm, a_e = {C.AE*1e3} mm, '
        f'a_p = {C.AP_S*1e3} mm)')
    log('=' * 96)
    log(f'{"controller":<18}{"par":>4}{"a_p,lim min":>13}{"mean":>9}'
        f'{"A_max um":>10}{"A_rms um":>10}{"T_s ms":>9}{"E_u V2s":>10}'
        f'{"u_max V":>9}')
    for k, mk in ctrls.items():
        t = time.time()
        c = mk(plant)
        L = limits(plate, c)
        tm = time_run(plate, plant, c)
        st = settling(plate, plant, c)
        log(f'{LABEL.get(k,k):<18}{n_params(k):>4}{L.min()*1e3:>13.4f}'
            f'{L.mean()*1e3:>9.4f}{tm["A_max"]:>10.3f}{tm["A_rms"]:>10.3f}'
            f'{st["T_s"]*1e3:>9.1f}{tm["E_u"]:>10.4g}{tm["u_max"]:>9.2f}'
            + ('  DIVERGED' if tm['diverged'] else '')
            + f'   [{time.time()-t:.0f}s]')
        store[f'S1_{k}'] = dict(limits=L, **tm, **{'T_s': st['T_s']})


def s2(plate, plant, ctrls, store, n_pos=11):
    log('\n' + '=' * 96)
    log('STAGE 8(a) - POSITION SWEEP   a_p,lim (mm) along the top edge')
    log('=' * 96)
    fr = np.linspace(0.0, 1.0, n_pos)
    log(f'{"controller":<18}' + ''.join(f'{f"{v*100:.0f}":>7}' for v in fr)
        + f'{"min":>9}{"spread":>9}')
    for k, mk in ctrls.items():
        c = mk(plant)
        L = limits(plate, c, positions=fr)
        log(f'{LABEL.get(k,k):<18}' + ''.join(f'{v*1e3:>7.3f}' for v in L)
            + f'{L.min()*1e3:>9.4f}{(L.max()-L.min())*1e3:>9.4f}')
        store[f'S2_{k}'] = L
    store['S2_positions'] = fr


def s3(plate, plant, ctrls, store):
    log('\n' + '=' * 96)
    log('STAGE 8(b) - MODAL PARAMETER VARIATION   a_p,lim (mm), min over positions')
    log('=' * 96)
    rem = RemovalFamily(n=C.N_MODES_DESIGN)
    cases = [('eta=1/2 max', mode_scale_from_removal(rem, C.ETA_MAX / 2,
                                                     C.N_MODES), 1.0, C.ETA_MAX / 2),
             ('eta=max', mode_scale_from_removal(rem, C.ETA_MAX, C.N_MODES),
              1.0, C.ETA_MAX),
             ('box -10%', paper_box_scale(-0.10, 0.10, C.N_MODES), 1.0, 0.0),
             ('box +10%', paper_box_scale(0.10, -0.10, C.N_MODES), 1.0, 0.0),
             ('zeta x0.8', None, 0.8, 0.0),
             ('zeta x1.2', None, 1.2, 0.0)]
    log(f'{"controller":<18}' + ''.join(f'{n:>13}' for n, _, _, _ in cases)
        + f'{"worst":>9}')
    for k, mk in ctrls.items():
        c = mk(plant)
        row = []
        for _, ms, zs, eta in cases:
            pl2 = _perturbed(plate, ms, zs)
            row.append(limits(pl2, c, eta=eta).min() * 1e3)
        log(f'{LABEL.get(k,k):<18}' + ''.join(f'{v:>13.4f}' for v in row)
            + f'{min(row):>9.4f}')
        store[f'S3_{k}'] = np.array(row)
    store['S3_cases'] = [n for n, _, _, _ in cases]


def s4(plate, plant, ctrls, store):
    log('\n' + '=' * 96)
    log('STAGE 8(c) - INCREASED DELAY   a_p,lim (mm) at tau/tau_0')
    log('=' * 96)
    ratios = (1.0, 1.25, 1.5, 2.0, 3.0)
    log(f'{"controller":<18}' + ''.join(f'{f"{r:.2f}t0":>11}' for r in ratios)
        + f'{"worst":>9}')
    for k, mk in ctrls.items():
        row = []
        for r in ratios:
            rpm = C.RPM_S / r
            p2 = ControlledPlant(plate, rpm=rpm)
            row.append(limits(plate, mk(p2), rpm=rpm).min() * 1e3)
        log(f'{LABEL.get(k,k):<18}' + ''.join(f'{v:>11.4f}' for v in row)
            + f'{min(row):>9.4f}')
        store[f'S4_{k}'] = np.array(row)
    store['S4_ratios'] = np.array(ratios)


def main(which=(1, 2, 3, 4)):
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    plant = ControlledPlant(plate)
    ctrls = load_controllers(plate, plant)
    p = os.path.join(OUT, 'stage78.pkl')
    store = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
    for w, fn in ((1, s1), (2, s2), (3, s3), (4, s4)):
        if w in which:
            fn(plate, plant, ctrls, store)
    with open(p, 'wb') as f:
        pickle.dump(store, f)
    log(f'\ntotal {time.time()-t0:.0f}s -> results/stage78.pkl')


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4])
