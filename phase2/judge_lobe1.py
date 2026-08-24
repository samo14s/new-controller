"""judge_lobe1.py — the post-round battery for the lobe-first (-L) winners.

Stage A of the -L judgement, everything by the standing machinery:

  1. mixed real/complex mu_RS gate at the probed nodes (x = 0, L/2, L),
     actuator block included — judge_mixed, the same judge as every member;
  2. one fresh batch of S1-style recovery metrics (mu-TDC + both -L
     members in the SAME batch, so the ratio row mixes no epochs);
  3. reference-condition time responses (2 s, moving) for the figure;
  4. position limits at the reference speed (nominal floor / mean row).

The whole-pass certificate ladder for the -L members is Stage B
(cert_lobe1.py) — the adiabatic slice machinery takes its own run.

Outputs: results/mu_real_l.npz, results/s1_l.npz, results/timeresp_l.npz;
appends to results/log_lobe1.txt.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers
from mu_real import judge_mixed
from robust_design import freq_grid
from eval2 import limits
from run_stage78 import time_run, settling
from simulate import MillingSimulation
from sim_ctrl2 import ScheduledLTI

KINDS_L = ('ps_ac_rfa_l', 'ps_ac_ri_l')

LOG = open(os.path.join(C.RESULTS, 'log_lobe1.txt'), 'a')


def log(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate, ap=C.AP_S)
mks = load_controllers(plate, plant, include_open=True)

log('')
log('=' * 78)
log('JUDGE (-L winners), Stage A  ' + time.strftime('%Y-%m-%d %H:%M'))
log('=' * 78)

# ---- (1) mixed gate at the probed nodes ---------------------------------
f = freq_grid()
res_mu = dict(f=f)
log('')
log('MIXED GATE at the probed nodes (actuator block included)')
for k in KINDS_L:
    if k not in mks:
        log(f'  {k}: not in store, skipped')
        continue
    c = mks[k](plant)
    for frx in (0.0, 0.5, 1.0):
        x = frx * plate.lp
        t0 = time.time()
        r = judge_mixed(plate, c.at(x)[0], x, f)
        res_mu[f'{k}_{int(100 * frx)}_mixed'] = r['curve_mixed']
        res_mu[f'{k}_{int(100 * frx)}_cx'] = r['curve_cx']
        log(f'  {k:12s} x={frx:4.0%}: complex {r["cx"]:7.3f} -> MIXED '
            f'{r["mixed"]:7.3f} at {r["f_mixed"]:6.0f} Hz'
            + ('   G1(real) MET' if r['mixed'] < 1 else '   >= 1: NOT MET')
            + f'   [{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'mu_real_l.npz'), **res_mu)
log('-> results/mu_real_l.npz')

# ---- (2) fresh one-batch recovery metrics -------------------------------
log('')
log('FRESH RECOVERY BATCH (mu_tdc + -L members, one batch, identical '
    'calls)')
fr = {}
for k in ('mu_tdc',) + KINDS_L:
    if k not in mks:
        continue
    c = mks[k](plant)
    t0 = time.time()
    tm = time_run(plate, plant, c)
    st = settling(plate, plant, c)
    fr[f'{k}_Ts'] = st['T_s']
    fr[f'{k}_Eu'] = tm['E_u']
    fr[f'{k}_umax'] = tm['u_max']
    fr[f'{k}_Amax'] = tm['A_max']
    log(f'  {k:12s}: A_max {tm["A_max"]:8.2f} um  T_s {st["T_s"]*1e3:6.1f}'
        f' ms  E_u {tm["E_u"]:10.1f}  u_max {tm["u_max"]:7.2f} V'
        + ('  DIVERGED' if tm['diverged'] else '')
        + f'  [{time.time() - t0:.0f}s]')
for k in KINDS_L:
    if f'{k}_Ts' not in fr:
        continue
    log(f'  ratios vs mu_tdc ({k}):  T_s {fr[f"{k}_Ts"]/fr["mu_tdc_Ts"]:.2f}'
        f'  E_u {fr[f"{k}_Eu"]/fr["mu_tdc_Eu"]:.2f}'
        f'  u_max {fr[f"{k}_umax"]/fr["mu_tdc_umax"]:.2f}')

# ---- (4) reference-speed position limits --------------------------------
log('')
log('POSITION LIMITS at the reference speed (nominal floor / mean row)')
for k in KINDS_L:
    if k not in mks:
        continue
    c = mks[k](plant)
    t0 = time.time()
    L = limits(plate, c)
    fr[f'{k}_limits'] = L
    log(f'  {k:12s}: {np.round(L * 1e3, 4).tolist()} mm  floor '
        f'{L.min()*1e3:.4f}  mean {L.mean()*1e3:.4f}  '
        f'[{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 's1_l.npz'), **fr)
log('-> results/s1_l.npz')

# ---- (3) reference-condition time responses (figure data) ---------------
log('')
log('TIME RESPONSES at the reference condition (-L members, 2 s, moving)')
tr = {}
for k in KINDS_L:
    if k not in mks:
        continue
    c = mks[k](plant)
    sim = MillingSimulation(plate, C.RPM_S, C.AP_S, ae=C.AE, fz=C.FZ,
                            sign=C.SIGN, n_modes=C.N_MODES, n_sub=C.N_SUB)
    ctl = ScheduledLTI(c, sim.dt, plate.lp, tau=sim.tau,
                       feed=plant.feed_speed(), moving=True)
    t0 = time.time()
    r = sim.run(controller=ctl, T=2.0, moving=True)
    y = np.asarray(r['y_obs'], float)
    u = np.asarray(r['u'], float)
    dt = float(r['dt'])
    n = y.size
    blk = max(1, n // 2000)
    nb = n // blk
    tr[f'{k}_env'] = np.abs(y[:nb * blk]).reshape(nb, blk).max(axis=1)
    tr[f'{k}_urms'] = np.sqrt((u[:nb * blk] ** 2).reshape(nb, blk)
                              .mean(axis=1))
    tr[f'{k}_t'] = np.arange(nb) * blk * dt
    tr[f'{k}_div'] = float(r['t_div']) if r['diverged'] else np.nan
    log(f'  {k:12s}: A_max {np.max(np.abs(y))*1e6:9.1f} um  u_max '
        f'{np.max(np.abs(u)):8.1f} V'
        + (f'  DIVERGED at {r["t_div"]:.3f}s' if r['diverged'] else '')
        + f'  [{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'timeresp_l.npz'), **tr)
log('-> results/timeresp_l.npz')
log('judge_lobe1 Stage A done.')
