"""judge_tdcj.py — the delay-exploiting family members measured on the
lobe axis, with their full honest profile.

Motivation (measured first, results/log_lobe1.txt): the STORED ps_tdc_j
— the family's scheduled base with the grafted delayed pair, designed
under the original shared J protocol, six parameters as mu-TDC — already
carries a full-resolution lobe floor of 0.583 mm: above the benchmark's
0.474 mm floor at every one of the 37 speeds and above the benchmark's
own curve pointwise at 31/37.  No lobe-first redesign is run for it: the
gap the DI members could not close is closed by the family's own
delay-exploiting member under the unchanged protocol, which is the
sharper statement.

This battery completes its profile the way every member's profile is
completed (nothing assumed, everything measured):

  1. full-resolution lobes of mu_tdc_ps as well (the benchmark's own
     mu base with OUR position-scheduled cancellation pair; its gate is
     inherited from the witness by the delay-free judge convention);
  2. the mixed gate of ps_tdc_j's scheduled base at the probed nodes
     (same delay-free convention under which mu-TDC scores
     0.242-0.419 — the delayed pair is invisible to the mu judge
     either way);
  3. one fresh recovery batch: mu_tdc + ps_tdc_j + mu_tdc_ps,
     identical calls, ratios within the batch;
  4. reference-speed position limits and the 2-s reference time
     response of ps_tdc_j (figure data).

Classification is unchanged and stated: both members are
delay-exploiting, out of the delay-independent whole-pass contract by
design, exactly as the benchmark controller itself.

Outputs: results/lobes_tdcj.npz, results/mu_real_tdcj.npz,
results/s1_tdcj.npz, results/timeresp_tdcj.npz; appends to
results/log_lobe1.txt.
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
log('DELAY-EXPLOITING FAMILY MEMBERS ON THE LOBE AXIS  '
    + time.strftime('%Y-%m-%d %H:%M'))
log('=' * 78)

# ---- (1) full-resolution lobes: ps_tdc_j + mu_tdc_ps --------------------
ref = np.load(os.path.join(C.RESULTS, 'lobes_l2.npz'))
rpms = ref['rpm']
out = {'rpm': rpms, 'mu_tdc': ref['mu_tdc']}
log('')
log('FULL-RESOLUTION LOBES (m = 120, min over {0, L/2, L})')
for k in ('ps_tdc_j', 'mu_tdc_ps'):
    c = mks[k](plant)
    t0 = time.time()
    cur = []
    for r in rpms:
        L = limits(plate, c, rpm=float(r), positions=(0.0, 0.5, 1.0))
        cur.append(L.min())
    out[k] = np.array(cur)
    ge = int((out[k] * 1e3 >= 0.474).sum())
    pw = int((out[k] >= ref['mu_tdc']).sum())
    log(f'  {k:10s}: floor {min(cur) * 1e3:.3f}  max {max(cur) * 1e3:.3f}'
        f' mm  >=0.474: {ge}/37  >= mu-TDC pointwise: {pw}/37'
        f'  [{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'lobes_tdcj.npz'), **out)
log('-> results/lobes_tdcj.npz')

# ---- (2) mixed gate of ps_tdc_j's base (delay-free judge) ---------------
f = freq_grid()
res_mu = dict(f=f)
log('')
log('MIXED GATE of the ps_tdc_j scheduled base (delay-free judge '
    'convention, as mu-TDC is judged; actuator block included)')
c = mks['ps_tdc_j'](plant)
for frx in (0.0, 0.5, 1.0):
    x = frx * plate.lp
    t0 = time.time()
    r = judge_mixed(plate, c.at(x)[0], x, f)
    res_mu[f'ps_tdc_j_{int(100 * frx)}_mixed'] = r['curve_mixed']
    res_mu[f'ps_tdc_j_{int(100 * frx)}_cx'] = r['curve_cx']
    log(f'  x={frx:4.0%}: complex {r["cx"]:7.3f} -> MIXED {r["mixed"]:7.3f}'
        f' at {r["f_mixed"]:6.0f} Hz'
        + ('   G1(real) MET' if r['mixed'] < 1 else '   >= 1: NOT MET')
        + f'   [{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'mu_real_tdcj.npz'), **res_mu)
log('-> results/mu_real_tdcj.npz')

# ---- (3) one-batch recovery + (4) limits and time response --------------
log('')
log('FRESH RECOVERY BATCH (mu_tdc + delay-exploiting members, one batch)')
fr = {}
for k in ('mu_tdc', 'ps_tdc_j', 'mu_tdc_ps'):
    c = mks[k](plant)
    t0 = time.time()
    tm = time_run(plate, plant, c)
    st = settling(plate, plant, c)
    fr[f'{k}_Ts'] = st['T_s']
    fr[f'{k}_Eu'] = tm['E_u']
    fr[f'{k}_umax'] = tm['u_max']
    fr[f'{k}_Amax'] = tm['A_max']
    log(f'  {k:10s}: A_max {tm["A_max"]:8.2f} um  T_s {st["T_s"]*1e3:6.1f}'
        f' ms  E_u {tm["E_u"]:10.1f}  u_max {tm["u_max"]:7.2f} V'
        + ('  DIVERGED' if tm['diverged'] else '')
        + f'  [{time.time() - t0:.0f}s]')
for k in ('ps_tdc_j', 'mu_tdc_ps'):
    log(f'  ratios vs mu_tdc ({k}):  T_s {fr[f"{k}_Ts"]/fr["mu_tdc_Ts"]:.2f}'
        f'  E_u {fr[f"{k}_Eu"]/fr["mu_tdc_Eu"]:.2f}'
        f'  u_max {fr[f"{k}_umax"]/fr["mu_tdc_umax"]:.2f}')

log('')
log('POSITION LIMITS at the reference speed')
for k in ('ps_tdc_j', 'mu_tdc_ps'):
    c = mks[k](plant)
    t0 = time.time()
    L = limits(plate, c)
    fr[f'{k}_limits'] = L
    log(f'  {k:10s}: {np.round(L * 1e3, 4).tolist()} mm  floor '
        f'{L.min()*1e3:.4f}  mean {L.mean()*1e3:.4f}  '
        f'[{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 's1_tdcj.npz'), **fr)
log('-> results/s1_tdcj.npz')

log('')
log('TIME RESPONSE at the reference condition (ps_tdc_j, 2 s, moving)')
tr = {}
c = mks['ps_tdc_j'](plant)
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
tr['ps_tdc_j_env'] = np.abs(y[:nb * blk]).reshape(nb, blk).max(axis=1)
tr['ps_tdc_j_urms'] = np.sqrt((u[:nb * blk] ** 2).reshape(nb, blk)
                              .mean(axis=1))
tr['ps_tdc_j_t'] = np.arange(nb) * blk * dt
tr['ps_tdc_j_div'] = float(r['t_div']) if r['diverged'] else np.nan
log(f'  ps_tdc_j: A_max {np.max(np.abs(y))*1e6:9.1f} um  u_max '
    f'{np.max(np.abs(u)):8.1f} V'
    + (f'  DIVERGED at {r["t_div"]:.3f}s' if r['diverged'] else '')
    + f'  [{time.time() - t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'timeresp_tdcj.npz'), **tr)
log('-> results/timeresp_tdcj.npz')
log('judge_tdcj done.')
