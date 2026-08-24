"""Stability lobes vs spindle speed + time-domain response traces, with
the repository's own validated machinery (eval2.limits Floquet bisection,
run_stage78 MillingSimulation conventions).  Head-to-head scope only:
open loop, mu-TDC, PS-AC-RFA, PS-AC-RI.

Outputs: results/lobes.npz, results/timeresp.npz, log_stage78.txt."""
import sys, time, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config as C
from eval2 import limits
from plate_model import build_plate
from plant_ss import ControlledPlant
from stage_common import load_controllers
from run_stage78 import time_run, settling, log
from simulate import MillingSimulation
from sim_ctrl2 import ScheduledLTI

plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
plant = ControlledPlant(plate, ap=C.AP_S)
mks = load_controllers(plate, plant, include_open=True)
KINDS = ('open', 'mu_tdc', 'ps_ac_rfa', 'ps_ac_ri')

# ---- (1) speed lobes: min over the three design positions ---------------
log('')
log('=' * 78)
log('STABILITY LOBES vs SPINDLE SPEED (Floquet m = %d, five-mode truth '
    'model,' % C.M_FLOQUET)
log('min over positions {0, L/2, L}; head-to-head members only)')
log('=' * 78)
rpms = np.arange(3600, 7201, 100)
out = {'rpm': rpms}
for k in KINDS:
    c = mks[k](plant)
    t0 = time.time()
    cur = []
    for r in rpms:
        L = limits(plate, c, rpm=float(r), positions=(0.0, 0.5, 1.0))
        cur.append(L.min())
    out[k] = np.array(cur)
    log(f'  {k:10s}: min {min(cur)*1e3:.3f}  max {max(cur)*1e3:.3f} mm '
        f'over {rpms[0]}-{rpms[-1]} rpm  [{time.time()-t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'lobes.npz'), **out)
log('-> results/lobes.npz')

# ---- (2) RI's missing S1-style metrics ----------------------------------
log('')
log('S1-STYLE METRICS FOR ps_ac_ri (fills the head-to-head blanks)')
c = mks['ps_ac_ri'](plant)
L = limits(plate, c)
tm = time_run(plate, plant, c)
st = settling(plate, plant, c)
log(f'  limits {np.round(L*1e3,4).tolist()} mm  min {L.min()*1e3:.4f} '
    f'mean {L.mean()*1e3:.4f}')
log(f'  A_max {tm["A_max"]:.1f} um  E_u {tm["E_u"]:.0f}  '
    f'u_max {tm["u_max"]:.0f}  T_s {st["T_s"]*1e3:.1f} ms'
    + ('  DIVERGED' if tm['diverged'] else ''))
np.savez_compressed(os.path.join(C.RESULTS, 'ri_s1.npz'),
                    limits=L, T_s=st['T_s'], **{k: v for k, v in tm.items()
                                                if not isinstance(v, bool)
                                                and v is not None})

# ---- (3) time responses at the reference condition ----------------------
log('')
log('TIME RESPONSES at the reference condition (4900 rpm, a_p = 0.3 mm, '
    'T = 2 s, moving)')
T = 2.0
tr = {}
for k in KINDS:
    c = mks[k](plant)
    sim = MillingSimulation(plate, C.RPM_S, C.AP_S, ae=C.AE, fz=C.FZ,
                            sign=C.SIGN, n_modes=C.N_MODES, n_sub=C.N_SUB)
    ss, _ = c.at(0.0)
    ctl = None if ss is None else ScheduledLTI(c, sim.dt, plate.lp,
                                               tau=sim.tau,
                                               feed=plant.feed_speed(),
                                               moving=True)
    t0 = time.time()
    r = sim.run(controller=ctl, T=T, moving=True)
    y = np.asarray(r['y_obs'], float)
    u = np.asarray(r['u'], float)
    dt = float(r['dt'])
    # block-max envelope decimation to ~2000 points
    n = y.size
    blk = max(1, n // 2000)
    nb = n // blk
    env = np.abs(y[:nb * blk]).reshape(nb, blk).max(axis=1)
    urms = np.sqrt((u[:nb * blk] ** 2).reshape(nb, blk).mean(axis=1))
    tr[f'{k}_env'] = env
    tr[f'{k}_urms'] = urms
    tr[f'{k}_t'] = np.arange(nb) * blk * dt
    tr[f'{k}_div'] = float(r['t_div']) if r['diverged'] else np.nan
    log(f'  {k:10s}: A_max {np.max(np.abs(y))*1e6:9.1f} um  '
        f'u_max {np.max(np.abs(u)):8.1f} V'
        + (f'  DIVERGED at t = {r["t_div"]:.3f}s' if r['diverged'] else '')
        + f'  [{time.time()-t0:.0f}s]')
np.savez_compressed(os.path.join(C.RESULTS, 'timeresp.npz'), **tr)
log('-> results/timeresp.npz')

# ---- (4) fresh one-batch recovery metrics (head-to-head ratios) ---------
log('')
log('FRESH RECOVERY BATCH (one batch, identical calls; supersedes the')
log('stored-epoch S1 recovery numbers for the head-to-head ratios)')
fr = {}
for k in ('mu_tdc', 'ps_ac_rfa', 'ps_ac_ri'):
    c = mks[k](plant)
    t0 = time.time()
    tm = time_run(plate, plant, c)
    st = settling(plate, plant, c)
    fr[f'{k}_Ts'] = st['T_s']
    fr[f'{k}_Eu'] = tm['E_u']
    fr[f'{k}_umax'] = tm['u_max']
    fr[f'{k}_Amax'] = tm['A_max']
    log(f'  {k:10s}: A_max {tm["A_max"]:8.2f} um  T_s {st["T_s"]*1e3:6.1f} ms'
        f'  E_u {tm["E_u"]:10.1f}  u_max {tm["u_max"]:7.2f} V'
        f'  [{time.time()-t0:.0f}s]')
r = fr
log('  ratios vs mu_tdc:  T_s  rfa %.2f  ri %.2f;  E_u  rfa %.2f  ri %.2f;'
    '  u_max  rfa %.2f  ri %.2f' % (
        r['ps_ac_rfa_Ts'] / r['mu_tdc_Ts'], r['ps_ac_ri_Ts'] / r['mu_tdc_Ts'],
        r['ps_ac_rfa_Eu'] / r['mu_tdc_Eu'], r['ps_ac_ri_Eu'] / r['mu_tdc_Eu'],
        r['ps_ac_rfa_umax'] / r['mu_tdc_umax'],
        r['ps_ac_ri_umax'] / r['mu_tdc_umax']))
np.savez_compressed(os.path.join(C.RESULTS, 's1_fresh.npz'), **fr)
log('-> results/s1_fresh.npz')
