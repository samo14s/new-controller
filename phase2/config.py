"""
config.py — operating point, sign convention and fairness protocol for Phases 4-16.
===================================================================================
`baseline/` is frozen as the historical record.  Everything here is new work and
uses SIGN = +1, the convention of Eq. (13) as published, which is the one that
reproduces Figs. 13, 14(a) and 18 of Du et al. (2024) -- see docs/01_REPRODUCTION.md.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, 'baseline', 'plant'),
           os.path.join(ROOT, 'baseline', 'control_old')):
    if _p not in sys.path:
        sys.path.append(_p)
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ------------------------------------------------------------------ plate [P]
F_MEASURED = [540.0, 1068.0, 2787.0, 3351.0, 4122.0]        # Table 4
F_THEORETICAL = [537.0, 1101.0, 2805.0, 3423.0, 4254.0]
ZETA = (0.0031, 0.0017, 0.0027, 0.0056, 0.0035)             # Table 4

# ------------------------------------------------------- cutting conditions [P]
AE = 0.1e-3
FZ = 0.02e-3
RPM_S, AP_S = 4900, 0.3e-3          # reference condition "S", Section 4.2
RPM_GRID = (4300, 4900, 5500, 6100, 6700)                   # Section 5

# ------------------------------------------------------------ sign convention
SIGN = +1.0                          # Eq. (13) as published -- see docs/01

# ------------------------------------------------------------------ numerics
N_MODES_DESIGN = 2                   # the paper's reduced model
N_MODES_EVAL = 5                     # truth model used for every evaluation
M_FLOQUET = 120                      # sub-intervals for the monodromy
M_FLOQUET_FAST = 40                  # during optimisation
N_PERIOD = 40
N_SUB_TIME = 164                     # Newmark sub-steps per tooth period
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)      # fractions of l_P

# --------------------------------------------------- physics uncertainty set
ETA_MAX = 0.0755        # calibrated on the +17 % drift reported in Section 5
ALPHA_LO, ALPHA_HI = 0.3, 2.9        # Eq. (23) range, times abar4
ZETA_LO, ZETA_HI = 0.8, 1.2          # +/-20 % on the damping ratio (Section 3.2)

# ------------------------------------------------------- fairness protocol [!]
# Every controller sees: the same plate, patch, sensor and sign; the same
# operating point; the same evaluation (Floquet m = 120 on the five-mode model,
# Newmark n_sub = N_SUB = 656 -- 164 aliases the Oustaloup poles, see
# run_stage78.time_run); the same constraints; the same optimiser and seeds.
# Only the controller STRUCTURE differs, and its parameter count is reported.
MS_MAX = 2.0                         # modulus margin  max |S| <= 2
V_MAX_PER_N = 450.0                  # effort  max |K S P_f| <= 450 V/N
U_SAT = 450.0                        # amplifier saturation, V (PI E-420, x100)
AP_PROBE = (0.3e-3, 0.6e-3, 1.0e-3)  # depths probed by the objective
OPT = dict(n_particles=20, n_iter=20, w=0.72, c1=1.5, c2=1.5,
           v_max=0.25, seeds=(1, 2, 3))
PSO = OPT

# --------------------------------------------------------- sensor / actuator
# Actuator side.  Sections 3-4 of the paper say "left lower corner", Section 5
# says "right lower corner of the plate back" with the sensor at the right upper
# corner.  Only the RIGHT configuration makes the products D_obs(i) H_Pe(i) share
# one sign, i.e. a dual (collocated-like) actuator/sensor pair with bounded phase
# and interlaced poles and zeros.  With a single input that is the only layout in
# which one controller can damp every mode, and it is the one the experiments of
# Section 5 use, so it is the layout compared here.
PATCH_SIDE = 'right'
X_OBS_FRAC, Z_OBS_FRAC = 1.0, 1.0    # displacement sensor at the upper corner


def build_plate(n_modes=5, calibrate=True, patch=True):
    from chebyshev_plate import ChebyshevPlate
    p = ChebyshevPlate(PX=14, PZ=14, n_modes=n_modes,
                       zeta_modes=ZETA[:n_modes])
    if patch:
        if PATCH_SIDE == 'left':
            p.add_piezo_patch(x1=0.0, x2=0.020, z1=0.0, z2=0.060)
        else:
            p.add_piezo_patch(x1=p.lp - 0.020, x2=p.lp, z1=0.0, z2=0.060)
    if calibrate:
        p.calibrate_frequencies(F_MEASURED[:n_modes])
    return p


# ---------------------------------------------------------------------------
# Aliases expected by the evaluation modules imported from baseline/control.
# ---------------------------------------------------------------------------
SIGN_SIM = SIGN
N_MODES = N_MODES_EVAL
RPM_DESIGN = RPM_S
POSITIONS_DESIGN = (0.0, 0.5, 1.0)
M_FLOQUET_PSO = M_FLOQUET_FAST
N_SUB = 656
V_PER_N = V_MAX_PER_N
F_THEORETICAL_ = F_THEORETICAL

# fractional-order realisation (kept identical for every controller that uses it)
OUST_WB = 2 * np.pi * 1.0
OUST_WH = 2 * np.pi * 1.0e5
OUST_N = 3
ROLLOFF_HZ = 8000.0
ROLLOFF_ORDER = 2

# ------------------------------------------------- search bounds per structure
BOUNDS = dict(
    pid=dict(log_Kp=(2.0, 8.0), log_Ki=(1.0, 8.0), log_Kd=(-2.0, 5.0),
             log_Nd=(2.0, 5.0)),
    lqr=dict(log_q_pos=(-2.0, 8.0), log_q_vel=(-6.0, 4.0), log_r=(-8.0, 2.0),
             log_qo=(-4.0, 8.0), log_ro=(-14.0, -4.0)),
    smc=dict(log_lam=(1.0, 5.0), log_k=(0.0, 8.0), log_phi=(-9.0, -3.0),
             log_qo=(-4.0, 8.0), log_ro=(-14.0, -4.0)),
    proposed=dict(log_q_pos=(-2.0, 8.0), log_q_vel=(-6.0, 4.0), log_r=(-8.0, 2.0),
                  log_qo=(-4.0, 8.0), log_ro=(-14.0, -4.0),
                  kpd=(-1.0, 1.0), kdd=(-1.0, 1.0)),
)


# ------------------------------------------- Stage 3-4 search bounds
# FOPID and LQG/PS-AC are given the plage physically useful for THEIR OWN
# parameters; the intervals have comparable relative extent (5-10 decades).
# PS-AC deliberately gets the SAME four parameters as LQG: the comparison is
# then between two designs with identical tuning freedom, differing only in the
# model each is designed on.
BOUNDS2 = dict(
    fopid=dict(log_Kp=(2.0, 8.0), log_Ki=(2.0, 9.0), log_Kd=(0.0, 5.0),
               lam=(0.05, 1.0), mu=(0.05, 1.0)),
    lqg=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
             log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0)),
    mu_tdc=dict(kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
    mu_phys_tdc=dict(kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
    ps_ac_r=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                 log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0),
                 a4_mult=(0.4, 2.0)),
    ps_tdc=dict(kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
    ps_tdc_r=dict(kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
    ps_tdc_j=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                  log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0),
                  kpd=(-1.5, 1.5), kdd=(-1.5, 1.5)),
    ps_ac=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
               log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0)),
    ps_ac_eta=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                   log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0)),
    ps_ac_obs=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                   log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0)),
    ps_ac_full=dict(log_q_pos=(10.0, 20.0), log_q_vel=(-4.0, 8.0),
                    log_r=(-12.0, -4.0), log_ratio=(4.0, 16.0)),
)
