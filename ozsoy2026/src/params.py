"""Every number the reproduction takes from Ozsoy, Sims & Ozturk (JMP 171, 2026).

Each constant cites the place in the paper it comes from.  Nothing here is
fitted; quantities the paper does not state are collected at the bottom under
ASSUMED, with the reason for each choice.
"""
import numpy as np

# --- Table 1: preloaded structure (single dominant mode, Sec. 2.2) ----------
FN = 129.3                # Hz
ZETA_TABLE1 = 0.0145      # -  (Table 1 and the identified curve of Fig. 5)
K = 1.34e7                # N/m
WN = 2 * np.pi * FN
M = K / WN**2             # modal mass, kg

# The paper's computed curves were not produced with 1.45 %.  The uncontrolled
# SLD of Figs. 9 and 11 and the uncontrolled FRF peak of Fig. 6a (-111.10 dB)
# are matched exactly by 1.34 % (see docs, Sec. 2); 1.34 is also the mantissa
# of the stiffness, which suggests a transcription slip.  The reproduction uses
# 1.34 % by default and reports 1.45 % as a sensitivity case.
ZETA_PAPER_CURVES = 0.0134
ZETA = ZETA_PAPER_CURVES
C = 2 * ZETA * np.sqrt(K * M)


def set_damping(zeta):
    """Switch the structural damping ratio for every module (read at call time)."""
    global ZETA, C
    ZETA = zeta
    C = 2 * zeta * np.sqrt(K * M)

# --- Eq. (3) / Table 1: proof-mass actuator, force per volt ----------------
GA_GAIN = 3.0             # N/V  (force constant, Sec. 2.2)
GA_A1 = 15.834            # s^-1  (2 zeta_p w_p)
GA_A0 = 2785.6            # s^-2  (w_p^2)
WP = np.sqrt(GA_A0)
ZETA_P = GA_A1 / (2 * WP)
F_MAX = 27.0              # N, peak actuator force (Sec. 2.1 Step 2, Sec. 2.2)

# --- Table 2: cutting conditions ------------------------------------------
D_TOOL = 16e-3            # m
N_TEETH = 4
HELIX = np.deg2rad(45.0)
K_TC = 660e6              # N/m^2
K_RC = 180e6              # N/m^2
FZ = 0.05e-3              # m/tooth
RADIAL_IMMERSION = 0.5    # half immersion
MILLING = "down"

# --- Table 2: time-domain simulation parameters ---------------------------
STEPS_PER_REV = 512
N_REVS = 50
N_LAYERS = 30

# --- Sec. 3: speed grid ----------------------------------------------------
RPM_GRID = np.arange(500, 3001, 10)

# --- ASSUMED (not stated in the paper) -------------------------------------
# Flexible direction: the actuator and beam act normal to the feed (Fig. 3a),
# so the workpiece mode is placed on y (normal), feed on x.
FLEX_DIR = "y"


def immersion_angles(milling=MILLING, ae_ratio=RADIAL_IMMERSION):
    """Entry/exit angles in Altintas' convention (clockwise, from +y)."""
    phi_e = np.arccos(1 - 2 * ae_ratio)
    if milling == "down":
        return np.pi - phi_e, np.pi
    return 0.0, phi_e
