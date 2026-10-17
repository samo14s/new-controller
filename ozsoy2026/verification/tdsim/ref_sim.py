"""Independent reference simulator for the 1-DOF regenerative milling model.

Written for the verification of src/tdsim.c without reusing its code.  The
numbers are re-entered from the paper (Tables 1-2, Eq. 3) and only compared with
src/params.py by assert_params_match().

Model (same physics as the paper's Sec. 2.1, re-derived here):
  * Altintas convention: x = feed, y = normal; the immersion angle phi is
    measured clockwise from +y, tooth j of slice z sits at
        phi = Omega t + j 2 pi / N - 2 z tan(beta) / D       (helix lag).
  * Down milling, half immersion: the tooth is in cut for phi in [pi/2, pi].
  * The flexible workpiece displaces by w along y.  The tool centre moves by -w
    relative to the workpiece, so the chip grows by (-w) projected on the radial
    unit vector (sin phi, cos phi):
        h = fz sin(phi) - (w(t) - w(t - T)) cos(phi),   h <= 0 -> no cut.
  * Tangential / radial forces on the tool  dFt = Kt h dz, dFr = Kr h dz,
    y-component on the tool  dFy = dFt sin(phi) - dFr cos(phi)
    (Altintas 2012 Eq. 2.x), force on the workpiece = -sum dFy.
  * Closed loop: G = 1/(m s^2 + c s + k), V = -g wdot (optionally clipped to
    +-vmax), actuator force = Ga(s) V with Ga = 3 s^2/(s^2 + a1 s + a0).

Two integrators:
  ref_linear  - exact zero-order-hold discretisation built from the closed-loop
                transfer functions (scipy tf2ss + cont2discrete), no limit.
  ref_clipped - continuous nonlinear loop with the voltage clip, integrated
                over each step with scipy's DOP853 at rtol=1e-11.
Both hold the cutting force constant over one step, as the paper's scheme does.
"""
import numpy as np
from scipy.signal import tf2ss, cont2discrete
from scipy.integrate import solve_ivp

# --- numbers re-entered from the paper -------------------------------------
FN, ZETA, K = 129.3, 0.0134, 1.34e7           # Table 1 (zeta: 1.34 %, see params.py)
WN = 2 * np.pi * FN
M = K / WN**2
C = 2 * ZETA * np.sqrt(K * M)
KA, A1, A0 = 3.0, 15.834, 2785.6               # Eq. (3)
D, NT, BETA = 16e-3, 4, np.deg2rad(45.0)        # Table 2
KT, KR = 660e6, 180e6                           # Table 2 (N/m^2)
FZ = 0.05e-3                                    # Table 2
PHI_IN, PHI_OUT = np.pi / 2, np.pi              # down milling, ae = D/2
SPR, NREV, NLAY = 512, 50, 30                   # Table 2


def assert_params_match():
    import params as P
    pairs = dict(M=(M, P.M), C=(C, P.C), K=(K, P.K), KA=(KA, P.GA_GAIN),
                 A1=(A1, P.GA_A1), A0=(A0, P.GA_A0), D=(D, P.D_TOOL),
                 NT=(NT, P.N_TEETH), BETA=(BETA, P.HELIX), KT=(KT, P.K_TC),
                 KR=(KR, P.K_RC), FZ=(FZ, P.FZ), SPR=(SPR, P.STEPS_PER_REV),
                 NREV=(NREV, P.N_REVS), NLAY=(NLAY, P.N_LAYERS))
    st, ex = P.immersion_angles()
    pairs["PHI_IN"] = (PHI_IN, st)
    pairs["PHI_OUT"] = (PHI_OUT, ex)
    bad = {k: v for k, v in pairs.items() if not np.isclose(v[0], v[1], rtol=1e-12)}
    return bad


def slice_angles(spr, a, nlay):
    """phi[s, j, l] for spindle step s, tooth j, slice l (rad, in [0, 2pi))."""
    dz = a / nlay
    z = dz * (np.arange(nlay) + 0.5)
    theta = 2 * np.pi * np.arange(spr) / spr
    lag = 2 * z * np.tan(BETA) / D
    phi = theta[:, None, None] + 2 * np.pi * np.arange(NT)[None, :, None] / NT \
        - lag[None, None, :]
    return np.mod(phi, 2 * np.pi), dz


def cut_force(phi, dz, dw):
    """Force on the workpiece along y for one step (phi: array of slice angles)."""
    inside = (phi >= PHI_IN) & (phi <= PHI_OUT)
    s, c = np.sin(phi), np.cos(phi)
    h = FZ * s - dw * c
    h = np.where(inside & (h > 0.0), h, 0.0)
    dFt = KT * h * dz
    dFr = KR * h * dz
    Fy_tool = np.sum(dFt * s - dFr * c)
    return -Fy_tool


def closed_loop_ss(g):
    """SIMO state space from cutting force to [w, wdot, f_act] (exact, via tf2ss)."""
    act = np.array([1.0, A1, A0])
    den = np.polyadd(np.polymul([M, C, K], act), [KA * g, 0.0, 0.0, 0.0])
    num_w = np.concatenate([[0.0, 0.0], act])                 # (s^2+a1 s+a0)
    num_v = np.concatenate([[0.0], np.polymul([1.0, 0.0], act)])
    num_fa = np.array([0.0, -KA * g, 0.0, 0.0, 0.0])          # -KA g s^3
    num = np.vstack([num_w, num_v, num_fa])
    A, B, Cm, Dm = tf2ss(num, den)
    return A, B, Cm, Dm


def ref_linear(rpm, a, g, nrev=NREV, spr=SPR, nlay=NLAY):
    """Exact-ZOH reference; returns dict of histories w, v, fact, fcut."""
    A, B, Cm, Dm = closed_loop_ss(g)
    dt = 60.0 / (rpm * spr)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, Cm, Dm), dt, method="zoh")
    Bd = Bd[:, 0]
    phi, dz = slice_angles(spr, a, nlay)
    nd = spr // NT
    assert nd * NT == spr
    n = spr * nrev
    w = np.zeros(n + 1); v = np.zeros(n + 1); fa = np.zeros(n + 1); fc = np.zeros(n + 1)
    z = np.zeros(Ad.shape[0])
    for i in range(n):
        y = Cm @ z                      # outputs at t_i (Dm = 0)
        w[i], v[i] = y[0], y[1]
        dw = w[i] - (w[i - nd] if i >= nd else 0.0)
        f = cut_force(phi[i % spr], dz, dw)
        fc[i] = f
        fa[i] = y[2]
        z = Ad @ z + Bd * f
    y = Cm @ z
    w[n], v[n], fa[n], fc[n] = y[0], y[1], y[2], fc[n - 1]
    return dict(w=w, v=v, fact=fa, fcut=fc, dt=dt)


def _actuator_ss():
    Aa, Ba, Ca, Da = tf2ss([KA, 0.0, 0.0], [1.0, A1, A0])
    return Aa, Ba[:, 0], Ca[0], Da[0, 0]


def ref_clipped(rpm, a, g, vmax, nrev=10, spr=SPR, nlay=NLAY, rtol=1e-11):
    """Nonlinear (clipped) reference with DOP853 over each step."""
    Aa, Ba, Ca, Da = _actuator_ss()
    dt = 60.0 / (rpm * spr)
    phi, dz = slice_angles(spr, a, nlay)
    nd = spr // NT
    n = spr * nrev

    def volt(v):
        u = -g * v
        return np.clip(u, -vmax, vmax) if vmax > 0 else u

    def rhs(t, s, f):
        w, v, q = s[0], s[1], s[2:]
        V = volt(v)
        fact = Ca @ q + Da * V
        return np.concatenate([[v, (f + fact - C * v - K * w) / M], Aa @ q + Ba * V])

    w = np.zeros(n + 1); vv = np.zeros(n + 1); fa = np.zeros(n + 1)
    fc = np.zeros(n + 1); vc = np.zeros(n + 1)
    s = np.zeros(4)
    for i in range(n):
        w[i], vv[i] = s[0], s[1]
        vc[i] = volt(s[1])
        fa[i] = Ca @ s[2:] + Da * vc[i]
        dw = w[i] - (w[i - nd] if i >= nd else 0.0)
        f = cut_force(phi[i % spr], dz, dw)
        fc[i] = f
        sol = solve_ivp(rhs, (0.0, dt), s, method="DOP853", args=(f,),
                        rtol=rtol, atol=1e-16)
        s = sol.y[:, -1]
    w[n], vv[n] = s[0], s[1]
    vc[n] = volt(s[1]); fa[n] = Ca @ s[2:] + Da * vc[n]; fc[n] = fc[n - 1]
    return dict(w=w, v=vv, fact=fa, fcut=fc, vcmd=vc, dt=dt)


def mean_force_analytic(a):
    """Average y force on the workpiece over one revolution (static chip)."""
    # (N a fz / 2 pi) * int_{pi/2}^{pi} (Kt sin^2 - Kr sin cos) dphi, sign flipped
    integral = KT * np.pi / 4 + KR / 2
    return -(NT * a * FZ / (2 * np.pi)) * integral
