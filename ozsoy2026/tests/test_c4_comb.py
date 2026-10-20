"""C4: the comb terms of the command in the simulator."""
import numpy as np
import pytest
import params as P
from tdsim import simulate

SPR = P.STEPS_PER_REV
ND = SPR // P.N_TEETH


def test_comb_build_with_vanishing_gains_is_the_default():
    """The COMB build at a negligible g2 reproduces the plain-DVF build (a
    clipped run, where any change in the command path would show)."""
    d = simulate(2500, 12e-3, 1500.0, vmax=9.0, store=True, nrev=40, edge=True, nreg=4)
    c = simulate(2500, 12e-3, 1500.0, vmax=9.0, store=True, nrev=40, edge=True, nreg=4, g2=1e-300)
    assert np.abs(c["x"] - d["x"]).max() <= 1e-15 * np.abs(d["x"]).max()


def test_comb_terms_vanish_on_the_settled_cut():
    """On the tooth-periodic forced motion x'(t) = x'(t - T) and x(t) = x(t - T),
    so the command settles to -g1 x', and the settled motion is the DVF one."""
    rpm, a, g1 = 2500, 8e-3, 750.0
    d = simulate(rpm, a, g1, store=True, nrev=300, edge=True)
    c = simulate(rpm, a, g1, store=True, nrev=300, edge=True, g2=1500.0, kd=2e4)
    k = slice(-SPR, None)
    assert np.abs(c["vcmd"][k] + g1 * c["v"][k]).max() < 1e-6 * np.abs(c["vcmd"][k]).max()
    assert np.abs(c["x"][k] - d["x"][k]).max() < 1e-6 * np.abs(d["x"][k]).max()


def test_delayed_velocity_of_the_comb_term():
    """Kick a nearly free structure (negligible cut, actuator on): the logged
    command is -g1 x' - g2 (x' - x'(t - T)) with x'(t - T) the stored velocity
    one tooth period back; over the first period after the kick that is the
    (almost zero) pre-kick velocity, so the command is -(g1 + g2) x'."""
    g1, g2 = 300.0, 700.0
    ks = 3 * SPR
    r = simulate(2500, 1e-9, g1, store=True, nrev=6, kick_step=ks, kick_dv=1e-3, g2=g2)
    v, u = r["v"], r["vcmd"]
    k = np.arange(ks + 1, ks + 3 * ND)
    assert np.allclose(u[k], -g1 * v[k] - g2 * (v[k] - v[k - ND]), rtol=1e-9, atol=1e-9 * np.abs(u[k]).max())
    k = np.arange(ks + 1, ks + ND)
    assert np.abs(v[k - ND]).max() < 1e-5 * np.abs(v[k]).max()
    assert np.abs(u[k] + (g1 + g2) * v[k]).max() < 1e-5 * np.abs(u[k]).max()


# ------------------------------------------------------ collocated model ----
import c2_certificate as c2
import c4_comb as c4
from zoa import a_lim


def test_comb_model_without_comb_is_the_c2_model():
    A, B, K = c4.model(2500.0, 10e-3, 1500.0)
    A2, B2, K2 = c2.model(2500.0, 10e-3, 1500.0)
    assert np.allclose(A, A2, rtol=0, atol=1e-12 * np.abs(A2).max()) and np.array_equal(B, B2)
    assert np.array_equal(K, K2)


@pytest.mark.parametrize("rpm,g", [(1090.0, 1500.0), (2500.0, 1500.0), (1900.0, 196.0)])
def test_dvf_limit_of_the_scan_is_the_zoa_limit(rpm, g):
    assert c4.a_lim_ctrl(rpm, g) == pytest.approx(a_lim(rpm, g), rel=1e-3)


def test_comb_row_vanishes_on_a_tooth_periodic_state():
    """K applied to a state on a tooth-periodic motion (sum of tooth-passing
    harmonics, history included) is -g1 x' up to the collocation error."""
    rpm, g1, g2 = 2500.0, 1500.0, 2000.0
    T = 60.0 / (rpm * P.N_TEETH)
    w = 2 * np.pi / T * np.array([1, 2, 3])
    c = np.array([1.0, 0.4, 0.1]) * 1e-6
    th = c2.history_nodes(rpm)
    x = lambda t: np.sum(c * np.cos(np.outer(np.atleast_1d(t), w) + 0.3), axis=1)
    v = lambda t: -np.sum(c * w * np.sin(np.outer(np.atleast_1d(t), w) + 0.3), axis=1)
    X = np.concatenate([x(0.0), v(0.0), [0.0, 0.0], x(th)])
    K = c4.model(rpm, 10e-3, g1, g2)[2]
    assert abs(K @ X + g1 * v(0.0)[0]) < 1e-5 * g2 * np.sum(c * w)


def test_comb_limit_converges_in_the_nodes():
    """The first unstable depth with the comb term moves by < 1e-4 between
    the per-speed node count and 1.6x / 2.4x as many nodes."""
    rpm, g1, g2 = 2500.0, 0.0, 1000.0
    a0 = c4.a_lim_ctrl(rpm, g1, g2)
    N = c2.nodes_for(rpm)
    for n2 in (int(1.6 * N), int(2.4 * N)):
        assert c4.lead(rpm, a0 * (1 - 1e-4), g1, g2, N=n2) < 0 < c4.lead(rpm, a0 * (1 + 1e-4), g1, g2, N=n2)


@pytest.mark.slow
@pytest.mark.parametrize("rpm,g1,g2", [(2500.0, 1500.0, 500.0), (2500.0, 0.0, 2000.0), (1090.0, 0.0, 2000.0)])
def test_comb_limit_matches_the_simulator(rpm, g1, g2):
    """The simulated (unclipped, step entry) stability limit with the comb
    terms agrees with the collocated model to 1.5 % (the simulator keeps the
    time-varying cutting coefficient)."""
    from tdboundary import boundary
    am = c4.a_lim_ctrl(rpm, g1, g2)
    at = boundary(rpm, g1, a_lo=0.8 * am, a_hi=1.3 * am, growth=1.04, g2=g2)
    assert at == pytest.approx(am, rel=0.015)


def test_negative_kd_opposes_regeneration():
    """KAPPA < 0: kd < 0 raises the stability limit (about 1 mm per 1e5 V/m
    at 2500 rpm), kd > 0 lowers it."""
    base = c4.a_lim_ctrl(2500.0, 1500.0)
    assert c4.a_lim_ctrl(2500.0, 1500.0, kd=-1e5) - base == pytest.approx(0.98e-3, rel=0.05)
    assert c4.a_lim_ctrl(2500.0, 1500.0, kd=1e5) < base


@pytest.mark.parametrize("rpm", [1090.0, 2500.0])
def test_feasible_dvf_gain_is_the_feasibility_optimum(rpm):
    """With the delayed terms off, the authority rule gives feas_g3000's gain."""
    from conftest import path
    d = np.load(path("results", "feas_g3000.npz"))
    k = int(np.argmin(np.abs(d["rpm"] - rpm)))
    assert c4.feasible_g1(rpm) == pytest.approx(float(d["g_best"][k]), abs=0.5)


# --------------------------------------------- comb delay and runout options ----
def test_comb_delay_near_one_is_continuous_with_the_exact_delay():
    """The fractional-delay path (rho != 1) at rho = 1 + 1e-9 reproduces the
    exact-delay path to the size of the perturbation."""
    a = simulate(2500, 20e-3, 1500.0, vmax=9.0, store=True, nrev=30, edge=True, nreg=4, g2=2000.0)
    b = simulate(2500, 20e-3, 1500.0, vmax=9.0, store=True, nrev=30, edge=True, nreg=4, g2=2000.0,
                 comb_delay=1.0 + 1e-9)
    assert not np.array_equal(a["x"], b["x"])
    assert np.abs(b["x"] - a["x"]).max() < 1e-6 * np.abs(a["x"]).max()


def test_loop_delay_uses_the_motion_one_delay_back():
    """With a loop delay of exactly k steps the logged command at a step is
    -g1 x'(t - tau) - g2 (x'(t - tau) - x'(t - tau - T)) from the stored velocity."""
    rpm, g1, g2, k = 2500.0, 1200.0, 800.0, 3
    dt = 60.0 / (rpm * SPR)
    r = simulate(rpm, 8e-3, g1, store=True, nrev=6, edge=True, nreg=4, g2=g2, loop_delay=k * dt)
    v, u = r["v"], r["vcmd"]
    n = np.arange(2 * SPR, 6 * SPR)
    ref = -g1 * v[n - k] - g2 * (v[n - k] - v[n - k - ND])
    assert np.allclose(u[n], ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())


def test_spindle_comb_vanishes_on_the_runout_orbit():
    """With runout the settled motion is spindle-periodic: the tooth comb acts
    on it, the spindle-period comb (delay N T) does not."""
    rpm, a, g1, g2, ro = 2500, 10e-3, 1500.0, 2000.0, 2e-6
    k = slice(-SPR, None)
    t = simulate(rpm, a, g1, store=True, nrev=300, edge=True, nreg=4, g2=g2, runout=ro)
    s = simulate(rpm, a, g1, store=True, nrev=300, edge=True, nreg=4, g2=g2, runout=ro, comb_delay=P.N_TEETH)
    res = lambda r: np.abs(r["vcmd"][k] + g1 * r["v"][k]).max()
    assert res(s) < 1e-6 * np.abs(s["vcmd"][k]).max()
    assert res(t) > 0.05 * np.abs(t["vcmd"][k]).max()


# ------------------------------------------------- closed form (c4_robust) ----
import c4_robust as R


@pytest.mark.parametrize("rpm,g1,g2,kd", [(2500.0, 1500.0, 0.0, 0.0), (2500.0, 1500.0, 2000.0, 0.0),
                                          (1090.0, 1500.0, 1000.0, 0.0), (2500.0, 1500.0, 0.0, -3e5)])
def test_closed_form_limit_is_the_collocated_limit(rpm, g1, g2, kd):
    assert R.a_lim_cf(rpm, g1, g2, kd) == pytest.approx(c4.a_lim_ctrl(rpm, g1, g2, kd), rel=2e-4)


def test_closed_form_orbit_is_the_c2_orbit():
    for rpm, g1, a in [(2500.0, 1500.0, 20e-3), (1420.0, 1500.0, 10e-3)]:
        assert R.orbit_command(rpm, a, g1)[0] == pytest.approx(c2.orbit(rpm, g1, a)[2], abs=1e-4)
        assert R.orbit_command(rpm, a, g1, 2000.0)[0] == pytest.approx(R.orbit_command(rpm, a, g1)[0], rel=1e-12)


def test_runout_force_has_only_tooth_harmonics_without_runout():
    W = R.slice_force_revolution()
    k = np.arange(len(W))
    assert np.abs(W[k % P.N_TEETH != 0]).max() < 1e-12 * np.abs(W).max()
    Wr = R.slice_force_revolution(2e-6)
    assert np.abs(Wr[1]) > 1e-3 * np.abs(W[P.N_TEETH])


@pytest.mark.slow
@pytest.mark.parametrize("case", [(2500.0, 20e-3, 1500.0, 2000.0, 1.0, 2e-6, 0.0),
                                  (2500.0, 20e-3, 1500.0, 2000.0, 0.99, 0.0, 0.0),
                                  (1420.0, 15e-3, 1500.0, 2000.0, 0.98, 2e-6, 2.0)])
def test_closed_form_orbit_command_matches_the_simulator(case):
    """Runout and comb-delay error: the settled peak command of the simulator
    (time-varying cut, 30 layers) and the closed form agree to 2 %."""
    td, cf, drift = R._td_orbit(case)
    assert drift < 1e-8 and td == pytest.approx(cf, rel=0.02)


def test_comb_fragility_to_delay_is_its_phase_not_its_gain():
    """2500 rpm, 300 us loop delay and an assumed 400 Hz mode: the delayed
    design (g1 = 1396, g2 = 2000) loses most of its limit, while a plain DVF
    with the comb's peak gain g1 + 2 g2 stays stable much deeper; so the loss
    comes from the phase of 1 - exp(-sT), not from the gain magnitude."""
    g1, g2, mode = 1395.8, 2000.0, (400.0, 6.0, 0.01)
    comb = R.a_lim_cf(2500.0, g1, g2, tau=300e-6, mode2=mode)
    flat = R.a_lim_cf(2500.0, g1 + 2 * g2, tau=300e-6, mode2=mode)
    assert comb < 0.5 * R.a_lim_cf(2500.0, g1, g2) and flat > 3 * comb
