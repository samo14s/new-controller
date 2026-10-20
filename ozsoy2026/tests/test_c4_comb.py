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
