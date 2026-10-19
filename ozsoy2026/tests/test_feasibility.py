"""The reformulated feasibility condition (src/feasibility.py)."""
import numpy as np
import pytest
import params as P
import feasibility as F
from tdsim import simulate
from zoa import a_lim


def test_helix_cancels_the_first_harmonic_at_one_axial_pitch():
    pitch = 2 * np.pi * (P.D_TOOL / 2) / (P.N_TEETH * np.tan(P.HELIX))     # 12.57 mm
    F1 = np.abs(F.force_harmonics([0.5 * pitch, pitch])[:, 1])
    assert F1[1] < 1e-9 * F1[0]


def test_mean_force_equals_the_time_domain_one():
    F0 = F.force_harmonics([1e-3])[0, 0].real
    exact = -(P.N_TEETH * 1e-3 * P.FZ / (2 * np.pi)) * (P.K_TC * np.pi / 4 + P.K_RC / 2)
    assert F0 == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("rpm,g,frac", [(1300, 1500.0, 0.6), (2300, 600.0, 0.9),
                                        (3000, 700.0, 0.9), (1950, 300.0, 0.3)])
def test_closed_form_matches_settled_simulation(rpm, g, frac):
    a = frac * a_lim(rpm, g)
    r = simulate(rpm, a, g, store=True, nrev=150)
    td = P.GA_GAIN * g * np.abs(r["v"][130 * 512:]).max()
    assert F.forced_demand(rpm, g, [a])[0] == pytest.approx(td, rel=0.01)


@pytest.mark.parametrize("rpm", [700, 1300, 2300, 2700])
def test_envelope_rises_with_gain(rpm):
    gs = np.arange(0, 1501, 50.0)
    env = np.array([F.demand_envelope(rpm, g)[1] for g in gs])
    assert np.all(np.diff(env) >= -1e-6 * env.max())


@pytest.mark.parametrize("rpm", [1950, 2500])
def test_edge_gain_is_the_feasibility_edge(rpm):
    g = F.edge_gain(rpm, g_hi=3000.0)
    assert F.fitness(rpm, g) > 0
    assert F.fitness(rpm, g + 1.0) == 0


def test_never_below_the_uncontrolled_limit():
    for rpm in (700, 1950, 2300):
        g = F.edge_gain(rpm, g_hi=1500.0)
        assert F.fitness(rpm, g) >= a_lim(rpm, 0.0) * 1e3 - 1e-9


def test_guard_and_margin_only_tighten():
    g0 = F.edge_gain(2500, g_hi=3000.0)
    assert F.edge_gain(2500, g_hi=3000.0, guard=0.85) < g0
    assert F.edge_gain(2500, g_hi=3000.0, kappa=0.95) <= g0 + 0.25
