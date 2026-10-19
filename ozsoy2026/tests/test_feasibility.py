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


# ---- stored results of the reformulated condition -------------------------
from conftest import path                                     # noqa: E402


@pytest.fixture(scope="module")
def fmap():
    return np.load(path("results", "feas_g1500.npz"))


@pytest.fixture(scope="module")
def fcert():
    return np.load(path("results", "feas_cert_g1500.npz"))


def test_sade_reaches_the_bisection_edge(fmap):
    assert np.all(fmap["a_sade"] >= fmap["a_ref"] * (1 - 1e-6))
    assert np.max(np.abs(fmap["a_sade"] - fmap["a_ref"]) / fmap["a_ref"]) < 1e-3


def test_gain_map_shape(fmap):
    g = fmap["g_best"]
    assert g.max() <= 1500.0
    assert (g >= 1499.5).sum() == 157
    i = int(np.argmin(g))
    assert fmap["rpm"][i] == 1940 and g[i] == pytest.approx(134.9, abs=0.5)


def test_certificate_binds_exactly_where_it_failed(fmap, fcert):
    binds = fcert["a_td_clipped"] < fcert["a_op_linear"] - 1e-9
    assert binds.sum() == 126
    assert np.array_equal(binds, fmap["cert_stable"] == 0)
    assert np.all(fcert["a_cert"] <= fcert["a_op_linear"] + 1e-9)


def test_certified_depths(fcert):
    assert fcert["a_cert"].mean() == pytest.approx(15.74, abs=0.05)
    assert fcert["a_cert"].min() == pytest.approx(12.43, abs=0.05)


def test_entry_saturation_destabilises_a_linearly_stable_cut():
    from tdboundary import classify
    a = 0.9 * a_lim(2500, 1500.0)
    assert classify(2500, a, 1500.0)["stable"]
    assert not classify(2500, a, 1500.0, vmax=P.F_MAX / P.GA_GAIN)["stable"]


def test_clipped_limit_rises_with_gain():
    sweep = np.loadtxt(path("results", "table3_clip_gain_sweep.csv"), delimiter=",", skiprows=1)
    t3 = np.genfromtxt(path("results", "table3_main.csv"), delimiter=",", names=True)
    for rpm in (2500, 3000):
        i = int(np.nonzero(t3["rpm"] == rpm)[0][0])
        pts = [(t3["g_prop"][i], t3["a_prop_td"][i])]
        pts += [(g, a) for r, g, a in sweep if r == rpm]
        pts += [(1500.0, t3["a_clip1500_td"][i]), (3000.0, t3["a_clip3000_td"][i])]
        pts.sort()
        assert np.all(np.diff([a for _, a in pts]) > 0)


def test_single_gain_under_the_new_condition():
    import json
    c = json.load(open(path("results", "feas_compare_g1500.json")))
    assert c["single_gain"]["g"] == pytest.approx(134.9, abs=0.5)
