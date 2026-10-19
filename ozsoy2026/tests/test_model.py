"""Plant and closed loop against the paper's Figs. 5 and 6."""
import json
import numpy as np
import pytest
import params as P
from structure import G, Gcl, closed_loop_poles, state_space
from conftest import path


def test_default_damping_is_the_one_behind_the_papers_curves():
    assert P.ZETA == pytest.approx(0.0134)
    assert P.ZETA_TABLE1 == pytest.approx(0.0145)


def test_set_damping_switches_every_module():
    P.set_damping(0.0145)
    assert P.C == pytest.approx(2 * 0.0145 * np.sqrt(P.K * P.M))
    w = 2 * np.pi * np.linspace(100, 160, 20001)
    assert 20 * np.log10(np.abs(G(1j * w)).max()) == pytest.approx(-111.79, abs=0.01)


def test_uncontrolled_peak_matches_fig6a():
    w = 2 * np.pi * np.linspace(100, 160, 20001)
    assert 20 * np.log10(np.abs(G(1j * w)).max()) == pytest.approx(-111.10, abs=0.01)


def test_uncontrolled_curve_matches_fig6a():
    d = json.load(open(path("digitized", "fig6a_closed_loop_frf.json")))
    p = np.array([q for s in d["uncontrolled"] for q in s])
    p = p[(p[:, 0] > 1) & (p[:, 0] < 250)]
    err = p[:, 1] - 20 * np.log10(np.abs(G(2j * np.pi * p[:, 0])))
    assert np.sqrt(np.mean(err ** 2)) < 0.2


def test_closed_loop_at_1300rpm_matches_fig6():
    """The only Fig. 6 label that agrees with its own curve (g = 1162)."""
    d = json.load(open(path("digitized", "fig6a_closed_loop_frf.json")))
    p = np.array([q for s in d["1300rpm_g1162_dashed"] for q in s])
    p = p[(p[:, 0] > 1) & (p[:, 0] < 250)]
    err = p[:, 1] - 20 * np.log10(np.abs(Gcl(2j * np.pi * p[:, 0], 1162.0)))
    assert np.sqrt(np.mean(err ** 2)) < 0.1
    # pole read from the vector Fig. 6b (verification/targets/fig6b_poles.csv)
    s = closed_loop_poles(1162.0)
    dom = s[np.argmax(s.imag)]
    assert dom.real == pytest.approx(-97.03, abs=0.1)
    assert dom.imag == pytest.approx(805.4, abs=0.5)


def test_state_space_has_the_closed_loop_poles():
    for g in (0.0, 600.0, 1500.0):
        A, _ = state_space(g)
        assert np.allclose(np.sort_complex(np.linalg.eigvals(A)),
                           np.sort_complex(closed_loop_poles(g)))


def test_dvf_loop_stable_far_beyond_the_search_interval():
    assert np.all(closed_loop_poles(1e4).real < 0)
    assert np.any(closed_loop_poles(3e4).real > 0)
