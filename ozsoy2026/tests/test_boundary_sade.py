"""Chatter detector, time-domain limit, and the optimiser."""
import numpy as np
import pytest
from sade import sade
from tdboundary import boundary, classify
from zoa import a_lim


def test_detector_on_both_sides_of_the_limit():
    al = a_lim(2500, 535.0)
    assert classify(2500, 0.9 * al, 535.0)["stable"]
    assert not classify(2500, 1.1 * al, 535.0)["stable"]


@pytest.mark.slow
@pytest.mark.parametrize("rpm,g", [(2500, 535.0), (1300, 991.0), (500, 0.0)])
def test_time_domain_limit_close_to_zoa(rpm, g):
    rel = boundary(rpm, g) / a_lim(rpm, g) - 1
    assert -0.005 < rel < 0.03


def test_sade_finds_a_cliff_edge_optimum():
    """Same shape as Eq. (2): rising fitness, then zero past an edge."""
    edge = 1234.5
    f = lambda x: (x[0] / 1000.0) ** 0.7 if x[0] <= edge else 0.0
    for s in (1, 2, 3):
        r = sade(f, 0.0, 3000.0, np_=20, max_gen=60, window=25, tol=1e-6, seed=s)
        assert r["x"][0] == pytest.approx(edge, rel=5e-3)


def test_sade_multimodal_1d():
    f = lambda x: -(x[0] ** 2) + 10 * np.cos(2 * np.pi * x[0])     # global max at 0
    hits = sum(abs(sade(f, -5.12, 5.12, np_=20, max_gen=60, window=25, seed=s)["x"][0]) < 1e-2
               for s in range(1, 6))
    assert hits >= 4
