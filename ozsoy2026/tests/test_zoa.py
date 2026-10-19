"""Zero-order stability limit: paper curve, independent SDM, known pitfalls."""
import csv
import numpy as np
import pytest
import params as P
from zoa import ALPHA, a_lim, sld
from conftest import path, load_points


def test_directional_factor():
    assert ALPHA == pytest.approx(-1 - np.pi / 2 * P.K_RC / P.K_TC, rel=1e-12)


def test_uncontrolled_lobes_match_fig9():
    u = load_points("fig9_uncontrolled_points.csv")
    u = u[(u[:, 0] > 400) & (u[:, 0] < 3000)][::8]
    rel = np.abs(sld(u[:, 0], 0.0) * 1e3 / u[:, 1] - 1)
    assert np.median(rel) < 1e-3
    assert np.percentile(rel, 95) < 6e-3


@pytest.mark.parametrize("rpm,mm", [(488, 6.78), (651, 8.99), (977, 13.63), (1952, 29.26)])
def test_lobe_peaks(rpm, mm):
    assert a_lim(rpm, 0.0) * 1e3 == pytest.approx(mm, abs=0.02)


def test_uncontrolled_minimum():
    assert sld(np.arange(1000, 1250, 1.0), 0.0).min() * 1e3 == pytest.approx(1.213, abs=0.002)


def test_no_spurious_root_at_a_tooth_passing_pole():
    """Found by the verification: 29.296875 mm used to come back here."""
    P.set_damping(0.0145)
    assert a_lim(1983, 300.0) * 1e3 == pytest.approx(30.07, abs=0.02)


def test_agrees_with_independent_semi_discretisation():
    rows = list(csv.DictReader(open(path("verification", "zoa_sdm", "compare_12speeds.csv"))))
    assert len(rows) == 96
    worst = 0.0
    for r in rows:
        P.set_damping(float(r["zeta"]))
        ours = a_lim(float(r["rpm"]), float(r["g"])) * 1e3
        worst = max(worst, abs(ours / float(r["a_sdm_meanH_mm"]) - 1))
    assert worst < 1e-4


def test_limit_grows_with_gain_at_1300rpm():
    """Underlies the inconsistency between the worked example and Table 3."""
    a = [a_lim(1300, g) for g in (0, 600, 1000, 1162, 1240, 1500)]
    assert np.all(np.diff(a) > 0)
