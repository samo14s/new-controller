"""Stored results: the numbers the report quotes."""
import os
import numpy as np
import pytest
from objective import fitness
from conftest import path, load_points

MAIN = path("results", "gainmap_main.npz")


@pytest.fixture(scope="module")
def gm():
    return np.load(MAIN)


def test_controlled_never_below_uncontrolled(gm):
    assert np.all(gm["a_best"] >= gm["a_unc"] - 1e-9)


def test_sade_agrees_with_reference(gm):
    rel = np.abs(gm["a_sade"] - gm["a_ref"]) / gm["a_ref"]
    assert (rel < 0.01).sum() >= 248
    assert set(gm["rpm"][rel >= 0.01]) == {2280.0, 2300.0, 2310.0}


def test_high_speed_gain_matches_the_paper(gm):
    g8 = load_points("fig8_gain_points.csv")
    rpm, g = gm["rpm"], gm["g_best"]
    r = g / np.interp(rpm, g8[:, 0], g8[:, 1])
    hi = (rpm >= 2320) & (rpm <= 3000)
    mid = (rpm >= 2060) & (rpm <= 2310)
    assert np.median(r[hi]) == pytest.approx(0.97, abs=0.01)
    assert 0.89 < r[hi].min() and r[hi].max() < 1.05
    assert np.median(r[mid]) == pytest.approx(1.83, abs=0.02)


def test_spot_values(gm):
    i = int(np.argmin(np.abs(gm["rpm"] - 2500)))
    assert gm["g_best"][i] == pytest.approx(535.6, abs=0.5)
    assert gm["a_best"][i] == pytest.approx(6.22, abs=0.01)


@pytest.mark.parametrize("rpm", [700.0, 1300.0, 2500.0, 3000.0])
def test_stored_optimum_is_still_the_feasibility_edge(gm, rpm):
    i = int(np.argmin(np.abs(gm["rpm"] - rpm)))
    g = float(gm["g_best"][i])
    assert fitness(rpm, g) == pytest.approx(float(gm["a_best"][i]), rel=1e-3)
    assert fitness(rpm, g * 1.02) == 0


def test_single_gain():
    sg = np.load(path("results", "single_gain_main.npz"))
    assert float(sg["g_ref"]) == pytest.approx(60.31, abs=0.05)


def test_papers_single_gain_is_infeasible_at_18_speeds():
    bad = [n for n in np.arange(1850, 2210, 10.0) if fitness(n, 237.0) == 0]
    assert len(bad) == 18
    assert bad[0] == 1910 and bad[-1] == 2120


def test_table3_reversal():
    t = np.genfromtxt(path("results", "table3_main.csv"), delimiter=",", names=True)
    assert np.all(t["a_clip1500_td"] > t["a_prop_td"])


def test_extracted_points_are_sorted_and_complete():
    for name, n in [("fig8_gain_points.csv", 214), ("fig9_controlled_points.csv", 250),
                    ("fig9_uncontrolled_points.csv", 2048)]:
        p = load_points(name)
        assert len(p) == n and np.all(np.diff(p[:, 0]) >= 0)


@pytest.mark.skipif(not os.environ.get("OZSOY_PDF"), reason="set OZSOY_PDF to the article PDF")
def test_extractor_reproduces_the_stored_points(tmp_path):
    from extract_vector_figs import main
    main(os.environ["OZSOY_PDF"], str(tmp_path))
    for name in ("fig8_gain_points.csv", "fig9_controlled_points.csv", "fig9_uncontrolled_points.csv"):
        a = load_points(name)
        b = np.loadtxt(tmp_path / name, delimiter=",", skiprows=1)
        assert np.array_equal(a, b)
