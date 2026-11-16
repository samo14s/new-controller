"""C1 results the report quotes (results/c1_*.npz, summarised by src/c1_summary.py)."""
import json
import numpy as np
import pytest
from conftest import path


@pytest.fixture(scope="module")
def s():
    import c1_summary
    return {k: f() for k, f in (("map", c1_summary.map_summary), ("band", c1_summary.band_summary),
                                ("basin", c1_summary.basin_summary),
                                ("gainplane", c1_summary.gainplane_summary))}


def test_summary_file_is_current(s):
    stored = json.load(open(path("results", "c1_summary.json")))
    assert stored["map"]["step_clip"] == s["map"]["step_clip"]
    assert stored["band"]["found"] == s["band"]["found"]


def test_results_are_reproduced_by_the_current_simulator():
    """The C1 results were made with the full surface memory (nreg = 0) and
    the upward kick search.  One stored critical kick is re-derived with the
    current simulator: the kick at the stored value chatters and one
    bisection step below (a factor 2^(1/64)) does not."""
    import c1_unsafe_zone as c
    import params as P
    for name in ("c1_map.npz", "c1_band.npz", "c1_basin.npz", "c1_gainplane.npz"):
        d = np.load(path("results", name))
        assert int(d["nreg"]) == 0, name
    b = np.load(path("results", "c1_basin.npz"))
    k = int(np.where((b["rpm"] == 2500) & np.isclose(b["frac"], 0.9) & (b["vmax"] > 0))[0][0])
    dv = float(b["dv"][k, 0])
    step = 2.0 ** (1.0 / 2 ** c.N_BISECT)
    a = float(b["frac"][k]) * float(b["a_lim"][k]) * 1e-3
    ks = c.KICK_REV * P.STEPS_PER_REV
    assert c._kick_chatters(2500.0, a, float(b["g"][k]), c.VMAX, ks, dv)
    assert not c._kick_chatters(2500.0, a, float(b["g"][k]), c.VMAX, ks, dv / step)


def test_step_entry_zone(s):
    m = s["map"]
    assert m["step_clip"]["below_097"] == 157
    assert m["step_clip"]["ratio_min"] == pytest.approx(0.736, abs=2e-3)
    assert m["step_zone_ratio_min_rpm"] == 2460.0
    assert m["at_2500"]["step_clip"] == pytest.approx(13.00, abs=0.02)


def test_soft_entries_reach_alim(s):
    m = s["map"]
    assert m["edge_clip"]["ratio_min"] >= 0.98 and m["edge_clip"]["below_097"] == 0
    assert m["feed_clip"]["below_097"] <= 1
    assert m["soft_both_ge_097"] >= 250
    assert m["soft_minus_step_mean_mm"] == pytest.approx(2.51, abs=0.02)


def test_unclipped_reference_sits_on_alim(s):
    """Unclipped step entry at or just above a_lim (median 1 % above, up to 5 %
    near the forced resonance at 1970-2030 rpm: the D1 detector calls a small
    cycle just above the onset decaying; soft entries sit on a_lim)."""
    assert 1.0 <= s["map"]["step_free"]["ratio_min"] and s["map"]["step_free"]["ratio_median"] < 1.025


def test_bistable_band(s):
    b = s["band"]
    assert b["found"] == 109 and b["n_speeds"] == 126 and b["nreg"] == 0
    assert b["ratio_median"] == pytest.approx(0.46, abs=0.01)
    assert b["over_alim0_min"] > 1.0          # the band always lies above the uncontrolled limit
    assert not {600.0, 880.0, 900.0} & set(b["none_at_rpm"])     # 'no band' there came from a 3 m/s-only test


def test_basin(s):
    b = s["basin"]
    assert 0.5 < b["td_over_df_min"] and b["td_over_df_max"] < 1.25
    for n in ("1090", "1420", "2500", "2870"):
        assert b[n]["dv_098_mmps"] < b[n]["dv_090_mmps"]
    assert b["2500"]["dv_098_mmps"] == pytest.approx(2.55, abs=0.05)


def test_without_the_limit_only_very_large_kicks_end_in_chatter():
    """Without the voltage limit, 13 of the 16 cases never chatter for a kick
    of the grid (up to 3 m/s); in the other 3 one phase of 16 chatters, for a
    kick of at least 0.78 m/s: 50 times or more the clipped critical kick."""
    d = np.load(path("results", "c1_basin.npz"))
    free, clip = d["vmax"] == 0, d["vmax"] > 0
    fin = np.isfinite(d["dv"][free])
    assert free.sum() == 16 and fin.any(axis=1).sum() == 3 and fin.sum() == 3
    assert d["dv"][free][fin].min() > 0.75
    for i in np.nonzero(fin.any(axis=1))[0]:
        n, f = d["rpm"][free][i], d["frac"][free][i]
        j = np.nonzero(clip & (d["rpm"] == n) & np.isclose(d["frac"], f))[0][0]
        assert d["dv"][free][i][fin[i]].min() > 50 * d["dv"][j].min()


def test_gain_plane_plateau(s):
    g = s["gainplane"]
    assert g["1090"]["step_clip_plateau_mm"] == pytest.approx(19.78, abs=0.1)
    assert g["2500"]["step_clip_plateau_mm"] == pytest.approx(13.19, abs=0.1)
    assert g["2500"]["orbit_saturates_from_g"] == 1650.0


def test_basin_stays_finite_at_0999_of_the_limit():
    """verification/tdsim/check11: at 0.999 a_lim the critical kick (current
    C1 protocol, 4 phases) is 1.4-10.5 mm/s, not zero."""
    import json
    c = json.load(open(path("verification", "tdsim", "check11_kick_near_limit.json")))
    assert c["nreg"] == 0 and c["frac"] == 0.999
    lo = {n: r["min"] for n, r in c["by_rpm"].items()}
    assert lo["2500"] == pytest.approx(1.41, abs=0.01) and lo["2870"] > 7.0 and lo["1090"] > 9.0
    assert all(r["max"] < 11.0 for r in c["by_rpm"].values())

