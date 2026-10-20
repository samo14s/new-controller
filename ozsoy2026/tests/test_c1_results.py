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
    """The C1 results carry the hash of the simulator they came from
    (7b5ced7072ce).  The source has since gained an option that is off by
    default (a full-state perturbation for C2; the default path stays
    bit-identical, see test_c1_entry and test_c2_certificate).  So instead
    of the hash, one stored critical kick is re-derived: the kick at the
    stored value chatters and one bisection step below does not."""
    import c1_unsafe_zone as c
    import params as P
    for name in ("c1_map.npz", "c1_band.npz", "c1_basin.npz", "c1_gainplane.npz"):
        d = np.load(path("results", name))
        assert int(d["nreg"]) == 4, name
    b = np.load(path("results", "c1_basin.npz"))
    k = int(np.where((b["rpm"] == 2500) & np.isclose(b["frac"], 0.9) & (b["vmax"] > 0))[0][0])
    dv = float(b["dv"][k, 0])
    step = np.exp(np.log(c.DV_HI / c.DV_LO) / 2 ** c.N_BISECT)
    a = float(b["frac"][k]) * float(b["a_lim"][k]) * 1e-3
    ks = c.KICK_REV * P.STEPS_PER_REV
    assert c._kick_chatters(2500.0, a, float(b["g"][k]), c.VMAX, ks, dv)
    assert not c._kick_chatters(2500.0, a, float(b["g"][k]), c.VMAX, ks, dv / step)


def test_step_entry_zone(s):
    m = s["map"]
    assert m["step_clip"]["below_097"] == 154
    assert m["step_clip"]["ratio_min"] == pytest.approx(0.749, abs=2e-3)
    assert m["step_zone_ratio_min_rpm"] == 2460.0
    assert m["at_2500"]["step_clip"] == pytest.approx(13.26, abs=0.02)


def test_soft_entries_reach_alim(s):
    m = s["map"]
    assert m["edge_clip"]["ratio_min"] >= 0.98 and m["edge_clip"]["below_097"] == 0
    assert m["feed_clip"]["below_097"] <= 1
    assert m["soft_both_ge_097"] >= 250
    assert m["soft_minus_step_mean_mm"] == pytest.approx(2.25, abs=0.02)


def test_unclipped_reference_sits_on_alim(s):
    """Unclipped step entry within 0-2.5 % above a_lim (the D1 detector calls a
    small cycle just above a_lim decaying; soft entries sit on a_lim)."""
    assert 1.0 <= s["map"]["step_free"]["ratio_min"] and s["map"]["step_free"]["ratio_median"] < 1.025


def test_bistable_band(s):
    b = s["band"]
    assert b["found"] == 108 and b["n_speeds"] == 126
    assert b["ratio_median"] == pytest.approx(0.42, abs=0.01)
    assert b["over_alim0_min"] > 1.0          # the band always lies above the uncontrolled limit


def test_basin(s):
    b = s["basin"]
    assert b["free_all_inf"]                   # without the limit no kick up to 3 m/s causes chatter
    assert 0.5 < b["td_over_df_min"] and b["td_over_df_max"] < 1.25
    for n in ("1090", "1420", "2500", "2870"):
        assert b[n]["dv_098_mmps"] < b[n]["dv_090_mmps"]
    assert b["2500"]["dv_098_mmps"] == pytest.approx(2.56, abs=0.05)


def test_gain_plane_plateau(s):
    g = s["gainplane"]
    assert g["1090"]["step_clip_plateau_mm"] == pytest.approx(20.35, abs=0.1)
    assert g["2500"]["step_clip_plateau_mm"] == pytest.approx(13.42, abs=0.1)
    assert g["2500"]["orbit_saturates_from_g"] == 1650.0
