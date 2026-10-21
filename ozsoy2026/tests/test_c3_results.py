"""C3 results the report quotes (results/c3_*.npz, summarised by src/c3_summary.py)."""
import json
import os
import numpy as np
import pytest
from conftest import path


@pytest.fixture(scope="module")
def s():
    return json.load(open(path("results", "c3_summary.json")))


def test_summary_is_current(s):
    """The stored summary is what c3_summary.build() gives from the stored
    results (the errata are pinned by the slow test below)."""
    import c3_summary
    new = json.loads(json.dumps(c3_summary.build(errata=False)))
    assert new == {k: v for k, v in s.items() if k != "errata"}


@pytest.mark.slow
def test_errata_are_current(s):
    import c3_summary
    assert json.loads(json.dumps(c3_summary.margin_errata())) == s["errata"]


def test_averaging_is_conservative_for_the_helical_cutter(s):
    """Floquet limit of the periodic model never below the ZOA on C2's gain map,
    no flip lobe, collocation converged."""
    for ae in ("0.5", "0.1"):
        f = s["floquet"][ae]
        assert f["below_zoa"] == 0 and f["flip"] == 0
        assert 1.0 <= f["floquet_over_zoa"]["min"] and f["floquet_over_zoa"]["max"] < 1.003
        assert f["node_convergence"] < 1e-5


def test_basin_certified_on_the_periodic_model(s):
    b = s["basin"]
    assert b["points"] == 48 and b["certified"] == 47
    assert b["failed"] == ["2500 rpm 0.98: no candidate passed the check"]
    assert b["phase_cert_below_model"][0] == b["phase_cert_below_model"][1] == 688      # finite model kicks only
    assert b["phase_cert_below_td"][0] == b["phase_cert_below_td"][1] == 448
    assert b["cert_below_td_min"][0] == b["cert_below_td_min"][1] == 28
    assert b["offaxis"] == [0, 3384]
    assert b["proof_vmax"] < 1.0
    assert b["c3_over_c2"]["median"] == pytest.approx(1.064, abs=0.002)
    assert b["c3_above_c2"] == 44
    assert b["c3_over_c2_by_rpm"]["2500"]["max"] > 2.0


def test_averaging_does_not_change_the_basin(s):
    """Phase by phase the periodic and averaged models' critical kicks agree to
    about 5 %, and C2's averaged certificate lies below the periodic model's
    critical kick at every point."""
    for k in ("basin", "lowimm"):
        q = s[k]["periodic_over_averaged_model"]
        assert 0.97 < q["min"] and q["max"] < 1.06
        assert s[k]["c2_below_periodic_model"][0] == s[k]["c2_below_periodic_model"][1]


def test_low_immersion(s):
    L = s["lowimm"]
    assert L["certified"] == L["points"] == 8
    assert L["offaxis"] == [0, 576] and L["proof_vmax"] < 1.0
    assert L["phase_cert_below_td"][0] == L["phase_cert_below_td"][1] == 19
    assert 0.94 < L["c3_over_c2"]["min"] and L["c3_over_c2"]["max"] < 1.04


def test_what_each_ingredient_adds(s):
    """At 0.9 a_lim: the periodic coefficient alone costs, the time-resolved
    margin gains, and the certificate grows with the subintervals."""
    d = s["decomp"]
    assert d["periodic, single level"]["max"] < 1.0 < d["averaged, resolved"]["min"]
    m8, m16, m32 = (d[f"periodic, resolved, m_c {m}"]["median"] for m in (8, 16, 32))
    assert m8 < m16 < m32


def test_flip_lobes_only_where_c2_claims_nothing(s):
    """Straight teeth: below the ZOA limit without control and with weak DVF;
    with DVF the forced orbit alone then needs far more than 9 V."""
    st = s["straight"]
    assert st["0"]["unstable_speeds"] == 12 and st["0"]["from_frac"] == pytest.approx(0.947, abs=1e-3)
    assert st["250"]["unstable_speeds"] == st["500"]["unstable_speeds"] == 0
    for g in ("100", "map"):
        assert st[g]["unstable_speeds"] > 0 and st[g]["bmax_where_unstable_V"][0] > 9.0


def test_counterexample_to_averaging(s):
    """Same averaged certificate for both cutters; only the helical cut is stable."""
    c = s["counter"]
    assert c["straight"]["c2_mm_s"] == c["helical"]["c2_mm_s"] > 0
    assert c["straight"]["mu_abs"] > 1.0 and abs(abs(c["straight"]["mu_arg_deg"]) - 180) < 1
    assert c["straight"]["c3_mm_s"] == 0.0 and not c["straight"]["simulated_stable"]
    assert c["helical"]["mu_abs"] < 1.0 and c["helical"]["c3_mm_s"] > 0 and c["helical"]["simulated_stable"]


def test_command_bound_errata_change_no_quoted_number(s):
    """C2's and C4's certificates rescaled to the closed-form command bound:
    a few mV, and no number quoted for C2 or C4 changes."""
    e = s["errata"]
    assert e["c2_validate"]["extra_mV"]["max"] < 3.5 and e["c2_validate"]["size_factor"]["min"] > 0.997
    assert e["c2_surface"]["kick_090_mmps"][0] == e["c2_surface"]["kick_090_mmps"][1]
    a, b = e["c4_gainmap"]["certified_without_gap_mm"]
    assert a == b
    assert e["c2_step"]["certified"][0] == e["c2_step"]["certified"][1]
    v = e["c4_validate"]
    assert v["cert_below_td"][0] == v["cert_below_td"][1] == v["cert_below_td"][2] == 89
