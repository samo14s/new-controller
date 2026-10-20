"""C4 results the report quotes (results/c4_*.npz, summarised by src/c4_summary.py)."""
import json
import numpy as np
import pytest
from conftest import path


@pytest.fixture(scope="module")
def s():
    return json.load(open(path("results", "c4_summary.json")))


def test_summary_is_current(s):
    """The stored summary is what c4_summary.build() gives from the stored results."""
    import c4_summary
    assert json.loads(json.dumps(c4_summary.build())) == s


def test_feasible_dvf_is_the_gmax3000_optimum():
    """The DVF baseline of C4 is feasibility.py's optimum with g <= 3000."""
    L = np.load(path("results", "c4_limits.npz"))
    d = np.load(path("results", "feas_g3000.npz"))
    names = list(L["feas_names"])
    assert np.allclose(L["feas_g1"][:, names.index("dvf")], np.interp(L["rpm"], d["rpm"], d["g_best"]), atol=0.5)
    assert np.allclose(L["feas_a_lim"][:, names.index("dvf")], np.interp(L["rpm"], d["rpm"], d["a_best"]), rtol=2e-3)


def test_delayed_terms_raise_the_linear_limit_at_every_speed(s):
    f = s["limits"]["feasible"]
    for n in ("d1000", "d2000", "d4000", "p3e5", "p1e6"):
        assert f[n]["ratio_to_dvf"]["min"] > 1.0, n
    assert f["d2000"]["ratio_to_dvf"]["median"] == pytest.approx(1.78, abs=0.02)
    assert f["d1000"]["ratio_to_dvf"]["median"] < f["d2000"]["ratio_to_dvf"]["median"] < f["d4000"]["ratio_to_dvf"]["median"]


def test_face_entry_follows_the_linear_limit_and_step_entry_does_not(s):
    e = s["entry"]
    assert e["edge_clip"]["feas_d2000"]["over_a_lim"]["median"] >= 0.99
    assert e["edge_clip"]["d2000_over_dvf"]["min"] > 1.3
    # step entry: no gain where it is set by saturation (high speed), never much worse
    assert e["step_clip"]["d2000_over_dvf"]["min"] > 0.95
    assert e["step_clip"]["feas_d2000"]["over_a_lim"]["median"] < 0.6


def test_certificate_below_the_simulation_and_its_model(s):
    v = s["validate"]
    assert v["cert_below_td"] == v["td_points"] > 0
    assert v["cert_below_model"] == v["model_points"] > 0
    assert v["proof_vmax"] <= 1.0
    assert v["offaxis_bad"] == 0 and v["offaxis_runs"] > 0


def test_closed_form_matches_the_simulator(s):
    rb = s["robust"]
    assert 0.98 < rb["td_limit_ratio"]["min"] and rb["td_limit_ratio"]["max"] < 1.02
    R = np.load(path("results", "c4_robust.npz"))
    ratio = R["td_orbit"][:, 0] / R["td_orbit"][:, 1]
    small = R["td_orbit"][:, 1] < 2.0          # a command of ~1 V (helix cancellation): absolute error instead
    assert np.all(np.abs(ratio[~small] - 1) < 0.02)
    assert np.all(np.abs(R["td_orbit"][small, 0] - R["td_orbit"][small, 1]) < 0.3)
    for e, (td, lin) in rb["td_face_eps"].items():
        assert td == pytest.approx(lin, rel=0.01), e


def test_robustness_limits_of_the_tooth_comb(s):
    """Runout: the tooth comb tolerates less than DVF at every basin speed;
    an unmodelled 400 Hz mode with 400 us of loop delay destabilises it."""
    for n, row in s["robust"]["by_rpm"].items():
        assert row["tooth comb"]["runout_tolerance_um"] <= row["DVF"]["runout_tolerance_um"], n
        assert row["tooth comb"]["tau_a_lim_mm"]["mode_400Hz"][-1] == 0.0, n
        assert row["DVF"]["tau_a_lim_mm"]["mode_400Hz"][-1] > 10.0, n
