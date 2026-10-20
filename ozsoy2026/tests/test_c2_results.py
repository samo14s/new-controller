"""C2 results the report quotes (results/c2_*.npz, summarised by src/c2_summary.py)."""
import json
import numpy as np
import pytest
from conftest import path


@pytest.fixture(scope="module")
def s():
    return json.load(open(path("results", "c2_summary.json")))


def test_summary_is_current(s):
    v = np.load(path("results", "c2_validate.npz"))
    assert s["validate"]["offaxis_runs"] == int(v["offaxis_runs"].sum())
    assert s["validate"]["below_td"] == int(np.sum(v["cert"][np.isfinite(v["td_min"])] <=
                                                 v["td_min"][np.isfinite(v["td_min"])]))


def test_certificate_is_checked_and_holds_on_its_own_model(s):
    v = s["validate"]
    assert v["checked"] == v["points"] == 48
    assert v["below_model"] == v["model_points"] == 44
    assert v["proof_vmax"] <= 1.0
    assert v["delta"] == 0.5


def test_certificate_below_the_simulated_cut(s):
    v = s["validate"]
    assert v["below_td"] == v["td_points"] == 29
    assert v["td_over_cert_min"] >= 1.0
    assert v["offaxis_bad"] == 0 and v["offaxis_runs"] == 3456


def test_averaged_model_is_slightly_more_fragile_than_the_cut(s):
    v = s["validate"]
    assert 0.6 < v["model_over_td_min"] and v["model_over_td_max"] < 1.0
    assert v["model_over_td_median"] == pytest.approx(0.92, abs=0.01)


def test_step_entry_certified_only_where_the_simulation_agrees(s):
    for n, d in s["step"]["by_rpm"].items():
        for f in d["certified_fracs"]:
            assert f < d["td_step_limit_frac"], (n, f)
    assert s["step"]["by_rpm"]["2500"]["certified_fracs"] == []
    assert s["step"]["by_rpm"]["1090"]["certified_fracs"] == [0.3]
