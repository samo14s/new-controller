"""Time-domain simulator: convergence, force model, actuator limit."""
import json
import numpy as np
import pytest
import params as P
from tdsim import simulate, simulate_exact_python
from zoa import a_lim
from conftest import path


def test_delay_must_be_whole_steps():
    with pytest.raises(AssertionError):
        simulate(1000, 1e-3, 0.0, spr=510, nrev=1)


def test_mean_force_matches_closed_form():
    r = simulate(2000, 1e-3, 0.0, store=True, nrev=20, spr=2048)
    mean = r["fcut"][10 * 2048:20 * 2048].mean()
    exact = -(P.N_TEETH * 1e-3 * P.FZ / (2 * np.pi)) * (P.K_TC * np.pi / 4 + P.K_RC / 2)
    assert mean == pytest.approx(exact, rel=5e-3)


def test_stagewise_force_converged_at_512_steps():
    """The final scheme at the paper's 512 steps/rev against 4096."""
    for rpm, a, g in [(500, 1.5e-3, 0.0), (1300, 5e-3, 600.0), (3000, 10e-3, 1200.0)]:
        x = simulate(rpm, a, g, store=True, nrev=10)["x"]
        xr = simulate(rpm, a, g, store=True, nrev=10, spr=4096)["x"][::8]
        assert np.abs(x - xr).max() / np.abs(xr).max() < 1e-2


def test_held_force_scheme_matches_its_exact_discretisation():
    x = simulate(1300, 5e-3, 600.0, store=True, nrev=5, fmode=0, nsub=8)["x"]
    xr, _ = simulate_exact_python(1300, 5e-3, 600.0, nrev=5)
    assert np.abs(x - xr).max() / np.abs(xr).max() < 1e-6


@pytest.mark.parametrize("rpm,g", [(1300, 991.0), (2500, 535.0)])
def test_demand_at_alim_step_converged(rpm, g):
    al = a_lim(rpm, g)
    d512 = simulate(rpm, al, g)["max_demand"]
    d4096 = simulate(rpm, al, g, spr=4096)["max_demand"]
    assert d512 == pytest.approx(d4096, rel=5e-3)


def test_voltage_limit_is_respected():
    vmax = P.F_MAX / P.GA_GAIN
    r = simulate(2500, 10e-3, 3000.0, vmax=vmax, store=True, nrev=20)
    assert r["max_v"] <= vmax + 1e-12
    assert np.allclose(r["vcmd"][:-1], np.clip(-3000.0 * r["v"][:-1], -vmax, vmax))


def test_stored_final_scheme_check():
    c = json.load(open(path("verification", "tdsim", "check7_final_scheme.json")))
    assert max(t["rel_err_fmode1"] for t in c["trajectory"]) < 7e-3
    assert max(abs(d["rel"]) for d in c["demand"]) < 2e-3
    assert all(0 <= b["rel"] < 0.03 for b in c["boundary"])
