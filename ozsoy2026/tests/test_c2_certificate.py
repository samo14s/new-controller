"""C2: the saturated-LMI certificate, its model and its checks."""
import numpy as np
import pytest
import params as P
import c2_certificate as c2
from zoa import a_lim


def test_chebyshev_differentiation_is_exact_on_polynomials():
    D, x = c2.cheb(12)
    for k in range(1, 12):
        assert np.allclose(D @ x ** k, k * x ** (k - 1), atol=1e-9)


@pytest.mark.parametrize("rpm,g", [(500.0, 1500.0), (1090.0, 1500.0), (2500.0, 1500.0), (2500.0, 0.0)])
def test_collocated_model_crosses_at_alim(rpm, g):
    """The rightmost eigenvalue of the collocated ODE changes sign at the
    ZOA limit (to 0.1 %)."""
    al = a_lim(rpm, g)
    lead = lambda a: np.linalg.eigvals(c2.model(rpm, a, g)[0]).real.max()
    assert lead(0.999 * al) < 0 < lead(1.001 * al)


def test_psd_triangle_convention():
    """max y s.t. [[1, 0, y], [0, 1, 0], [y, 0, 1]] >= 0 -> y = 1, and an
    off-diagonal variable in a later column."""
    B = c2.LMIBlock(3)
    for i in range(3):
        B.add(i, i, -1, 1.0)
    B.add(0, 2, 0, 1.0)
    x, st = c2.solve_sdp(1, [1.0], [B])
    assert x[0] == pytest.approx(1.0, abs=1e-6)
    B = c2.LMIBlock(2)
    B.add(0, 0, -1, 2.0)
    B.add(1, 1, -1, 0.5)
    B.add(0, 1, 0, 1.0)
    assert c2.solve_sdp(1, [1.0], [B])[0][0] == pytest.approx(1.0, abs=1e-6)


def test_sector_condition_for_the_moving_deadzone():
    """psi(w) = w - (sat(b + w) - b): with |b| <= b_max < V and
    |w - v| <= V - b_max, psi (psi - v) <= 0 (random samples)."""
    rng = np.random.default_rng(1)
    V, bmax = 9.0, 6.5
    b = rng.uniform(-bmax, bmax, 200000)
    w = rng.uniform(-40, 40, 200000)
    v = w + rng.uniform(-(V - bmax), V - bmax, 200000)
    psi = w - (np.clip(b + w, -V, V) - b)
    assert np.all(psi * (psi - v) <= 1e-9)


def test_orbit_command_and_its_bound():
    """b(t) equals -g times the closed-form forced velocity, and b_max bounds
    a 16x denser sampling."""
    from feasibility import forced_velocity
    rpm, g = 2500.0, 1500.0
    a = 0.5 * a_lim(rpm, g)
    _, b, bmax = c2.orbit(rpm, g, a, n_t=256)
    vf = forced_velocity(rpm, g, [a], n_t=256)[0]
    assert np.allclose(b, -g * vf, atol=1e-9 * np.abs(b).max())
    _, bd, _ = c2.orbit(rpm, g, a, n_t=65536)
    assert np.abs(bd).max() <= bmax


def test_orbit_state_matches_settled_simulation():
    """x(0), x'(0) of the closed-form orbit against a settled unclipped run
    at a whole number of tooth periods."""
    from tdsim import simulate
    rpm, g, a = 2500.0, 1500.0, 10e-3
    x0, _, _ = c2.orbit(rpm, g, a)
    r = simulate(rpm, a, g, store=True, nrev=300, edge=True, nreg=4)
    assert r["x"][-1] == pytest.approx(x0[0], rel=0.02)
    assert r["v"][-1] == pytest.approx(x0[1], rel=0.05, abs=0.05 * np.abs(r["v"][-512:]).max())


@pytest.fixture(scope="module")
def small():
    """A small collocation (N = 8) at 2500 rpm, 0.9 a_lim: fast to certify."""
    rpm, g = 2500.0, 1500.0
    a = 0.9 * a_lim(rpm, g)
    A, B, K = c2.model(rpm, a, g, N=8)
    _, b, bmax = c2.orbit(rpm, g, a, N=8)
    e = np.zeros(len(B))
    e[1] = 1.0
    size, note, cert = c2.certify(A, B, K, c2.VMAX - bmax, e, rpm, full=True)
    return dict(rpm=rpm, A=A, B=B, K=K, b=b, u0=c2.VMAX - bmax, e=e, size=size, cert=cert)


def test_small_certificate_is_checked_and_holds_on_its_model(small):
    s = small
    assert s["size"] > 0
    vmax, vend = c2.proof_check(s["A"], s["B"], s["K"], s["b"], 60.0 / (s["rpm"] * P.N_TEETH), s["cert"],
                                nruns=24, nper=20)
    assert vmax <= 1.0 and vend < vmax


def test_check_rejects_a_destabilised_model(small):
    """The certificate of the cut must fail the decrease test once the
    depth is past a_lim."""
    s = small
    Tm, Pz = s["cert"]
    W = np.linalg.inv(Pz)
    A2, _, _ = c2.model(s["rpm"], 1.05 * a_lim(s["rpm"], 1500.0), 1500.0, N=8)
    size, msg = c2.check(A2, s["B"], s["K"], s["u0"], s["e"], Tm, W, np.zeros(len(s["B"])), 1.0)
    assert size == 0.0 and "decrease" in msg


def test_check_size_scales_with_the_margin(small):
    """Halving u0 at a fixed candidate halves the size (the inclusion shrinks
    (W, Y) by u0^2)."""
    s = small
    Tm = np.diag(c2.scaling(s["A"], s["rpm"]))
    sig = -np.linalg.eigvals(s["A"]).real.max()
    beta, (W, Y, S), _ = c2.lmi(s["A"], s["B"], s["K"], s["u0"], s["e"], Tm, lam=0.05 * sig)
    full = c2.check(s["A"], s["B"], s["K"], s["u0"], s["e"], Tm, W, Y, S)[0]
    half = c2.check(s["A"], s["B"], s["K"], s["u0"] / 2, s["e"], Tm, W, Y, S)[0]
    assert full > 0 and half == pytest.approx(full / 2, rel=1e-6)


def test_certified_set_respects_the_validity_bound(small):
    """|x - x(t - T)| <= DELTA fz everywhere in the certified ellipsoid."""
    s = small
    Tm, Pz = s["cert"]
    n = len(s["B"])
    c, h = c2.validity(n)
    cz = c @ Tm
    assert np.sqrt(cz @ np.linalg.inv(Pz) @ cz) <= h * (1 + 1e-9)


def test_full_state_perturbation_reduces_to_the_kick():
    from tdsim import simulate
    rpm, a, g = 2500.0, 12e-3, 1500.0
    X = np.zeros(4 + c2.nodes_for(rpm))
    X[1] = 5e-3
    ks = 100 * P.STEPS_PER_REV + 17
    r1 = simulate(rpm, a, g, vmax=9.0, store=True, nrev=110, edge=True, nreg=4, kick_step=ks, kick_dv=5e-3)
    r2 = simulate(rpm, a, g, vmax=9.0, store=True, nrev=110, edge=True, nreg=4, kick_step=ks,
                  pert=c2.deviation_to_pert(rpm, X))
    assert np.array_equal(r1["x"], r2["x"])


def test_history_of_a_deviation_is_interpolated_exactly():
    rpm = 2500.0
    T = 60.0 / (rpm * P.N_TEETH)
    w = 2 * np.pi * 130.0
    th = c2.history_nodes(rpm)
    X = np.concatenate([[1e-6, 0.0, 0.0, 0.0], 1e-6 * np.cos(w * th)])
    p = c2.deviation_to_pert(rpm, X)
    nd = P.STEPS_PER_REV // P.N_TEETH
    tt = -np.arange(nd + 1) * T / nd
    assert np.abs(p["hx"] - 1e-6 * np.cos(w * tt)).max() < 1e-12
    assert np.abs(p["hv"] + 1e-6 * w * np.sin(w * tt)).max() < 1e-9 * w
