"""C3: the periodic model, the enclosures and the periodic certificate."""
import os
import sys
import numpy as np
import pytest
import params as P
import c2_certificate as c2
import c3_periodic as c3
from zoa import a_lim
from conftest import path


def test_half_immersion_cut_is_c2s():
    """At the paper's immersion the cut and the forced orbit are C2's: the
    same mean coefficient, the slice-force harmonics in closed form (C2's
    4096-point DFT is off by at most 4.1 N/m, the entry jump falling on a
    sample), and a command at most 4 mV from C2's (that error plus
    the harmonics above N_HARM)."""
    import feasibility as fe
    assert c3.HALF.kappa == c2.KAPPA
    assert np.abs(c3.HALF.W - fe._W).max() < 4.1
    rpm, g = 2500.0, 1500.0
    a = 0.6 * a_lim(rpm, g)
    s0, b0, bmax = c2.orbit(rpm, g, a)
    s1, b1, bk = c3.orbit(rpm, g, a)
    assert np.allclose(s1, s0, rtol=1e-3, atol=1e-3 * np.abs(s0).max())
    assert np.abs(b1 - b0).max() < 4e-3
    assert bmax < bk.max() < bmax + 4e-3
    assert np.all(bk >= np.abs(b1).reshape(len(bk), -1).max(axis=1))


def test_slice_force_harmonics_in_closed_form():
    """Against a 2^20-point midpoint quadrature of the same force (all teeth),
    at half and low immersion; and the bound on the harmonics above N_HARM is
    below 2 mV of command at g = 1500."""
    for ae in (0.5, 0.1):
        cut = c3.Cut(ae)
        n, pitch = 2 ** 20, 2 * np.pi / P.N_TEETH
        th = (np.arange(n) + 0.5) / n * pitch
        w = np.zeros(n)
        for i in range(P.N_TEETH):
            phi = np.mod(th + i * pitch, 2 * np.pi)
            on = (phi >= cut.st) & (phi <= cut.ex)
            w += np.where(on, -(P.K_TC * np.sin(phi) - P.K_RC * np.cos(phi)) * P.FZ * np.sin(phi), 0.0)
        m = np.arange(c3.N_HARM + 1)
        W = (np.fft.rfft(w) / n)[:len(m)] * np.exp(1j * m * P.N_TEETH * 0.5 / n * pitch)
        assert np.abs(W - cut.W).max() < 0.1
        for rpm in (1090.0, 2870.0):
            assert 0 < 1500.0 * c3.velocity_tail(rpm, 1500.0, 0.9 * cut.a_lim(rpm, 1500.0), cut) < 2e-3


def test_coefficient_is_the_independent_helical_integral():
    """a H(t) equals verification/zoa_sdm's exact z-integral (same angle
    convention as the simulator) and averages to C2's KAPPA a."""
    sys.path.insert(0, path("verification", "zoa_sdm"))
    import model as VM
    rpm = 2500.0
    th = np.linspace(0, 2 * np.pi / P.N_TEETH, 3001)
    for a in (3e-3, 12e-3, 20e-3):
        h = c3.aH(th / (2 * np.pi * rpm / 60), rpm, a)
        assert np.allclose(h, VM.aH_helical(th, a), rtol=1e-12, atol=1e-6 * abs(c2.KAPPA * a))
        assert h[:-1].mean() == pytest.approx(c2.KAPPA * a, rel=1e-6)


def test_rate_bound_and_enclosures_contain_the_coefficient():
    """|d(aH)/dt| <= L, and both enclosures contain a H(t) on a grid 20 x
    denser than their samples, at half and low immersion."""
    for cut, a in ((c3.HALF, 3e-3), (c3.HALF, 15e-3), (c3.Cut(0.1), 10e-3)):
        rpm = 2500.0
        Tp = 60.0 / (rpm * P.N_TEETH)
        t = np.linspace(0, Tp, 200001)
        h = c3.aH(t, rpm, a, cut)
        assert np.abs(np.diff(h) / np.diff(t)).max() <= c3.aH_rate_bound(rpm, cut)
        for order in (1, 2):
            al, be, ep = c3.enclosure(rpm, a, cut, mc=8, order=order)
            k = np.minimum((t / Tp * 8).astype(int), 7)
            s = t / Tp * 8 - k
            assert np.all(np.abs(h - al[k] - be[k] * s) <= ep[k])


def test_second_order_enclosure_is_much_tighter():
    rpm, a = 2500.0, 0.9 * a_lim(2500.0, 1500.0)
    e1 = c3.enclosure(rpm, a, mc=16, order=1)[2]
    e2 = c3.enclosure(rpm, a, mc=16, order=2)[2]
    assert np.all(e2 <= e1) and np.median(e2) < 0.1 * np.median(e1)
    assert np.median(e2) < 1e-3 * abs(c2.KAPPA * a)


def test_corners_reproduce_the_decrease_matrix_and_cover_the_curve():
    """With sigma = s^2 the corner parametrisation is exactly A(t) W(s), and
    (s, s^2) lies in the triangle of the corners (nonnegative barycentric
    coordinates), so M1 on the subinterval is a convex combination of the
    corner values."""
    rng = np.random.default_rng(1)
    n = 6
    A0, E = rng.standard_normal((n, n)), np.outer(rng.standard_normal(n), rng.standard_normal(n))
    Wk, Wk1 = (lambda X: X @ X.T)(rng.standard_normal((n, n))), (lambda X: X @ X.T)(rng.standard_normal((n, n)))
    al, be = 2.0, -0.7
    for sv in np.linspace(0, 1, 11):
        for r in (-0.3, 0.1):
            sg = sv * sv
            ck = (al + r) * (1 - sv) + be * (sv - sg)
            ck1 = (al + r) * sv + be * sg
            lhs = ((1 - sv) * A0 + ck * E) @ Wk + (sv * A0 + ck1 * E) @ Wk1
            W = (1 - sv) * Wk + sv * Wk1
            assert np.allclose(lhs, (A0 + (al + be * sv + r) * E) @ W)
        # barycentric coordinates of (s, s^2) in (0,0), (1/2,0), (1,1)
        lam3 = sv * sv
        lam2 = 2 * (sv - lam3)
        lam1 = 1 - lam2 - lam3
        assert min(lam1, lam2, lam3) >= -1e-15
    cs = c3.corners(np.array([al]), np.array([be]), np.array([0.3]))[0]
    assert len(cs) == 6
    assert len(c3.corners(np.array([al]), np.array([0.0]), np.array([0.3]))[0]) == 4


def test_petersen_merge_is_exact_for_a_scalar():
    """M0 - r (p v' + v p') > 0 on |r| <= eps (both ends) holds exactly when
    some mu > 0 makes [[M0 - mu p p', eps v], [eps v', mu]] > 0."""
    rng = np.random.default_rng(3)
    n = 5
    for trial in range(200):
        X = rng.standard_normal((n, n))
        M0 = X @ X.T + 0.2 * np.eye(n)
        p, v = rng.standard_normal(n), rng.standard_normal(n)
        eps = rng.uniform(0.05, 1.5)
        ends = all(np.linalg.eigvalsh(M0 - r * (np.outer(p, v) + np.outer(v, p))).min() > 0 for r in (-eps, eps))
        mus = np.geomspace(1e-4, 1e4, 4001)
        merged = any(np.linalg.eigvalsh(np.block([[M0 - m * np.outer(p, p), eps * v[:, None]],
                                                  [eps * v[None, :], np.array([[m]])]])).min() > 0 for m in mus)
        assert ends == merged, trial


def test_periodic_lyapunov_interpolation():
    rng = np.random.default_rng(0)
    W = np.array([(lambda X: X @ X.T)(rng.standard_normal((3, 3))) for _ in range(4)])
    cert = (np.eye(3), W, None, None)
    Wt = c3.W_at(cert, np.arange(16), 16)
    assert np.allclose(Wt[::4], W)
    assert np.allclose(Wt[2], 0.5 * (W[0] + W[1]))
    assert np.allclose(c3.W_at(cert, [15], 16)[0], 0.25 * W[3] + 0.75 * W[0])     # wraps to W_0


def test_floquet_limit_of_the_collocated_periodic_model():
    """The periodic collocated model's limit is within 0.2 % of the ZOA
    (slightly above it) at 2500 rpm, and its spectral radius converges in
    the number of history nodes."""
    rpm, g = 2500.0, 1500.0
    af = c3.floquet_a_lim(rpm, g, lo_frac=0.9, ratio=1.01)
    az = a_lim(rpm, g)
    assert 1.0 < af / az < 1.002
    N = c2.nodes_for(rpm)
    r1 = c3.spectral_radius(rpm, 0.95 * az, g, N=N)
    r2 = c3.spectral_radius(rpm, 0.95 * az, g, N=N + 8)
    assert abs(r1 - r2) < 1e-6


@pytest.mark.slow
def test_floquet_limit_against_the_semi_discretisation():
    """Against the independent first-order SDM (verification/zoa_sdm, m and
    2m intervals with Richardson extrapolation) for the helical cutter."""
    sys.path.insert(0, path("verification", "zoa_sdm"))
    import sdm
    rpm, g = 2500.0, 1500.0
    af = c3.floquet_a_lim(rpm, g, lo_frac=0.9, ratio=1.01)
    ref = sdm.boundary_richardson(rpm, g, P.ZETA, helix=True)["a_R"]
    assert af == pytest.approx(ref, rel=2e-3)


@pytest.mark.slow
def test_reduces_to_c2_with_one_subinterval_and_the_mean_coefficient():
    """One subinterval, the averaged coefficient and C2's single level: the
    periodic LMIs are C2's, and the size is C2's to solver tolerance."""
    rpm, g = 2500.0, 1500.0
    a = 0.5 * a_lim(rpm, g)
    r = c3.certify(rpm, a, g, mc=1, averaged=True, resolved=False, full=True)
    assert r["size"] == pytest.approx(r["c2_size"], rel=0.02)
    # the warm start is C2 at C3's command bound (closed-form harmonics, a few mV above C2's)
    A, B, K = c2.model(rpm, a, g)
    e = np.zeros(len(B))
    e[1] = 1.0
    u0 = c3.VMAX - c3.orbit(rpm, g, a)[2].max()
    assert r["c2_size"] == pytest.approx(c2.certify(A, B, K, u0, e, rpm)[0], rel=1e-4)   # A0 + KAPPA a E: round-off
    assert r["c2_size"] == pytest.approx(c2.certify_kick(rpm, a, g)[0], rel=2e-3)


@pytest.mark.slow
def test_merged_and_corner_lmis_give_the_same_certificate():
    rpm, g = 2500.0, 1500.0
    a = 0.9 * a_lim(rpm, g)
    A0, E, B, K = c3.model(rpm, g)
    n, Tp = len(B), 60.0 / (rpm * P.N_TEETH)
    e = np.zeros(n)
    e[1] = 1.0
    _, _, bk = c3.orbit(rpm, g, a, mc=4)
    u0n = np.minimum(c3.VMAX - bk, np.roll(c3.VMAX - bk, 1))
    s2, _, cert2 = c2.certify(A0 + c2.KAPPA * a * E, B, K, (c3.VMAX - bk).min(), e, rpm, full=True)
    from scipy.linalg import cholesky
    Tm = cert2[0] @ cholesky(np.linalg.inv(cert2[1]), lower=True)
    enc = c3.enclosure(rpm, a, mc=4)
    out = []
    for merged in (True, False):
        beta, (W, Y, S), _ = c3.lmi(A0, E, B, K, enc, u0n, e, Tm, Tp, lam=0.3, valid=c2.validity(n), merged=merged)
        out.append(c3.check(A0, E, B, K, enc, u0n, e, Tm, Tp, W, Y, S, valid=c2.validity(n))[0])
    assert out[0] > 0 and out[0] == pytest.approx(out[1], rel=2e-3)
