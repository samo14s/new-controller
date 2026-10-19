"""C1 simulator options: soft entries, kick, delayed clip-on, and the DF tools."""
import hashlib
import json
import numpy as np
import pytest
import params as P
from tdsim import simulate, entry_revs
from conftest import path

RPM, A, G = 1300, 5e-3, 600.0          # a stable controlled cut (a_lim 7.75 mm)
SPR = P.STEPS_PER_REV
EXACT_MEAN = -(P.N_TEETH * A * P.FZ / (2 * np.pi)) * (P.K_TC * np.pi / 4 + P.K_RC / 2)


def mean_force(r, r0, r1):
    return r["fcut"][r0 * SPR:r1 * SPR].mean() / EXACT_MEAN


def test_defaults_bit_identical_to_previous_build():
    """With every option off, the state trajectory equals the build before
    the options were added, bit for bit (the logged actuator force differs by
    floating-point contraction only)."""
    ref = json.load(open(path("verification", "c1", "reference_defaults.json")))["cases"]
    for c in ref:
        r = simulate(c["rpm"], c["a"], c["g"], vmax=c["vmax"], store=True, nrev=c["nrev"],
                     fmode=c["fmode"])
        for k in ("x", "v", "fcut", "vcmd"):
            assert hashlib.sha256(r[k].tobytes()).hexdigest() == c[f"sha256_{k}"], (c, k)
        assert np.abs(r["fact"]).max() == pytest.approx(c["fact_absmax"], rel=1e-12, abs=1e-12)


def test_soft_path_reduces_to_step_entry():
    """A feed ramp that keeps the feed constant runs the soft-entry branch
    of the force model; it must reproduce the step entry."""
    d = simulate(RPM, A, G, store=True, nrev=30)
    s = simulate(RPM, A, G, store=True, nrev=30, ramp_f=5, f0=1.0)
    assert np.abs(s["x"] - d["x"]).max() < 1e-12 * np.abs(d["x"]).max()


def test_zero_kick_and_clip_on_zero_are_no_ops():
    d = simulate(RPM, A, G, vmax=9.0, store=True, nrev=20)
    k = simulate(RPM, A, G, vmax=9.0, store=True, nrev=20, kick_step=3000, kick_dv=0.0, clip_on=0)
    assert np.array_equal(k["x"], d["x"])


def test_depth_ramp_force_is_linear():
    r = simulate(RPM, A, G, store=True, nrev=60, ramp_a=20)
    assert mean_force(r, 9, 10) == pytest.approx(0.475, abs=2e-3)
    assert mean_force(r, 40, 60) == pytest.approx(1.0, abs=1e-4)


def test_face_entry_follows_engaged_arc():
    """Engaged fraction of the static force after the tool centre advanced s:
    the slice force integrated over phi in [pi/2, pi - asin(1 - s/R)]."""
    R = 0.5 * P.D_TOOL
    ph = np.linspace(np.pi / 2, np.pi, 20001)
    w = (P.K_TC * np.sin(ph) - P.K_RC * np.cos(ph)) * np.sin(ph)
    r = simulate(RPM, A, G, store=True, nrev=80, edge=True)
    for r0 in (5, 10, 20, 30):
        s = (r0 + 0.5) * P.N_TEETH * P.FZ
        frac = np.trapezoid(w * (np.sin(ph) >= 1 - s / R), ph) / np.trapezoid(w, ph)
        assert mean_force(r, r0, r0 + 1) == pytest.approx(frac, abs=3e-3)
    assert mean_force(r, 60, 80) == pytest.approx(1.0, abs=1e-4)
    assert entry_revs(edge=True) == pytest.approx(R / (P.FZ * P.N_TEETH))


def test_feed_ramp_chip_is_advance_over_one_tooth():
    """Chip = s(t) - s(t - T).  At a negligible depth (no dynamics) the
    force under the ramp is the step-entry force waveform times the chip
    ratio, pointwise, after the first tooth period; during the ramp that
    ratio is (t - T/2) / T_f."""
    a, nrev, Tf = 1e-6, 12, 10
    st = simulate(RPM, a, 0.0, store=True, nrev=nrev)
    ra = simulate(RPM, a, 0.0, store=True, nrev=nrev, ramp_f=Tf, f0=0.0)
    trev = np.arange(len(st["fcut"])) / SPR
    nd = SPR // P.N_TEETH
    T = 1.0 / P.N_TEETH
    s = lambda u: np.where(u < 0, 0.0, np.where(u < Tf, u * u / (2 * Tf), u - Tf / 2))   # advance, rev units
    chip = (s(trev) - s(trev - T)) / T
    k = slice(nd, nrev * SPR)
    assert np.abs(ra["fcut"][k] - chip[k] * st["fcut"][k]).max() < 1e-5 * np.abs(st["fcut"]).max()
    r = simulate(RPM, A, G, store=True, nrev=60, ramp_f=Tf, f0=0.0)
    assert mean_force(r, 30, 60) == pytest.approx(1.0, abs=1e-4)


def test_entry_length_follows_the_kinematics():
    assert entry_revs(edge=True) == pytest.approx(40.0)
    assert entry_revs(edge=True, ramp_f=40) == pytest.approx(60.0)      # s(Tf) = R/2 at the end of the ramp
    assert entry_revs(edge=True, ramp_f=10, f0=0.2) == pytest.approx(44.0)
    assert entry_revs(edge=True, fz=0.5 * P.FZ) == pytest.approx(80.0)


def test_classify_window_starts_after_a_kick():
    """A kick late in the run must be inside the judged window."""
    from tdboundary import classify
    c = classify(2500, 15e-3, 1500.0, vmax=9.0, nrev=300, edge=True, kick_step=310 * SPR, kick_dv=30e-3)
    assert not c["stable"]


def test_kick_sets_velocity_and_free_decay():
    r = simulate(RPM, 1e-9, 0.0, store=True, nrev=20, kick_step=SPR, kick_dv=1e-3)
    assert r["v"][SPR] == pytest.approx(1e-3, rel=1e-6)
    assert np.abs(r["x"]).max() == pytest.approx(1e-3 / P.WN, rel=0.03)


def test_clip_on_delays_the_limit():
    r = simulate(2500, 10e-3, 3000.0, vmax=9.0, store=True, nrev=20, clip_on=5 * SPR)
    assert np.abs(r["vcmd"][:5 * SPR]).max() > 9.0
    assert np.abs(r["vcmd"][5 * SPR:-1]).max() <= 9.0 + 1e-12


def test_df_gain_limits():
    from c1_unsafe_zone import df_gain, critical_amplitude
    V = 9.0
    assert df_gain([0.5 * V], np.zeros(1), V)[0] == pytest.approx(1.0, abs=1e-12)
    rho = np.array([2.0, 5.0, 20.0])
    exact = (2 / np.pi) * (np.arcsin(1 / rho) + (1 / rho) * np.sqrt(1 - 1 / rho ** 2))
    assert np.allclose(df_gain(rho * V, np.zeros(1), V), exact, rtol=2e-4)
    # a forced command that already uses the authority lowers the threshold
    b = 6.0 * np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    assert critical_amplitude(0.8, b, V) < critical_amplitude(0.8, np.zeros(1), V)


def test_effective_gain_needed():
    from c1_unsafe_zone import effective_gain_needed
    gains = np.linspace(0, 1500, 4)
    al = np.array([1.0, 6.0, 12.0, 17.0])
    assert effective_gain_needed(al, gains, 14.5) == pytest.approx(1250.0)
    assert effective_gain_needed(al, gains, 0.5) == 0.0
    assert effective_gain_needed(al, gains, 18.0) == 1500.0


@pytest.mark.slow
def test_bistable_cut_at_2500_rpm():
    """15 mm at 2500 rpm, g = 1500 V s/m: below a_lim (17.2 mm) and stable
    without the limit; with the 9 V limit the step entry chatters, the face
    entry settles, and a 30 mm/s kick on the settled cut makes it chatter
    while the same kick on the unclipped cut decays."""
    from tdboundary import classify, classify_run
    a, g = 15e-3, 1500.0
    assert classify(2500, a, g, vmax=0.0)["stable"]
    assert not classify(2500, a, g, vmax=9.0)["stable"]
    assert classify(2500, a, g, vmax=9.0, edge=True)["stable"]
    ks, nd = 100 * SPR + 37, SPR // P.N_TEETH
    for vmax, chatter in ((9.0, True), (0.0, False)):
        r = simulate(2500, a, g, vmax=vmax, store=True, nrev=400, edge=True, kick_step=ks, kick_dv=30e-3)
        st = (ks // nd + 1) * nd
        assert (not classify_run(r["x"][st:], SPR, r["diverged"])["stable"]) == chatter
