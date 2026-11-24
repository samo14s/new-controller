"""Simulated Monte Carlo of the basin: worst-phase critical kick of the clipped cut
at 0.9 of the NOMINAL a_lim under parameter uncertainty.

NOT A ROBUSTNESS GUARANTEE.  A few tens of random samples per speed show how
the simulated basin moves when the parameters are off; they bound nothing (with
0 failures in 25 samples the 95 % upper confidence bound on a failure rate is
still 11 %), and the kick is sampled at 4 phases of one tooth period only.

Purpose.  mc_closed_form.py samples the free voltage of the forced orbit; this
script asks the time-domain question for a subset of the same samples: how large
a velocity kick the settled cut survives, compared with the nominal parameters.

Protocol (C1 kick protocol of src/c1_unsafe_zone.py, extended):
  samples   the first n (default 25) samples of the +-10 % closed-form draw
            (mc_common.draw(n, 0.1), the same seed as mc_closed_form.py): K_tc,
            K_rc +-10 %, f_n +-5 % (K fixed), damping +-20 %, K_a +-10 %, runout
            U(0, 5 um) with a runout phase psi ~ U(0, 2 pi), clip U(8.1, 9 V), loop
            delay U(0, 300 us); the same samples at the four speeds.
  cut       1090, 1420, 2500, 2870 rpm; g = 1500 V s/m; depth 0.9 of the NOMINAL
            a_lim (zoa, nominal parameters); clipped command |V| <= Vclip of the
            sample; face entry (edge); full surface memory (nreg = 0); eccentric
            runout and loop delay of the sample (tdsim COMB build).
  kick      as c1_unsafe_zone._critical_kick: the cut settles for KICK_REV = 100
            revolutions, x' is kicked at phase p of N_PHASE = 16 of the tooth period,
            the WINDOW = 300 revolutions after the kick are judged; the kicks of
            DV_GRID (1e-4 ... 3 m/s) are tried upwards and the first interval that
            ends in chatter is bisected N_BISECT = 6 times in log(dv); nan if the
            1e-4 m/s kick already ends in chatter (no settled cut), inf if no kick
            of the grid does.  Worst (smallest) over the phases 0, 4, 8, 12.
  judge     the C1 classifier (tdboundary.classify_run) samples x once per tooth
            period: with runout the settled orbit is spindle-periodic and that
            classifier calls it chatter (checked: mu = q = 1).  The runs are
            therefore judged once per revolution (classify_run with N = 1, same
            thresholds, same block length in time).  Both verdicts are recorded
            for every run.  The nominal reference is computed with both
            classifiers; with the tooth classifier it must reproduce
            results/c1_basin.npz exactly (same simulator build, same runs).
  delay     the simulator needs a loop delay of at least one step (60/(512 rpm):
            107, 83, 47, 41 us); a sampled delay below one step is simulated as 0
            (below half a step) or as one step.  The delay simulated is stored, and
            the closed-form values of the sample are computed with it.
  worst phase  phases in the order (nominal worst phase first, then the others);
            a phase stops as soon as its critical kick is proven larger than the
            smallest so far (a grid kick at or above it settled, or the bisection's
            lower end reached it).  The minimum is the same as with four full
            searches; the other phases then carry a lower bound ("lower").
  closed form  for each (sample, speed): a_lim, linear stability at the depth,
            bmax (worst of 8 runout phases, and at the sample's psi), u0 =
            Vclip - bmax (mc_common.Model, as mc_closed_form.py).

Run (one worker; about 70 s per sample and speed at full speed, so about 2 h for
25 samples at four speeds, plus about 6 min for the nominal reference):
    cd ozsoy2026 && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
        python3 verification/paper/mc_kick.py [--n 25] [--resume]
Writes verification/paper/mc_kick.json after every (sample, speed), so a stop
loses at most one unit; --resume skips the units already in that file.
Arrays for figures (no simulation, from the json):
    python3 verification/paper/mc_kick.py --export      -> mc_kick.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_common as MC                       # noqa: E402
from mc_common import P, zoa                 # noqa: E402
import c1_unsafe_zone as C                   # noqa: E402
from tdsim import simulate, SOURCE_SHA       # noqa: E402
from tdboundary import classify_run          # noqa: E402

OUT = os.path.join(HERE, "mc_kick.json")
FRAC = 0.9
PHASES = (0, 4, 8, 12)                       # of C1's 16 phases per tooth period
SPR, ND = C.SPR, C.ND
LEVEL = 0.1


# ------------------------------------------------------------ one run ----
def make_runner(rpm, a, vclip, runout=0.0, psi=0.0, tau_sim=0.0, g=MC.G_DESIGN):
    """run(phase, dv) -> verdicts of one kicked run (memoised per phase and dv)."""
    memo = {}
    stats = dict(runs=0, sim_s=0.0)

    def run(phase, dv):
        key = (phase, float(dv))
        if key in memo:
            return memo[key]
        ks = C.KICK_REV * SPR + (phase * ND) // C.N_PHASE
        kw = dict(edge=True, nreg=C.NREG, kick_step=ks, kick_dv=float(dv))
        if runout:
            kw.update(runout=runout, runout_phase=psi)
        if tau_sim:
            kw.update(loop_delay=tau_sim)
        t0 = time.time()
        r = simulate(rpm, a, g, vmax=vclip, store=True, nrev=C.KICK_REV + C.WINDOW, **kw)
        stats["runs"] += 1
        stats["sim_s"] += time.time() - t0
        st = (ks // ND + 1) * ND
        x = r["x"][st:]
        cr = classify_run(x, SPR, r["diverged"], N=1)
        ct = classify_run(x, SPR, r["diverged"])
        pre = slice(ks - 10 * SPR, ks)
        vpre = np.abs(r["vcmd"][pre])
        out = dict(dv=float(dv), chat_rev=not cr["stable"], chat_tooth=not ct["stable"],
                   diverged=bool(r["diverged"]), max_x_mm=float(r["max_x"] * 1e3),
                   pre_kick_peak_V=float(vpre.max()) if vpre.size else None,
                   pre_kick_clipped=bool(vpre.size and vpre.max() >= vclip * (1 - 1e-12)),
                   pre_kick_rev_residual_fz=float(np.abs(r["x"][ks - SPR:ks] - r["x"][ks - 2 * SPR:ks - SPR]).max() / P.FZ))
        memo[key] = out
        return out
    return run, memo, stats


def critical_kick(run, phase, judge, best=np.inf):
    """C1 critical kick at one phase with the given classifier ('rev' or 'tooth').
    Returns (value, kind): kind 'exact', 'inf', 'unsettled' (value nan) or 'lower'
    (the search stopped once the critical kick was proven > best; value is the bound)."""
    key = "chat_rev" if judge == "rev" else "chat_tooth"

    def ch(dv):
        return run(phase, dv)[key]
    if ch(C.DV_LO):
        return np.nan, "unsettled"
    for k in range(1, len(C.DV_GRID)):
        if ch(C.DV_GRID[k]):
            break
        if C.DV_GRID[k] >= best:
            return float(C.DV_GRID[k]), "lower"
    else:
        return np.inf, "inf"
    lo, hi = np.log(C.DV_GRID[k - 1]), np.log(C.DV_GRID[k])
    for _ in range(C.N_BISECT):
        if np.exp(lo) >= best:
            return float(np.exp(lo)), "lower"
        mid = 0.5 * (lo + hi)
        if ch(np.exp(mid)):
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi)), "exact"


def worst_phase(run, judge, order, shortcut=True):
    best, per, worst = np.inf, {}, None
    for ph in order:
        v, kind = critical_kick(run, ph, judge, best if shortcut else np.inf)
        per[str(ph)] = dict(value_mmps=None if not np.isfinite(v) else v * 1e3, kind=kind)
        if kind == "unsettled":
            worst = np.nan
            break
        if kind == "exact" and v < best:
            best = v
    if worst is None:
        worst = best
    return worst, per


def tau_for_sim(tau, rpm):
    dt = 60.0 / (rpm * SPR)
    if tau < 0.5 * dt:
        return 0.0
    return float(max(tau, dt))


# ------------------------------------------------------------- summary ----
def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound of a binomial proportion."""
    return 1.0 if k >= n else float(beta_dist.ppf(conf, k + 1, n - k))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5:
        return None
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarise(res):
    out = {}
    for rpm in MC.SPEEDS:
        k = str(int(rpm))
        nom = res["nominal"].get(k)
        units = [u for u in res["samples"] if u["rpm"] == rpm]
        if not units or nom is None:
            continue
        ref = nom["worst4_rev_mmps"]
        # no settled cut = no basin: counted as a critical kick of 0 in the order statistics
        v = np.array([0.0 if u["worst4_kind"] == "unsettled" else
                      (np.inf if u["worst4_mmps"] is None else u["worst4_mmps"]) for u in units])
        n = len(v)
        fin = v[np.isfinite(v)]
        uns = int(sum(u["worst4_kind"] == "unsettled" for u in units))
        below = int(np.sum(v < ref))
        half = int(np.sum(v < 0.5 * ref))
        u0w = [u["cf"]["u0_worst_psi_V"] for u in units]
        u0s = [u["cf"]["u0_sample_psi_V"] for u in units]
        marg = [u["cf"]["a_lim_mm"] / u["a_mm"] - 1 for u in units]
        out[k] = dict(
            n=n, nominal_worst4_rev_mmps=ref, nominal_worst4_tooth_mmps=nom["worst4_tooth_mmps"],
            median_mmps=float(np.median(v)), min_mmps=float(v.min()), max_mmps=float(v.max()) if np.isfinite(v.max()) else None,
            p10_mmps=float(np.quantile(v, 0.1, method="inverted_cdf")),
            p90_mmps=(lambda z: None if not np.isfinite(z) else z)(float(np.quantile(v, 0.9, method="inverted_cdf"))),
            ratio_to_nominal_median=float(np.median(v) / ref), ratio_to_nominal_min=float(v.min() / ref),
            n_unsettled=uns, n_inf=int(np.sum(np.isinf(v))), n_below_nominal=below, n_below_half_nominal=half,
            frac_below_nominal=below / n, cp95_upper_unsettled=cp_upper(uns, n),
            cp95_upper_below_half_nominal=cp_upper(half, n),
            n_closed_form_unstable=int(sum(not u["cf"]["stable"] for u in units)),
            n_closed_form_saturated=int(sum(u["cf"]["stable"] and u["cf"]["u0_worst_psi_V"] <= 0 for u in units)),
            n_pre_kick_clipped=int(sum(bool(u.get("pre_kick_clipped")) for u in units)),
            spearman_kick_vs_u0_worst_psi=spearman(v, [np.nan if z is None else z for z in u0w]),
            spearman_kick_vs_u0_sample_psi=spearman(v, [np.nan if z is None else z for z in u0s]),
            spearman_kick_vs_alim_margin=spearman(v, marg),
            finite_median_mmps=float(np.median(fin)) if fin.size else None)
    return out


def dump(res):
    res["summary"] = summarise(res)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(clean(res), fh, indent=1)
    os.replace(tmp, OUT)


# --------------------------------------------------------------- units ----
def closed_form(p, rpm, a, tau_sim):
    m = MC.Model(dict(p, tau=tau_sim))
    c = m.crossings(rpm)
    a0 = m.zeros(rpm, 0.0) == 0
    al = (float(c[0]) if c.size else np.inf) if a0 else 0.0
    ncr = int(np.sum(c < a))
    stable = bool(a0 and (ncr == 0 or (ncr >= 2 and m.zeros(rpm, a) == 0)))
    bw = float(m.bmax(rpm, [a])[0])
    bs = float(m.orbit_peaks(rpm, [a], psis=[p["psi"]])[0, 0]) if p["runout"] else bw
    mt = MC.Model(p)                         # the sampled (unrounded) delay
    ct = mt.crossings(rpm)
    return dict(a_lim_mm=al * 1e3, stable=stable, bmax_worst_psi_V=bw, bmax_sample_psi_V=bs,
                u0_worst_psi_V=p["vclip"] - bw if stable else None,
                u0_sample_psi_V=p["vclip"] - bs if stable else None,
                a_lim_sampled_tau_mm=(float(ct[0]) if ct.size else np.inf) * 1e3,
                bmax_worst_psi_sampled_tau_V=float(mt.bmax(rpm, [a])[0]))


def nominal_unit(rpm, a, stored):
    """Both classifiers, all four phases searched in full (no shortcut)."""
    run, memo, stats = make_runner(rpm, a, MC.NOMINAL["vclip"])
    t0 = time.time()
    order = PHASES
    w_t, per_t = worst_phase(run, "tooth", order, shortcut=False)
    w_r, per_r = worst_phase(run, "rev", order, shortcut=False)
    runs = list(memo.values())
    agree = sum(r["chat_rev"] == r["chat_tooth"] for r in runs)
    st = {str(ph): float(stored[ph] * 1e3) for ph in PHASES}
    return dict(rpm=rpm, a_mm=a * 1e3, worst4_tooth_mmps=w_t * 1e3, worst4_rev_mmps=w_r * 1e3,
                phases_tooth=per_t, phases_rev=per_r, stored_c1_basin_mmps=st,
                stored_worst4_mmps=float(min(stored[ph] for ph in PHASES) * 1e3),
                stored_worst16_mmps=float(np.nanmin(stored) * 1e3),
                reproduces_stored=bool(all(abs(per_t[str(ph)]["value_mmps"] - st[str(ph)]) < 1e-9 for ph in PHASES)),
                runs=stats["runs"], verdicts_agree=agree, verdict_pairs=len(runs),
                disagreements=[dict(phase=k[0], dv=r["dv"], rev=r["chat_rev"], tooth=r["chat_tooth"])
                               for k, r in memo.items() if r["chat_rev"] != r["chat_tooth"]],
                pre_kick_peak_V=memo[(PHASES[0], float(C.DV_LO))]["pre_kick_peak_V"],
                runtime_s=time.time() - t0, sim_s=stats["sim_s"])


def sample_unit(i, p, rpm, a, order):
    tau_sim = tau_for_sim(p["tau"], rpm)
    cf = closed_form(p, rpm, a, tau_sim)
    run, memo, stats = make_runner(rpm, a, p["vclip"], p["runout"], p["psi"], tau_sim)
    t0 = time.time()
    w, per = worst_phase(run, "rev", order, shortcut=True)
    runs = list(memo.values())
    lo = memo.get((order[0], float(C.DV_LO)))
    return dict(i=i, rpm=rpm, a_mm=a * 1e3, params=p, tau_sim=tau_sim, cf=cf,
                worst4_mmps=None if not np.isfinite(w) else w * 1e3,
                worst4_kind=("unsettled" if np.isnan(w) else ("inf" if np.isinf(w) else "exact")),
                phases=per, runs=stats["runs"],
                rev_vs_tooth_disagree=sum(r["chat_rev"] != r["chat_tooth"] for r in runs),
                pre_kick_peak_V=lo["pre_kick_peak_V"] if lo else None,
                pre_kick_clipped=lo["pre_kick_clipped"] if lo else None,
                pre_kick_rev_residual_fz=lo["pre_kick_rev_residual_fz"] if lo else None,
                runtime_s=time.time() - t0, sim_s=stats["sim_s"])


def export_npz(path=OUT):
    """Arrays for figures from mc_kick.json -> mc_kick.npz (no simulation).
    kick_mmps (speed, sample): worst-of-4-phase critical kick, 0 where the cut
    did not settle (no basin), inf where no kick up to 3 m/s ended in chatter;
    u0_* nan where the closed form calls the cut linearly unstable."""
    res = json.load(open(path))
    sp = list(MC.SPEEDS)
    n = 1 + max(u["i"] for u in res["samples"])
    f = lambda: np.full((len(sp), n), np.nan)
    kick, u0w, u0s, al, bw, bs, pre, ts = f(), f(), f(), f(), f(), f(), f(), f()
    kind = np.full((len(sp), n), "", dtype="<U9")
    par = np.full((n, len(MC.NAMES) + 1), np.nan)
    for u in res["samples"]:
        s, i = sp.index(u["rpm"]), u["i"]
        kind[s, i] = u["worst4_kind"]
        kick[s, i] = 0.0 if u["worst4_kind"] == "unsettled" else (np.inf if u["worst4_mmps"] is None else u["worst4_mmps"])
        c = u["cf"]
        u0w[s, i] = np.nan if c["u0_worst_psi_V"] is None else c["u0_worst_psi_V"]
        u0s[s, i] = np.nan if c["u0_sample_psi_V"] is None else c["u0_sample_psi_V"]
        al[s, i], bw[s, i], bs[s, i] = c["a_lim_mm"], c["bmax_worst_psi_V"], c["bmax_sample_psi_V"]
        pre[s, i] = np.nan if u["pre_kick_peak_V"] is None else u["pre_kick_peak_V"]
        ts[s, i] = u["tau_sim"]
        par[i] = [u["params"][k] for k in MC.NAMES] + [u["params"]["psi"]]
    nom = np.array([res["nominal"][str(int(r))]["worst4_rev_mmps"] for r in sp])
    a = np.array([res["nominal"][str(int(r))]["a_mm"] for r in sp])
    np.savez_compressed(path.replace(".json", ".npz"), speeds=np.array(sp), names=np.array(list(MC.NAMES) + ["psi"]),
                        params=par, kick_mmps=kick, kind=kind, nominal_kick_mmps=nom, depth_mm=a,
                        u0_worst_psi_V=u0w, u0_sample_psi_V=u0s, a_lim_mm=al, bmax_worst_psi_V=bw,
                        bmax_sample_psi_V=bs, pre_kick_peak_V=pre, tau_sim_s=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="samples per speed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-nominal", action="store_true", help="test only: no nominal reference")
    ap.add_argument("--export", action="store_true", help="only write mc_kick.npz from mc_kick.json")
    a = ap.parse_args()
    if a.export:
        export_npz()
        return
    t_start = time.time()
    MC.restore_nominal()
    al_nom = {rpm: zoa.a_lim(rpm, MC.G_DESIGN) for rpm in MC.SPEEDS}
    basin = np.load(os.path.join(MC.RES, "c1_basin.npz"))
    if a.resume and os.path.exists(OUT):
        res = json.load(open(OUT))
    else:
        res = dict(note=__doc__.split("\n\n")[1].replace("\n", " "), protocol=dict(
            frac=FRAC, phases=PHASES, n_phase=C.N_PHASE, kick_rev=C.KICK_REV, window=C.WINDOW,
            dv_grid=C.DV_GRID.tolist(), n_bisect=C.N_BISECT, nreg=C.NREG, entry="face (edge)", g=MC.G_DESIGN,
            level=LEVEL, seed=MC.SEED, judge="classify_run once per revolution (N = 1)",
            zeta_nominal=MC.ZETA0, build=SOURCE_SHA[:12], c1_basin_build=str(basin["build"])),
            a_lim_nominal_mm={str(int(r)): v * 1e3 for r, v in al_nom.items()},
            nominal={}, samples=[], runtime_s=0.0)
    t_prev = float(res.get("runtime_s", 0.0))
    # nominal reference
    if not a.skip_nominal:
        for rpm in MC.SPEEDS:
            k = str(int(rpm))
            if k in res["nominal"]:
                continue
            MC.restore_nominal()
            sel = np.nonzero((basin["rpm"] == rpm) & (np.abs(basin["frac"] - FRAC) < 1e-9) & (basin["vmax"] > 0))[0][0]
            res["nominal"][k] = nominal_unit(rpm, FRAC * al_nom[rpm], basin["dv"][sel])
            res["runtime_s"] = t_prev + time.time() - t_start
            dump(res)
            nu = res["nominal"][k]
            print(f"nominal {k}: tooth {nu['worst4_tooth_mmps']:.3f} rev {nu['worst4_rev_mmps']:.3f} stored "
                  f"{nu['stored_worst4_mmps']:.3f} mm/s, reproduces {nu['reproduces_stored']}, verdicts agree "
                  f"{nu['verdicts_agree']}/{nu['verdict_pairs']}, {nu['runtime_s']:.0f} s", flush=True)
    # samples, cycling through the speeds
    S = MC.draw(a.n, LEVEL)
    done = {(u["i"], u["rpm"]) for u in res["samples"]}
    for i in range(a.n):
        p = MC.sample(S, i)
        for rpm in MC.SPEEDS:
            if (i, rpm) in done:
                continue
            k = str(int(rpm))
            nom = res["nominal"].get(k)
            first = min(PHASES, key=lambda ph: (nom["phases_rev"][str(ph)]["value_mmps"] or np.inf)) if nom else PHASES[0]
            order = (first,) + tuple(ph for ph in PHASES if ph != first)
            MC.apply_params(p)                # every src module, the simulator reads params at call time
            u = sample_unit(i, p, rpm, FRAC * al_nom[rpm], order)
            MC.restore_nominal()
            res["samples"].append(u)
            res["runtime_s"] = t_prev + time.time() - t_start
            dump(res)
            print(f"sample {i} {k} rpm: worst4 {u['worst4_mmps']} mm/s ({u['worst4_kind']}), u0 "
                  f"{u['cf']['u0_worst_psi_V']}, a_lim {u['cf']['a_lim_mm']:.2f} mm, runs {u['runs']}, "
                  f"{u['runtime_s']:.0f} s", flush=True)
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
