"""Monte Carlo of the failure probability of the basin-aware design against a reference design under
the same sampled model errors (review round 2: the simulated Monte Carlo of Section 7.2, mc_kick.py,
was run on the g = 1500 V s/m design only).

NOT A ROBUSTNESS GUARANTEE.  N random samples per speed estimate a failure probability with a binomial
error (0 failures in 100 samples still allow 3.6 % at 95 % confidence); each design is kicked with one
kick size at 4 phases of one tooth period after one face entry; nothing is certified for the perturbed
models, and the samples cover only the ranges listed below.

Purpose.  Is the basin-aware design (basin_aware.py) more robust to model errors than a reference design
with the same gain range, voltage margin and depth rule?  Per speed and required kick b, both designs
are simulated on the same parameter samples (paired), each at its own operating depth and both at a
common depth, and their failure probabilities are compared.

Protocol.
  speeds    the six validation speeds of the basin-aware chart, 750, 1125, 1375, 2000, 2500, 2875 rpm
            (--speeds overrides; any speed of the 125 rpm grid of both design files works).
  designs   ref    'auth10' of fair_designs.json: the authority condition, Eq. (14), with a 10 % voltage
                   margin and g <= 3000 V s/m, operated at its checked depth a_ref = a_chk (the same
                   design for every b);
            BA(b)  basin_aware.json per_speed[n].designs[b]: gain g*(b) (g_rob) and depth a_rob(b),
                   b = 1, 2, 5 mm/s.
  compare   per speed and b:  own     ref at (g_ref, a_ref)  vs  BA(b) at (g*(b), a_rob(b));
                              common  ref at (g_ref, a_c)    vs  BA(b) at (g*(b), a_c),
                                      a_c = min(a_ref, a_rob(b)), each design with its own gain.
  samples   the first N (default 100, --n) samples of mc_common.draw(N, 0.1) with mc_common.SEED: the
            same samples, in the same order, as mc_closed_form.py (pm10) and mc_kick.py (both checked,
            against mc_closed_form.npz and mc_kick.json): K_tc, K_rc +-10 %, f_n +-5 % (K fixed),
            damping +-20 %, K_a +-10 % (set by mc_common.apply_params in every src module, restored by
            mc_common.restore_nominal after every unit; every worker starts from the nominal damping
            1.34 % of results/gainmap_main.npz), runout U(0, 5 um) with its phase psi, clip
            U(8.1, 9 V), loop delay U(0, 300 us) simulated as mc_kick.tau_for_sim (0 below half a
            step, otherwise at least one step).  The same samples at every speed, design and depth.
  one run   mc_kick.make_runner(rpm, a, vclip, runout, psi, tau_sim, g), the run of mc_kick.py with
            the design gain: face entry (edge), full surface memory (nreg = 0), |V| <= vclip, x'
            kicked by dv at phase p of N_PHASE = 16 of the tooth period in revolution KICK_REV = 100,
            the WINDOW = 300 revolutions after the kick judged once per revolution
            (tdboundary.classify_run with N = 1: with runout the settled orbit is spindle-periodic);
            the per-tooth verdict is recorded as well.
  verdicts  per (speed, g, a, sample):
            unsettled   as mc_kick.py defines it for a sample: mc_kick.critical_kick returns
                        'unsettled' when the smallest kick of the C1 grid, DV_LO = 1e-4 m/s, already
                        ends in chatter, and mc_kick.worst_phase calls the sample unsettled when this
                        happens at any of the phases 0, 4, 8, 12 (the cut chatters before the kick or
                        at a negligible kick).  The DV_LO kick is run at the phases 0, 4, 8, 12 in this
                        order, stopping at the first that ends in chatter;
            kick(b)     if settled: kicks of exactly b at the phases 0, 4, 8, 12 in this order,
                        stopping at the first that ends in chatter;
            failure(b)  unsettled, or a kick of b ends in chatter (per-revolution verdict).
            Breakdown of the failures: 'unsettled', and 'kick' (settled, but a kick of b chatters).
  nominal   the same per (speed, g, a, b) with the nominal parameters (clip 9 V, no runout, no delay:
            the plain simulator build of the validation of basin_aware.py), judged per revolution and
            per tooth period; the per-tooth kick verdicts at the basin-aware design points are compared
            with the stored validation of basin_aware.py (basin_aware_cache/050bc4a99e/validate.jsonl:
            kicks of b at 16 phases, critical kicks at 4 phases).
  closed form  diagnostic only, not part of the verdict: linear stability of the sample's averaged
            model at (g, a) with the simulated delay, and its linear limit (the test of
            mc_kick.closed_form, with mc_common.Model at the design gain).
  statistics  per speed, b and comparison: n paired samples; per design the failures, P_fail, the
            Clopper-Pearson 95 % interval (two-sided), the breakdown, the closed-form-unstable counts
            and the nominal outcome; paired counts (both fail, ref fails & BA passes, BA fails & ref
            passes, both pass) with the exact two-sided McNemar p-value (binomial test of the
            discordant pairs at p = 1/2).  Summary pooled over the speeds, per b and comparison: the
            same counts (the pooled interval and McNemar p treat every (speed, sample) pair as
            independent, which the samples shared by the speeds do not make true) and a sample-level
            exact sign test (per sample: the number of speeds at which only the reference fails minus
            the number at which only BA fails), which is valid under that dependence.
  cache     every run is appended, as soon as it finishes, to <out>_cache.jsonl, keyed by (rpm, g,
            a [mm], sample [-1 = nominal], phase, dv) and tagged with a protocol hash (the memo of
            mc_kick.make_runner, persisted); design-depth points shared by b values and comparisons
            are run once, and a restart skips every run (and closed-form record) already there.

Run (from the repository root ozsoy2026/).  N = 100 at the six speeds: 34 design-depth points, 3434
units (100 samples and the nominal), at most 35 552 simulator runs (every cut settles and no kick of b
chatters); about 30 800 expected (an unsettled cut stops after one run; the closed form calls 99 of
680 sample-point pairs, 15 %, linearly unstable over the first 20 samples).  1.32 CPU-s per run measured in
the smoke test (139 runs at 2500 rpm, another campaign running), so about 11-13 CPU-h, i.e. 5.5-6.5 h
with 2 workers:
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup python3 verification/paper/mc_designs.py \\
        --workers 2 > verification/paper/mc_designs.log 2>&1 &
  --plan      print the number of runs and the estimate, and stop;
  --assemble  write the JSON from the cache without simulating (partial results are flagged);
  --smoke     write mc_designs_smoke.json and mc_designs_smoke_cache.jsonl instead of the real files
              (smoke test: --smoke --speeds 2500 --n 2 --workers 1, about 3 CPU-min).
Writes verification/paper/mc_designs.json (also every --write-every seconds while running, with
'complete' false until every unit is done).
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")            # one thread per process (set before numpy loads BLAS)
import numpy as np                                       # noqa: E402
from scipy.stats import beta as beta_dist, binom          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, "results")

import params as P                                       # noqa: E402

ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])
P.set_damping(ZETA)                                      # before any module that reads the damping is imported

import mc_common as MC                                   # noqa: E402
import mc_kick as MK                                     # noqa: E402
import c1_unsafe_zone as C                               # noqa: E402
from tdsim import SOURCE_SHA                             # noqa: E402

SPEEDS = (750.0, 1125.0, 1375.0, 2000.0, 2500.0, 2875.0)   # basin_aware.VAL_SPEEDS
B_REQ = (1e-3, 2e-3, 5e-3)                               # m/s, basin_aware.BETA_REQ
PHASES = MK.PHASES                                       # (0, 4, 8, 12) of C.N_PHASE = 16
DV_LO = float(C.DV_LO)                                   # 1e-4 m/s: the smallest kick of the C1 grid
LEVEL = MK.LEVEL                                         # 0.1: the +-10 % draw of mc_kick.py
REF = "auth10"
N_DEFAULT = 100
BA_JSON = os.path.join(HERE, "basin_aware.json")
FD_JSON = os.path.join(HERE, "fair_designs.json")
VALIDATE = os.path.join(HERE, "basin_aware_cache", "050bc4a99e", "validate.jsonl")
PROTO = hashlib.sha1(repr((ZETA, LEVEL, MC.SEED, MC.NAMES, sorted(MC.ranges(LEVEL).items()),
                           sorted(MC.NOMINAL.items()), PHASES, DV_LO, C.KICK_REV, C.WINDOW, C.N_PHASE,
                           C.NREG, C.SPR, "edge", "mc_kick.make_runner", "tau_for_sim", SOURCE_SHA[:12],
                           "v1")).encode()).hexdigest()[:10]
NOMINAL_P = dict(MC.NOMINAL, psi=0.0)                    # clip 9 V, no runout, no delay
NOTE = ("NOT A ROBUSTNESS GUARANTEE. Simulated Monte Carlo (mc_designs.py) of the failure probability of the "
        "basin-aware design BA(b) (basin_aware.json) and of the reference design auth10 (fair_designs.json: Eq. 14 "
        "with a 10 % voltage margin, g <= 3000 V s/m, at its checked depth a_chk) under the same N samples of the "
        "+-10 % model errors of mc_kick.py (cutting coefficients, modal frequency and damping, actuator constant, "
        "runout, clip level, loop delay). Failure(b): the cut does not settle (the 1e-4 m/s kick of mc_kick.py "
        "already ends in chatter at one of the phases 0, 4, 8, 12 of 16) or a kick of exactly b at one of these "
        "phases ends in chatter (judged once per revolution over the 300 revolutions after the kick). 'own': each "
        "design at its own depth; 'common': both at min(a_ref, a_rob(b)), each with its own gain. With N samples the "
        "binomial error is large (0 failures in 100 samples still allow 3.6 % at 95 %), one kick size is tried at 4 "
        "phases of one tooth period after one face entry, and the same samples serve every speed, so the pooled "
        "interval and McNemar p are approximate (the sample-level sign test is not affected by that dependence).")


def lab(b):
    return f"{b * 1e3:.0f}"


# ------------------------------------------------------------------ designs ----
def load_designs(speeds):
    """Per speed: the reference design (auth10) and BA(b) for every b."""
    ba = json.load(open(BA_JSON))
    fd = json.load(open(FD_JSON))
    assert np.allclose(ba["protocol"]["beta_req_mmps"], [b * 1e3 for b in B_REQ]), "b values differ from basin_aware"
    D = {}
    for n in speeds:
        k = f"{n:.0f}"
        r = fd["speeds"][k][REF]
        ps = ba["per_speed"][k]
        ref = dict(design=REF, g=float(r["g"]), a_mm=float(r["a_chk"]), a_lim_mm=float(r["a_lim"]),
                   a_td_mm=float(r["a_td"]), kappa_a_lim_mm=float(r["kappa_a_lim"]),
                   a_over_a_lim=float(r["a_chk"]) / float(r["a_lim"]), source=f"fair_designs.json speeds[{k}].{REF}")
        bad = {}
        for b in B_REQ:
            z = ps["designs"][lab(b)]
            g = z.get("g")
            a = float(z.get("a_rob_mm") or 0.0)
            al = z.get("a_lim_at_g_mm")
            bad[lab(b)] = dict(g=None if g is None else float(g), a_mm=a, a_lim_mm=al,
                               a_over_a_lim=(a / al) if (g is not None and al) else None, binding=z.get("binding"),
                               beta_cert_mmps=z.get("beta_default_N_at_a_rob_mmps"),
                               source=f"basin_aware.json per_speed[{k}].designs[{lab(b)}]")
        D[float(n)] = dict(ref=ref, ba=bad)
    return D


def plan(D):
    """Comparisons and the distinct design-depth points (rpm, g, a_mm) with the b values each needs."""
    comps, points = [], {}

    def add(pt, b, role):
        e = points.setdefault(pt, dict(bs=set(), roles=[]))
        e["bs"].add(b)
        e["roles"].append(role)
    for n, d in D.items():
        gr, ar = d["ref"]["g"], d["ref"]["a_mm"]
        for b in B_REQ:
            z = d["ba"][lab(b)]
            if z["g"] is None or z["a_mm"] <= 0:
                continue                                   # no basin-aware design at this speed and b
            ac = min(ar, z["a_mm"])
            for kind, pr, pb in (("own", (n, gr, ar), (n, z["g"], z["a_mm"])), ("common", (n, gr, ac), (n, z["g"], ac))):
                comps.append(dict(rpm=n, b=b, kind=kind, ref=pr, ba=pb))
                add(pr, b, f"ref {kind} b={lab(b)}")
                add(pb, b, f"BA({lab(b)}) {kind}")
    for e in points.values():
        e["bs"] = tuple(sorted(e["bs"]))
    return comps, points


def sample_params(n):
    """The first n samples of mc_common.draw(n, 0.1) (mc_kick.py, mc_closed_form.py)."""
    S = MC.draw(max(n, 1), LEVEL)
    return [MC.sample(S, i) for i in range(n)]


# --------------------------------------------------------------- verdicts ----
def decide(run, bs, judge="rev"):
    """Verdicts of one (speed, g, a, sample) from the runs of run(phase, dv) (mc_kick.make_runner).
    unsettled: the DV_LO kick chatters at one of PHASES (mc_kick.critical_kick 'unsettled', any phase as in
    mc_kick.worst_phase); otherwise kick_phase[b]: the first phase at which a kick of exactly b chatters
    (None: none of the four)."""
    key = "chat_rev" if judge == "rev" else "chat_tooth"
    out = dict(unsettled=False, unsettled_phase=None, kick_phase={})
    for ph in PHASES:
        if run(ph, DV_LO)[key]:
            out.update(unsettled=True, unsettled_phase=ph)
            return out
    for b in bs:
        out["kick_phase"][lab(b)] = next((ph for ph in PHASES if run(ph, b)[key]), None)
    return out


def fail_of(v, b, judge="rev"):
    r = v[judge]
    return bool(r["unsettled"] or r["kick_phase"][lab(b)] is not None)


class Miss(Exception):
    """A run that the cache does not hold."""


def cached_runner(d):
    def run(ph, dv):
        rec = d.get((ph, float(dv)))
        if rec is None:
            raise Miss
        return rec["v"]
    return run


def resolve(unit, bs, idx):
    """Verdicts of a unit from the cache alone, or None if a run is missing."""
    d = idx.get(unit, {})
    try:
        rev = decide(cached_runner(d), bs, "rev")
        tooth = decide(cached_runner(d), bs, "tooth") if unit[3] < 0 else None
    except Miss:
        return None
    return dict(rev=rev, tooth=tooth)


def max_new_runs(unit, bs, idx):
    """Runs a unit can still need, given its cached runs: the decision path on which every missing run
    settles (the longest one: a chatter ends a loop)."""
    d = idx.get(unit, {})
    missing = set()

    def run(ph, dv):
        rec = d.get((ph, float(dv)))
        if rec is None:
            missing.add((ph, float(dv)))
            return dict(chat_rev=False, chat_tooth=False)
        return rec["v"]
    decide(run, bs, "rev")
    if unit[3] < 0:
        decide(run, bs, "tooth")
    return len(missing)


def closed_form(p, rpm, a, g, tau_sim):
    """Linear stability of the sample's averaged model at (g, a) with the simulated delay: the test of
    mc_kick.closed_form with the design gain (diagnostic)."""
    m = MC.Model(dict(p, tau=tau_sim), g=g)
    c = m.crossings(rpm)
    a0 = m.zeros(rpm, 0.0) == 0
    al = (float(c[0]) if c.size else np.inf) if a0 else 0.0
    ncr = int(np.sum(c < a))
    stable = bool(a0 and (ncr == 0 or (ncr >= 2 and m.zeros(rpm, a) == 0)))
    return dict(stable=stable, a_lim_mm=al * 1e3 if np.isfinite(al) else None, dvf_loop_stable=bool(a0),
                tau_sim=tau_sim)


# ------------------------------------------------------------------ worker ----
_NOM_STATE = None


def _init_worker():
    """Every worker starts from the nominal parameters and the nominal damping."""
    global _NOM_STATE
    P.set_damping(ZETA)
    MC.restore_nominal()
    _NOM_STATE = MC.cache_state()


def append(path, rec):
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def unit_job(args):
    """One (speed, g, a, sample): the runs the verdicts need (cached ones skipped), each appended to the
    cache as it finishes; the closed-form record if missing."""
    rpm, g, a_mm, i, bs, p, cached, need_cf, path = args
    t0, c0 = time.time(), time.process_time()
    MC.restore_nominal()
    assert P.ZETA == ZETA, "worker not at the nominal damping"
    new, cf = [], None
    try:
        MC.apply_params(p)                                 # every src module; the simulator reads them at call time
        tau_sim = MK.tau_for_sim(p["tau"], rpm)
        run, memo, stats = MK.make_runner(rpm, a_mm * 1e-3, p["vclip"], p["runout"], p["psi"], tau_sim, g=g)
        memo.update(cached)                                # the persisted memo: (phase, dv) -> verdicts

        def run_c(ph, dv):
            k = (ph, float(dv))
            if k not in memo:
                t, c = time.time(), time.process_time()
                v = run(ph, dv)
                rec = dict(kind="run", proto=PROTO, k=[rpm, g, a_mm, i, ph, float(dv)], v=v,
                           t=round(time.time() - t, 3), cpu=round(time.process_time() - c, 3))
                append(path, rec)
                new.append(rec)
            return memo[k]
        rev = decide(run_c, bs, "rev")
        tooth = decide(run_c, bs, "tooth") if i < 0 else None
        if need_cf:
            cf = dict(kind="cf", proto=PROTO, k=[rpm, g, a_mm, i], v=closed_form(p, rpm, a_mm * 1e-3, g, tau_sim))
            append(path, cf)
    finally:
        MC.restore_nominal()
    return dict(unit=(rpm, g, a_mm, i), rev=rev, tooth=tooth, new=new, cf=cf, wall=time.time() - t0,
                cpu=time.process_time() - c0, restored=bool(_NOM_STATE is None or MC.cache_state() == _NOM_STATE))


# ------------------------------------------------------------------- cache ----
def cache_path(out):
    return (out[:-5] if out.endswith(".json") else out) + "_cache.jsonl"


def load_cache(path):
    """Run records (unit -> (phase, dv) -> record) and closed-form records of this protocol."""
    idx, cfs, bad, other = {}, {}, 0, 0
    if not os.path.exists(path):
        return idx, cfs, bad, other
    with open(path, "rb") as fh:
        data = fh.read()
    if data and not data.endswith(b"\n"):                 # a line cut short: the next record starts on a new line
        with open(path, "a") as fh:
            fh.write("\n")
    for line in data.decode(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            k = rec["k"]
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
            continue
        if rec.get("proto") != PROTO:
            other += 1
            continue
        if rec.get("kind") == "run":
            idx.setdefault((k[0], k[1], k[2], k[3]), {})[(k[4], k[5])] = rec
        elif rec.get("kind") == "cf":
            cfs[tuple(k)] = rec["v"]
    return idx, cfs, bad, other


def per_run_estimate(idx):
    cpu = [r["cpu"] for d in idx.values() for r in d.values() if "cpu" in r]
    if len(cpu) >= 20:
        return float(np.mean(cpu)), f"mean CPU time of the {len(cpu)} cached runs"
    try:
        mk = json.load(open(os.path.join(HERE, "mc_kick.json")))
        runs = sum(u["runs"] for u in mk["samples"])
        sim = sum(u["sim_s"] for u in mk["samples"])
        return sim / runs, f"mean simulation time per run of mc_kick.json ({runs} runs)"
    except (OSError, KeyError, ZeroDivisionError, ValueError):
        return 1.5, "assumed"


# -------------------------------------------------------------- statistics ----
def cp95(k, n):
    """Clopper-Pearson two-sided 95 % interval of a binomial proportion."""
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def exact_two_sided(k1, k2):
    """Exact two-sided binomial test of k1 against k2 at p = 1/2 (McNemar on discordant pairs; sign test)."""
    m = k1 + k2
    return 1.0 if m == 0 else float(min(1.0, 2.0 * binom.cdf(min(k1, k2), m, 0.5)))


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


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(clean(obj), fh, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- assemble ----
def checks(D, samples, n, idx):
    """Same samples as mc_closed_form / mc_kick; nominal per-tooth verdicts against basin_aware's validation."""
    out = {}
    try:
        cf = np.load(os.path.join(HERE, "mc_closed_form.npz"))
        m = min(n, len(cf["pm10_psi"]))
        arr = np.array([[samples[i][k] for k in MC.NAMES] for i in range(m)]).reshape(m, len(MC.NAMES))
        psi = np.array([samples[i]["psi"] for i in range(m)])
        out["samples_equal_mc_closed_form_pm10"] = dict(
            compared=m, equal=bool(np.array_equal(arr, cf["pm10_params"][:m]) and np.array_equal(psi, cf["pm10_psi"][:m])))
    except (OSError, KeyError) as e:
        out["samples_equal_mc_closed_form_pm10"] = dict(error=str(e))
    try:
        mk = json.load(open(os.path.join(HERE, "mc_kick.json")))
        mk_p = {u["i"]: u["params"] for u in mk["samples"]}
        ids = sorted(i for i in mk_p if i < n)
        out["samples_equal_mc_kick"] = dict(compared=len(ids), equal=bool(all(
            all(samples[i][k] == mk_p[i][k] for k in list(MC.NAMES) + ["psi"]) for i in ids)))
    except (OSError, KeyError) as e:
        out["samples_equal_mc_kick"] = dict(error=str(e))
    stored = {}
    if os.path.exists(VALIDATE):
        for line in open(VALIDATE):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stored[json.dumps(rec["k"])] = rec["v"]["v"]
    rows = []
    for rpm, d in D.items():
        for b in B_REQ:
            z = d["ba"][lab(b)]
            if z["g"] is None or z["a_mm"] <= 0:
                continue
            runs = idx.get((rpm, z["g"], z["a_mm"], -1), {})
            a_m = z["a_mm"] * 1e-3
            for ph in PHASES:
                rk, rl = runs.get((ph, float(b))), runs.get((ph, DV_LO))
                sk = stored.get(json.dumps(["kick", rpm, a_m, z["g"], b, ph]))
                sc = stored.get(json.dumps(["crit", rpm, a_m, z["g"], ph]))
                rows.append(dict(rpm=rpm, b_mmps=b * 1e3, g=z["g"], a_mm=z["a_mm"], phase=ph,
                                 sim_kick_chatter_tooth=None if rk is None else rk["v"]["chat_tooth"],
                                 sim_kick_chatter_rev=None if rk is None else rk["v"]["chat_rev"],
                                 stored_kick_chatter=sk,
                                 sim_dv_lo_chatter_tooth=None if rl is None else rl["v"]["chat_tooth"],
                                 stored_crit_kick_mmps=None if sc is None else sc * 1e3))
    # stored critical kick: nan = the DV_LO kick already chatters (unsettled), inf = no kick up to 3 m/s chatters
    cmp_k = [r for r in rows if r["sim_kick_chatter_tooth"] is not None and r["stored_kick_chatter"] is not None]
    cmp_c = [r for r in rows if r["sim_kick_chatter_tooth"] is not None and r["stored_crit_kick_mmps"] is not None
             and not np.isnan(r["stored_crit_kick_mmps"])]
    cmp_l = [r for r in rows if r["sim_dv_lo_chatter_tooth"] is not None and r["stored_crit_kick_mmps"] is not None]
    out["nominal_vs_basin_aware_validation"] = dict(
        note="per-tooth verdicts of the nominal runs at the BA design points against basin_aware's validation "
             "(same simulator build and protocol): the kick of b at the phase (stored verdict), and the stored "
             "critical kick at the phase (a kick of b chatters iff b >= critical kick; the DV_LO kick chatters iff "
             "the stored critical kick is nan)",
        kicks_compared=len(cmp_k), kicks_agree=sum(r["sim_kick_chatter_tooth"] == r["stored_kick_chatter"] for r in cmp_k),
        kicks_agree_rev=sum(r["sim_kick_chatter_rev"] == r["stored_kick_chatter"] for r in cmp_k),
        crit_compared=len(cmp_c),
        crit_consistent=sum(r["sim_kick_chatter_tooth"] == bool(r["b_mmps"] >= r["stored_crit_kick_mmps"]) for r in cmp_c),
        dv_lo_compared=len(cmp_l),
        dv_lo_consistent=sum(r["sim_dv_lo_chatter_tooth"] == bool(np.isnan(r["stored_crit_kick_mmps"])) for r in cmp_l),
        rows=rows)
    nom = [r for (rpm, g, a, i), d in idx.items() if i < 0 for r in d.values()]
    out["nominal_rev_vs_tooth"] = dict(runs=len(nom), agree=sum(r["v"]["chat_rev"] == r["v"]["chat_tooth"] for r in nom))
    smp = [r for (rpm, g, a, i), d in idx.items() if i >= 0 for r in d.values()]
    out["samples_rev_vs_tooth"] = dict(runs=len(smp), agree=sum(r["v"]["chat_rev"] == r["v"]["chat_tooth"] for r in smp),
                                       note="information: with runout the per-tooth classifier calls the settled "
                                            "spindle-periodic orbit chatter (mc_kick.py); the verdict is per revolution")
    return out


def assemble(D, comps, points, samples, n, idx, cfs, meta):
    res = {}
    for pt, e in points.items():
        for i in [-1] + list(range(n)):
            v = resolve(pt + (i,), e["bs"], idx)
            if v is not None:
                res[pt + (i,)] = v

    def nominal(pt, b):
        v = res.get(pt + (-1,))
        if v is None:
            return None
        return dict(unsettled=v["rev"]["unsettled"], fail=fail_of(v, b),
                    kick_chatter_phase=None if v["rev"]["unsettled"] else v["rev"]["kick_phase"][lab(b)],
                    unsettled_tooth=v["tooth"]["unsettled"], fail_tooth=fail_of(v, b, "tooth"))

    def dstats(pt, b, ids):
        vs = [res[pt + (i,)] for i in ids]
        f = [fail_of(v, b) for v in vs]
        uns = [v["rev"]["unsettled"] for v in vs]
        cfu = [(cfs.get(pt + (i,)) is not None and not cfs[pt + (i,)]["stable"]) for i in ids]
        clip = [bool(idx[pt + (i,)][(PHASES[0], DV_LO)]["v"]["pre_kick_clipped"]) for i in ids]
        kph = {}
        for v in vs:
            ph = None if v["rev"]["unsettled"] else v["rev"]["kick_phase"][lab(b)]
            if ph is not None:
                kph[str(ph)] = kph.get(str(ph), 0) + 1
        k, m = int(sum(f)), len(vs)
        return dict(g=pt[1], a_mm=pt[2], n=m, failures=k, P_fail=(k / m) if m else None, cp95=cp95(k, m),
                    unsettled=int(sum(uns)), kick=k - int(sum(uns)), kick_first_chatter_phase=kph,
                    cf_linearly_unstable=int(sum(cfu)),
                    unsettled_and_cf_unstable=int(sum(u and c for u, c in zip(uns, cfu))),
                    settled_but_cf_unstable=int(sum((not u) and c for u, c in zip(uns, cfu))),
                    failed_but_cf_stable=int(sum(fl and not c for fl, c in zip(f, cfu))),
                    settled_pre_kick_clipped=int(sum((not u) and c for u, c in zip(uns, clip))),
                    nominal=nominal(pt, b))

    per_speed, pairs = {}, {}
    for rpm, d in D.items():
        per_speed[f"{rpm:.0f}"] = dict(ref=d["ref"], BA=d["ba"], b={})
    for c in comps:
        b, rpm = c["b"], c["rpm"]
        ids = [i for i in range(n) if c["ref"] + (i,) in res and c["ba"] + (i,) in res]
        rf = np.array([fail_of(res[c["ref"] + (i,)], b) for i in ids], bool)
        bf = np.array([fail_of(res[c["ba"] + (i,)], b) for i in ids], bool)
        n10, n01 = int(np.sum(rf & ~bf)), int(np.sum(bf & ~rf))
        blk = dict(depth_mm=dict(ref=c["ref"][2], BA=c["ba"][2]), gain=dict(ref=c["ref"][1], BA=c["ba"][1]),
                   ref=dstats(c["ref"], b, ids), BA=dstats(c["ba"], b, ids),
                   paired=dict(n=len(ids), both_fail=int(np.sum(rf & bf)), both_pass=int(np.sum(~rf & ~bf)),
                               ref_fails_BA_passes=n10, BA_fails_ref_passes=n01, mcnemar_exact_p=exact_two_sided(n10, n01)),
                   P_fail_BA_minus_ref=(float(bf.sum()) - float(rf.sum())) / len(ids) if ids else None,
                   samples_used="all" if len(ids) == n else ids)
        per_speed[f"{rpm:.0f}"]["b"].setdefault(lab(b), {})[c["kind"]] = blk
        pairs.setdefault((lab(b), c["kind"]), {})[rpm] = {i: (bool(x), bool(y)) for i, x, y in zip(ids, rf, bf)}
    # pooled over the speeds
    summ = {}
    for (lb, kind), bysp in pairs.items():
        blks = [per_speed[f"{rpm:.0f}"]["b"][lb][kind] for rpm in bysp]
        des = {}
        for side in ("ref", "BA"):
            k = sum(x[side]["failures"] for x in blks)
            m = sum(x[side]["n"] for x in blks)
            des[side] = dict(n=m, failures=k, P_fail=(k / m) if m else None, cp95=cp95(k, m),
                             unsettled=sum(x[side]["unsettled"] for x in blks), kick=sum(x[side]["kick"] for x in blks),
                             cf_linearly_unstable=sum(x[side]["cf_linearly_unstable"] for x in blks),
                             nominal_fail_rpm=[float(rpm) for rpm, x in zip(bysp, blks)
                                               if x[side]["nominal"] and x[side]["nominal"]["fail"]])
        pd_ = {k: sum(x["paired"][k] for x in blks) for k in ("n", "both_fail", "both_pass", "ref_fails_BA_passes",
                                                               "BA_fails_ref_passes")}
        pd_["mcnemar_exact_p_pooled"] = exact_two_sided(pd_["ref_fails_BA_passes"], pd_["BA_fails_ref_passes"])
        common = set.intersection(*[set(v) for v in bysp.values()]) if bysp else set()
        pos = neg = ties = 0
        for i in common:
            dd = sum(int(bysp[rpm][i][0]) - int(bysp[rpm][i][1]) for rpm in bysp)
            pos, neg, ties = pos + (dd > 0), neg + (dd < 0), ties + (dd == 0)
        summ.setdefault(lb, {})[kind] = dict(
            speeds=[float(r) for r in bysp], ref=des["ref"], BA=des["BA"], paired=pd_,
            sign_test_over_samples=dict(samples=len(common), ref_fails_at_more_speeds=pos, BA_fails_at_more_speeds=neg,
                                        ties=ties, p_exact_two_sided=exact_two_sided(pos, neg)),
            P_fail_BA_minus_ref=(des["BA"]["P_fail"] - des["ref"]["P_fail"]) if des["ref"]["n"] else None,
            speeds_BA_fails_more=[float(r) for r, x in zip(bysp, blks) if x["BA"]["failures"] > x["ref"]["failures"]],
            speeds_ref_fails_more=[float(r) for r, x in zip(bysp, blks) if x["ref"]["failures"] > x["BA"]["failures"]],
            speeds_equal=[float(r) for r, x in zip(bysp, blks) if x["ref"]["failures"] == x["BA"]["failures"]])
    # per design-depth point: the outcome of every sample (for figures and correlations)
    pts = []
    for pt, e in points.items():
        row = dict(rpm=pt[0], g=pt[1], a_mm=pt[2], b_mmps=[lab(b) for b in e["bs"]], roles=e["roles"],
                   nominal={lab(b): nominal(pt, b) for b in e["bs"]}, unsettled=[], fail={lab(b): [] for b in e["bs"]},
                   cf_stable=[], cf_a_lim_mm=[], tau_sim_s=[])
        for i in range(n):
            v, c = res.get(pt + (i,)), cfs.get(pt + (i,))
            row["unsettled"].append(None if v is None else int(v["rev"]["unsettled"]))
            for b in e["bs"]:
                row["fail"][lab(b)].append(None if v is None else int(fail_of(v, b)))
            row["cf_stable"].append(None if c is None else int(c["stable"]))
            row["cf_a_lim_mm"].append(None if c is None else c["a_lim_mm"])
            row["tau_sim_s"].append(MK.tau_for_sim(samples[i]["tau"], pt[0]))
        pts.append(row)
    units = [pt + (i,) for pt in points for i in [-1] + list(range(n))]
    recs = [r for u in units for r in idx.get(u, {}).values()]
    runs = dict(records=len(recs), cpu_s=float(sum(r.get("cpu", 0.0) for r in recs)),
                wall_s=float(sum(r.get("t", 0.0) for r in recs)),
                mean_cpu_s_per_run=float(np.mean([r["cpu"] for r in recs])) if recs else None,
                mean_wall_s_per_run=float(np.mean([r["t"] for r in recs])) if recs else None,
                chatter_rev=sum(bool(r["v"]["chat_rev"]) for r in recs), **meta)
    unres = [u for u in units if u not in res or u not in cfs]
    protocol = dict(
        speeds_rpm=sorted(D), n_samples=n, level=LEVEL, seed=MC.SEED,
        ranges={k: list(v) for k, v in MC.ranges(LEVEL).items()}, b_mmps=[b * 1e3 for b in B_REQ],
        designs=dict(ref=f"fair_designs.json speeds[n].{REF}: g, a_ref = a_chk",
                     BA="basin_aware.json per_speed[n].designs[b]: g (g_rob), a_rob_mm"),
        comparisons=dict(own="ref at (g_ref, a_ref) vs BA(b) at (g*(b), a_rob(b))",
                         common="both at a_c = min(a_ref, a_rob(b)), each with its own gain"),
        unsettled=("the kick of DV_LO = 1e-4 m/s ends in chatter at one of the phases 0, 4, 8, 12 (run in this order, "
                   "stopping at the first): mc_kick.critical_kick 'unsettled' at any phase, as in mc_kick.worst_phase"),
        failure="unsettled, or a kick of exactly b at one of the phases 0, 4, 8, 12 (in this order, stopping at the "
                "first chatter) ends in chatter",
        run="mc_kick.make_runner(rpm, a, vclip, runout, psi, tau_sim, g): face entry, nreg 0, |V| <= vclip, kick in "
            "revolution 100, 300-revolution window",
        judge="tdboundary.classify_run once per revolution (N = 1); the per-tooth verdict is recorded",
        phases=list(PHASES), n_phase=C.N_PHASE, kick_rev=C.KICK_REV, window=C.WINDOW, dv_lo_mps=DV_LO, nreg=C.NREG,
        delay="mc_kick.tau_for_sim", zeta_nominal=ZETA, build=SOURCE_SHA[:12], proto=PROTO,
        closed_form="diagnostic: mc_kick.closed_form's linear-stability test with mc_common.Model at the design gain",
        statistics="Clopper-Pearson two-sided 95 %; exact two-sided McNemar (binomial test of the discordant pairs at "
                   "p = 1/2); pooled over speeds: the same counts (approximate: shared samples) and a sample-level "
                   "exact sign test")
    return dict(note=NOTE, protocol=protocol, complete=not unres,
                progress=dict(units=len(units), resolved=len(units) - len(unres), missing=len(unres)),
                designs={f"{r:.0f}": dict(ref=d["ref"], BA=d["ba"]) for r, d in D.items()},
                summary=summ, per_speed=per_speed, points=pts, samples=samples,
                checks=checks(D, samples, n, idx), runs=runs)


def print_summary(out):
    for lb, kinds in out["summary"].items():
        for kind, s in kinds.items():
            r, b, pdd, st = s["ref"], s["BA"], s["paired"], s["sign_test_over_samples"]
            print(f"b {lb} mm/s {kind:6s}: ref {r['failures']}/{r['n']} (unsettled {r['unsettled']}, kick {r['kick']}), "
                  f"BA {b['failures']}/{b['n']} (unsettled {b['unsettled']}, kick {b['kick']}); only ref fails "
                  f"{pdd['ref_fails_BA_passes']}, only BA fails {pdd['BA_fails_ref_passes']}, McNemar p "
                  f"{pdd['mcnemar_exact_p_pooled']:.3g} (pooled), sign test p {st['p_exact_two_sided']:.3g} "
                  f"({st['ref_fails_at_more_speeds']}:{st['BA_fails_at_more_speeds']})", flush=True)
    for rpm, ps in out["per_speed"].items():
        for lb, kinds in ps["b"].items():
            for kind, blk in kinds.items():
                r, b = blk["ref"], blk["BA"]
                nr = r["nominal"]["fail"] if r["nominal"] else None
                nb = b["nominal"]["fail"] if b["nominal"] else None
                print(f"  {rpm} rpm b {lb} {kind:6s}: ref g {r['g']:.1f} a {r['a_mm']:.2f} mm {r['failures']}/{r['n']} "
                      f"(U {r['unsettled']}, cf-unstable {r['cf_linearly_unstable']}, nominal fail {nr}) | BA g {b['g']:.1f} "
                      f"a {b['a_mm']:.2f} mm {b['failures']}/{b['n']} (U {b['unsettled']}, cf-unstable "
                      f"{b['cf_linearly_unstable']}, nominal fail {nb}) | McNemar p {blk['paired']['mcnemar_exact_p']:.3g}",
                      flush=True)


# -------------------------------------------------------------------- main ----
def run_pool(a, jobs, idx, cfs, meta, mx, nmax, write):
    """Run the pending units on a.workers processes; one progress line per unit, a partial JSON every
    a.write_every seconds (the runs themselves are appended to the cache by the workers)."""
    left, nruns, wall_runs, last = nmax, 0, 0.0, time.time()
    with Pool(a.workers, initializer=_init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(unit_job, jobs), 1):
            u = r["unit"]
            for rec in r["new"]:
                idx.setdefault(u, {})[(rec["k"][4], rec["k"][5])] = rec
            if r["cf"] is not None:
                cfs[u] = r["cf"]["v"]
            meta["session_new_runs"] += len(r["new"])
            meta["session_cpu_s"] += r["cpu"]
            meta["restored_after_every_unit"] &= r["restored"]
            left -= mx[u]
            nruns += len(r["new"])
            wall_runs += sum(x["t"] for x in r["new"])
            rv = r["rev"]
            verdict = (f"unsettled (phase {rv['unsettled_phase']})" if rv["unsettled"] else "settled, kick " + ", ".join(
                f"{b} {'CHATTER@' + str(ph) if ph is not None else 'ok'}" for b, ph in rv["kick_phase"].items()))
            if r["tooth"] is not None:
                tv = r["tooth"]
                verdict += " | tooth: " + ("unsettled" if tv["unsettled"] else ", ".join(
                    f"{b} {'CHATTER' if ph is not None else 'ok'}" for b, ph in tv["kick_phase"].items()))
            cf = cfs.get(u)
            al = cf["a_lim_mm"] if cf and cf["a_lim_mm"] is not None else float("nan")
            eta = left * (wall_runs / nruns) / a.workers / 3600 if nruns else float("nan")
            print(f"[{k}/{len(jobs)}] {u[0]:.0f} rpm g {u[1]:.1f} a {u[2]:.2f} mm "
                  f"{'nominal' if u[3] < 0 else 'sample ' + str(u[3])}: {verdict}; cf "
                  f"{'stable' if cf and cf['stable'] else 'UNSTABLE'} (a_lim {al:.2f} mm); {len(r['new'])} runs "
                  f"{r['cpu']:.0f} CPU-s; at most {left} runs left, ETA <= {eta:.2f} h", flush=True)
            if time.time() - last > a.write_every:
                write()
                last = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="samples (the first n of mc_common.draw(n, 0.1))")
    ap.add_argument("--speeds", default=",".join(f"{s:.0f}" for s in SPEEDS), help="comma-separated rpm")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "mc_designs.json"), help="output JSON (cache: <out>_cache.jsonl)")
    ap.add_argument("--smoke", action="store_true", help="write mc_designs_smoke.json / mc_designs_smoke_cache.jsonl")
    ap.add_argument("--plan", action="store_true", help="print the runs needed and the estimate only")
    ap.add_argument("--assemble", action="store_true", help="no simulation: the JSON from the cache")
    ap.add_argument("--write-every", type=float, default=600.0, help="seconds between partial JSON writes")
    a = ap.parse_args()
    out_path = os.path.join(HERE, "mc_designs_smoke.json") if a.smoke else os.path.abspath(a.out)
    cpath = cache_path(out_path)
    speeds = tuple(float(x) for x in a.speeds.split(",") if x.strip())
    t_start = time.time()
    MC.restore_nominal()
    D = load_designs(speeds)
    comps, points = plan(D)
    samples = sample_params(a.n)
    idx, cfs, bad, other = load_cache(cpath)
    units = [(pt + (i,), e["bs"]) for i in [-1] + list(range(a.n)) for pt, e in points.items()]
    todo = [(u, bs) for u, bs in units if u not in cfs or resolve(u, bs, idx) is None]
    mx = {u: max_new_runs(u, bs, idx) for u, bs in todo}
    nmax, nmin = sum(mx.values()), sum(1 for v in mx.values() if v > 0)
    t_run, src = per_run_estimate(idx)
    print(f"mc_designs: {len(speeds)} speeds, {len(points)} design-depth points, {a.n} samples + nominal = {len(units)} "
          f"units; protocol {PROTO}, cache {os.path.relpath(cpath, ROOT)}", flush=True)
    print(f"  cache: {sum(len(d) for d in idx.values())} runs, {len(cfs)} closed-form records ({bad} unreadable lines, "
          f"{other} of another protocol); {len(units) - len(todo)} units resolved", flush=True)
    print(f"  to run: {len(todo)} units, at most {nmax} simulator runs (every cut settles and no kick of b chatters), "
          f"at least {nmin}", flush=True)
    print(f"  estimate at {t_run:.2f} s per run ({src}): at most {nmax * t_run / 3600:.2f} CPU-h, "
          f"{nmax * t_run / 3600 / max(a.workers, 1):.2f} h with {a.workers} workers", flush=True)
    if a.plan:
        return
    meta = dict(session_new_runs=0, session_cpu_s=0.0, restored_after_every_unit=True)

    def write(final=False):
        o = assemble(D, comps, points, samples, a.n, idx, cfs, dict(meta, runtime_s=time.time() - t_start))
        save(out_path, o)
        if final:
            print(f"wrote {os.path.relpath(out_path, ROOT)} (complete {o['complete']}, {o['progress']['resolved']}/"
                  f"{o['progress']['units']} units)", flush=True)
        return o
    if todo and not a.assemble:
        jobs = [(u[0], u[1], u[2], u[3], bs, NOMINAL_P if u[3] < 0 else samples[u[3]],
                 {k: r["v"] for k, r in idx.get(u, {}).items()}, u not in cfs, cpath) for u, bs in todo]
        try:
            run_pool(a, jobs, idx, cfs, meta, mx, nmax, write)
        except BaseException:                              # interrupted or a worker failed: keep what is done
            try:
                write()
                print(f"stopped: partial results written to {os.path.relpath(out_path, ROOT)}; every finished run is "
                      f"in {os.path.relpath(cpath, ROOT)}", flush=True)
            except Exception as e:                         # noqa: BLE001
                print(f"stopped; could not write partial results ({e})", flush=True)
            raise
    o = write(final=True)
    if not o["runs"]["restored_after_every_unit"]:
        print("WARNING: a worker did not return to the nominal parameters after a unit", flush=True)
    ck = o["checks"]
    nv = ck["nominal_vs_basin_aware_validation"]
    print(f"checks: samples = mc_closed_form {ck['samples_equal_mc_closed_form_pm10']}, = mc_kick "
          f"{ck['samples_equal_mc_kick']}; nominal per-tooth kicks vs basin_aware validation {nv['kicks_agree']}/"
          f"{nv['kicks_compared']} (per revolution {nv['kicks_agree_rev']}), consistent with the stored critical kicks "
          f"{nv['crit_consistent']}/{nv['crit_compared']}, DV_LO {nv['dv_lo_consistent']}/{nv['dv_lo_compared']}; "
          f"nominal rev = tooth {ck['nominal_rev_vs_tooth']['agree']}/{ck['nominal_rev_vs_tooth']['runs']}", flush=True)
    print_summary(o)
    print(f"done in {time.time() - t_start:.0f} s ({o['runs']['session_new_runs']} new runs, "
          f"{o['runs']['session_cpu_s']:.0f} CPU-s in the workers)", flush=True)


if __name__ == "__main__":
    main()
