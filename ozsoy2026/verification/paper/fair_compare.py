"""Like-for-like comparison of the basin-aware chart with the reference designs (review round 2, 4.1).

Purpose.  The referee pointed out that the reference schedule of the manuscript used g <= 1500 V s/m
while the basin-aware chart searched g <= 3000 V s/m, and that the two were not compared with the
same metrics.  fair_designs.py built reference designs with the chart's gain range and the common
operating-depth rule; this script evaluates every design, references and basin-aware alike, with the
same metrics at the same 21 speeds, and computes the certificates of the references too.

Designs (at the 21 speeds 500:125:3000 rpm of the chart):
  pub, auth0, auth10, auth1500   fair_designs.json speeds['<rpm>'][design]: g, a_lim and the checked
            depth a_chk in mm (fair_designs.py: the published condition and the authority condition
            with voltage margin 0 and 10 %, all with g <= 3000; the manuscript's schedule, g <= 1500).
            Operating depth a_op = a_chk.
  ba1, ba2, ba5   basin_aware.json per_speed['<rpm>']['designs']['1' | '2' | '5']: g* and a_op = a_rob
            (beta_req = 1, 2, 5 mm/s).
  Designs with the same (rpm, g, a_op) share their jobs.

Protocol (definitions; damping of every stored result: params.set_damping(gainmap_main.npz zeta)).
  bbar        basin_aware.bbar_of(rpm, g, a_op): the rigorous bound on the forced command |g x_p'(t)|
              over the tooth period (c3_periodic.orbit), V; voltage margin 1 - bbar / 9 V.
  beta_op     C2 certified kick at a_op with the default order N = c2_certificate.nodes_for(rpm):
              basin_aware.certificate(rpm, g, a_op, N, 'FAST', bbar) (the C2 kick rescaled to the
              rigorous bound), and if that is below 5 mm/s also 'TIGHT', the larger kept (as
              basin_aware.recert).  No certificate (0) where bbar >= 9 V or a >= a_lim, as in
              basin_aware.scan_gain.  ba designs: the stored default-order value
              (beta_default_N_at_a_rob_mmps); basin_aware.recert tried TIGHT only below the design's
              own beta_req, so some of these are FAST-only between beta_req and 5 mm/s (counted in
              checks; --ba-tight adds the TIGHT certificate there, off by default).
  beta_range  min of the reduced-order certificate (N = basin_aware.n_red(rpm) = nodes_for - 8,
              'FAST', the same rescaling and 0 rule) over the grid depths 2, 4, ... mm not above a_op
              and a_op itself: the quantity the chart constrains.  Rows of the basin-aware scan cache
              (basin_aware_cache/<id>/speed_<rpm>.jsonl, key ['scan', rpm, g]) are reused where the
              same (rpm, g, depth) exists (gain equal to 1e-6 V s/m: basin_aware keyed its gains by
              round(g, 6)).  ba designs: the stored min_beta_up_to_a_rob_mmps (default order at the
              depths basin_aware.recert re-certified, reduced order elsewhere); the purely
              reduced-order minimum is reported beside it (beta_range.reduced_only).
  crit_kick   simulated critical kick at a_op, the smallest over the kick phases 0, 4, 8, 12 of 16:
              basin_aware.kick_job(('crit', rpm, a, g, phase)) = c1_unsafe_zone._critical_kick (face
              entry, full surface memory, |V| <= 9 V, kick at revolution 100, 300-revolution window,
              classifier D1; DV_GRID 0.1 mm/s ... 3 m/s searched upwards, the first interval that ends
              in chatter bisected 6 times in log dv, the upper end returned).  A phase whose unkicked
              cut does not settle (nan) counts as 0 (no basin); inf = no chatter up to 3 m/s (stored
              as null with kind 'inf').  Stored results with the same job key
              (basin_aware_cache/<id>/validate.jsonl: the ba designs and auth1500 at the six
              validation speeds 750, 1125, 1375, 2000, 2500, 2875 rpm) are reused after the check below.
  entries     basin_aware.kick_job(('entry', rpm, a, g, 'step' | 'face')): one step entry from rest
              and one face entry with the 9 V clip and the full surface memory, judged by
              tdboundary.classify (D1): stable, max_x_mm, max_v_V.  Recomputed for every design (for
              the ba designs also compared with the stored entries_9V).
  admissible  the chart's admissibility of a design at its own depth, per beta_req = 1, 2, 5 mm/s:
              beta_range >= beta_req, bbar(a_op) <= 8.1 V and a_op < a_lim.  'admissible_strict'
              also asks bbar <= 8.1 V at every grid depth below a_op (the chart asks (V) at every grid
              depth up to a_rob, and bbar is not monotone in depth) and a_op <= 32 mm.
  check       before stored results are reused: basin_aware.json protocol (zeta, 9 V clip, phases
              0, 4, 8, 12, cache id = basin_aware.PROTO), the c1_unsafe_zone constants of the stored
              protocol (KICK_REV 100, WINDOW 300, N_PHASE 16, N_BISECT 6, NREG 0, kicks 1e-4 ... 3 m/s),
              the JSON validation block equal to the cache and its design points equal to the designs
              here; then two stored jobs are recomputed first: one critical kick (must agree bit for
              bit, else no stored kick is reused) and one default-order certificate (relative
              difference <= 1e-6, else no scan row is reused).
  summary     per design over the speeds: mean, median, min of a_op; mean g; median and min of beta_op
              and of beta_range and the numbers of speeds with each >= 1, 2, 5 mm/s; median and min
              crit_kick and the numbers of speeds with crit_kick < 1, 2, 5 mm/s; median and max bbar;
              the numbers of stable step and face entries; per beta_req the number of speeds at which
              the design at its own depth satisfies the chart's admissibility (the question for the
              reference designs; a check for the ba designs).

Run (from ozsoy2026/; resumable: every finished job is appended to fair_compare_cache.jsonl, and a
restart skips the jobs found there):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup python3 verification/paper/fair_compare.py \\
        --workers 4 > verification/paper/fair_compare.log 2>&1 &
  --plan lists the jobs still to run with a CPU-time estimate and exits (computes nothing).
  --ba-tight (symmetric TIGHT rule for the ba designs) must write elsewhere but can share the cache:
    ... fair_compare.py --ba-tight --out fair_compare_batight --cache fair_compare --workers 4
  Smoke test (writes only fair_compare_smoke.json and fair_compare_smoke_cache.jsonl):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/fair_compare.py --smoke \\
        --speeds 2500 --designs auth10,ba1 --phases 0 --workers 1
Writes verification/paper/fair_compare.json.
"""
import argparse
import glob
import hashlib
import json
import os
import queue
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, "results")
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])

import params as P                                    # noqa: E402
P.set_damping(ZETA)                                   # before any module that reads the damping
import basin_aware as BA                              # noqa: E402  (helpers; its jobs set the same damping)
import c1_unsafe_zone as C1                           # noqa: E402
import c2_certificate as C2                           # noqa: E402
from tdsim import SOURCE_SHA                          # noqa: E402
from zoa import a_lim as zoa_a_lim                    # noqa: E402

FD_JSON = os.path.join(HERE, "fair_designs.json")
BA_JSON = os.path.join(HERE, "basin_aware.json")
SPEEDS = tuple(float(n) for n in np.arange(500.0, 3001.0, 125.0))
REF = ("pub", "auth0", "auth10", "auth1500")
BAD = {"ba1": "1", "ba2": "2", "ba5": "5"}              # design -> basin_aware.json beta_req label (mm/s)
DESIGNS = REF + tuple(BAD)
PHASES = tuple(int(p) for p in BA.VAL_PHASES)         # 0, 4, 8, 12 of c1_unsafe_zone.N_PHASE = 16
LEVELS = (1.0, 2.0, 5.0)                              # mm/s: the chart's beta_req and the count thresholds
TIGHT_BELOW = 5e-3                                    # m/s: default-order FAST kicks below it are redone with TIGHT
G_TOL = 1e-6                                          # V s/m: a basin-aware scan row is for the same gain
CERT_RTOL = 1e-6                                      # recheck of a stored certificate
VMAX, VLIM, A_PHYS, DEPTHS = BA.VMAX, BA.VLIM, BA.A_PHYS, BA.DEPTHS
STORED_C1 = dict(KICK_REV=100, WINDOW=300, N_PHASE=16, N_BISECT=6, NREG=0, DV_LO=1e-4, DV_HI=3.0, VMAX=9.0)
VAL_LABEL = {"ba1": "1", "ba2": "2", "ba5": "5", "auth1500": "auth"}   # basin_aware.json validation entries

# cost model for the scheduling (longest first) and for --plan.  Measured: smoke test (2500 rpm, one worker)
# 1.29 s per simulated run, 1.28 s per reduced-order certificate (N = 12), 3.4-4.5 s per default-order one
# (N = 20), 0.95 s per entry; single 400-revolution runs 1.2-1.3 s at 500, 750 and 2500 rpm (the cost per
# run does not depend on the speed).  Other orders: means of the basin-aware cache (same code).
T_RUN = 1.3          # s per simulated run (400 revolutions with the face entry)
RUNS_CRIT = 15       # simulated runs per critical kick (1 + grid steps to the first chatter + 6 bisections;
#                      12-18, mean 14.5, in basin_aware_cache/<id>/validate.jsonl)
T_ENTRY = 1.0        # s per entry run
CERT_S = dict(reduced={12: 1.4, 16: 3.2, 20: 7.9, 24: 13.4},     # s per c2.FAST certificate
              default={20: 8.8, 24: 18.5, 28: 65.9, 32: 54.1})
TIGHT_FACTOR = 1.5   # c2.TIGHT against c2.FAST (one stored pair: 45.5 s against 41.6 s)


# ------------------------------------------------------------------ utilities ----
class Cache:
    """Append-only JSON-lines cache (as basin_aware.JL): key (a JSON list) -> value.  An unfinished
    last line (an interrupted write) is dropped and the file truncated after the last full line."""

    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            good = 0
            for line in data.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break
                self.d[json.dumps(rec["k"])] = rec["v"]
                good += len(line)
            if len(data) > good:
                with open(path, "r+b") as fh:
                    fh.truncate(good)

    def get(self, k):
        return self.d.get(json.dumps(k))

    def has(self, k):
        return json.dumps(k) in self.d

    def put(self, k, v):
        self.d[json.dumps(k)] = v
        with open(self.path, "a") as fh:
            fh.write(json.dumps(dict(k=k, v=v), default=float) + "\n")
            fh.flush()
        return v


def read_jsonl(path):
    """Read-only view of a basin_aware cache file: {json key: value} (nothing is written)."""
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break
                out[json.dumps(rec["k"])] = rec["v"]
    return out


def sha12(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def fin(x):
    """JSON-safe number: non-finite -> None."""
    return None if x is None or not np.isfinite(x) else float(x)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return fin(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def same(a, b, rtol=0.0):
    """Equal floats (nan == nan, inf == inf), to a relative tolerance."""
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b))


# ------------------------------------------------------------------ job keys ----
def cert_key(rpm, g, d, N, st):
    """Certificate at depth d (mm) with order N and solver settings st."""
    return ["cert", float(rpm), float(g), float(d), int(N), st]


def crit_key(rpm, g, d, ph):
    """Critical kick at depth d (mm), kick phase ph: the job key of basin_aware's validate.jsonl."""
    return ["crit", float(rpm), float(d) * 1e-3, float(g), int(ph)]


def entry_key(rpm, g, d, kind):
    return ["entry", float(rpm), float(d) * 1e-3, float(g), kind]


def is_default_fast(k):
    return k[0] == "cert" and k[5] == "FAST" and k[4] == C2.nodes_for(k[1])


def needs_tight(v):
    return v["beta"] < TIGHT_BELOW and not str(v.get("note", "")).startswith("not computed")


def est_cost(k):
    """Estimated CPU seconds of a job (see the cost model above)."""
    if k[0] in ("cert", "recheck_cert"):
        tab = CERT_S["default" if k[4] == C2.nodes_for(k[1]) else "reduced"]
        t = tab.get(k[4], 1.4 * (k[4] / 12.0) ** 3.4)
        return t * (TIGHT_FACTOR if k[5] == "TIGHT" else 1.0)
    if k[0] in ("crit", "recheck_crit"):
        return T_RUN * RUNS_CRIT
    return T_ENTRY


# ------------------------------------------------------------------- workers ----
def _init():
    P.set_damping(ZETA)


def cert_value(rpm, g, d, N, st):
    """Certified kick at depth d (mm) (basin_aware.certificate: rescaled to the rigorous command
    bound), computed only below the linear limit and with bbar < 9 V, else 0 (basin_aware.scan_gain)."""
    a = d * 1e-3
    bb = BA.bbar_of(rpm, g, a)
    al = float(zoa_a_lim(rpm, g)) * 1e3
    if d < al and bb < VMAX:
        rec = BA.certificate(rpm, g, a, N, st, bb)
    else:
        rec = dict(beta=0.0, beta_raw=0.0, N=int(N), settings=st, t=0.0,
                   note="not computed: " + ("a >= a_lim" if d >= al else "bbar >= 9 V"))
    rec.update(bbar=float(bb), a_lim=al, lin=bool(d < al))
    return rec


def run_job(key):
    """One unit of work -> (key, value, seconds)."""
    P.set_damping(ZETA)
    t = time.time()
    kind = key[0]
    if kind in ("cert", "recheck_cert"):
        v = cert_value(*key[1:])
    elif kind in ("crit", "entry"):
        v = BA.kick_job(tuple(key))[1]
    elif kind == "recheck_crit":
        v = BA.kick_job(("crit",) + tuple(key[1:]))[1]
    else:
        raise ValueError(f"unknown job {key}")
    return key, v, round(time.time() - t, 2)


def describe(k, v):
    if k[0] in ("cert", "recheck_cert"):
        return (f"{k[0]} {k[1]:.0f} rpm g {k[2]:.1f} d {k[3]:.2f} mm N {k[4]} {k[5]}: beta {v['beta'] * 1e3:.3f} mm/s, "
                f"bbar {v['bbar']:.2f} V")
    if k[0] in ("crit", "recheck_crit"):
        return f"{k[0]} {k[1]:.0f} rpm a {k[2] * 1e3:.2f} mm g {k[3]:.1f} phase {k[4]}: {v * 1e3:.4f} mm/s"
    return (f"{k[4]} entry {k[1]:.0f} rpm a {k[2] * 1e3:.2f} mm g {k[3]:.1f}: stable {v['stable']}, "
            f"max_x {v['max_x_mm']:.3f} mm, max_v {v['max_v_V']:.2f} V")


def run_jobs(todo, cache, workers, followup=None):
    """Run the jobs longest first (est_cost), at most `workers` in flight, appending each result to the
    cache as it arrives; follow-up jobs (TIGHT certificates) are queued as their FAST results arrive.
    Returns the failed jobs (not cached: a restart retries them)."""
    todo = sorted(todo, key=est_cost, reverse=True)
    if not todo:
        return []
    q = queue.Queue()
    inflight, done, total, failed = 0, 0, len(todo), []
    t0, cpu = time.time(), 0.0
    with Pool(workers, initializer=_init) as pool:
        def submit():
            nonlocal inflight
            k = todo.pop(0)
            pool.apply_async(run_job, (k,), callback=q.put, error_callback=lambda e, k=k: q.put((k, e, None)))
            inflight += 1
        while todo and inflight < workers:
            submit()
        while inflight:
            k, v, t = q.get()
            inflight -= 1
            if isinstance(v, BaseException):
                failed.append(dict(key=k, error=repr(v)))
                print(f"  FAILED {k}: {v!r}", flush=True)
            else:
                cache.put(k, dict(v=v, t=t))
                done, cpu = done + 1, cpu + t
                print(f"  [{done}/{total}] {describe(k, v)} ({t:.1f} s; {time.time() - t0:.0f} s elapsed)", flush=True)
                for f in (followup(k, v) if followup else []):
                    if not cache.has(f) and f not in todo:
                        todo.append(f)
                        total += 1
                        todo.sort(key=est_cost, reverse=True)
            while todo and inflight < workers:
                submit()
    print(f"  {done} jobs in {time.time() - t0:.0f} s wall, {cpu:.0f} s CPU", flush=True)
    return failed


def tight_followup(k, v):
    return [cert_key(k[1], k[2], k[3], k[4], "TIGHT")] if is_default_fast(k) and needs_tight(v) else []


# -------------------------------------------------------------------- inputs ----
def load_inputs(speeds, designs):
    fd, ba = json.load(open(FD_JSON)), json.load(open(BA_JSON))
    tab = {}
    for n in speeds:
        s = f"{n:.0f}"
        for d in designs:
            if d in BAD:
                dz = ba["per_speed"][s]["designs"][BAD[d]]
                if dz.get("g") is None or not dz.get("a_rob_mm"):
                    tab[(n, d)] = None                       # no basin-aware design at this speed
                    continue
                tab[(n, d)] = dict(g=float(dz["g"]), a_op=float(dz["a_rob_mm"]), a_lim=float(dz["a_lim_at_g_mm"]),
                                   source=f"basin_aware.json per_speed['{s}']['designs']['{BAD[d]}']: g*, a_rob")
            else:
                r = fd["speeds"][s][d]
                tab[(n, d)] = dict(g=float(r["g"]), a_op=float(r["a_chk"]), a_lim=float(r["a_lim"]),
                                   source=f"fair_designs.json speeds['{s}']['{d}']: g, a_chk"
                                          + (f" ({r['source']})" if r.get("source") else ""))
    return tab, fd, ba


def points_of(tab):
    """Unique (rpm, g, a_op) -> designs at it."""
    pts = {}
    for (n, d), z in tab.items():
        if z is not None:
            pts.setdefault((n, z["g"], z["a_op"]), []).append(d)
    return pts


def load_stored():
    """Read-only view of the basin-aware cache: scan rows {rpm: {g: {depth: row}}}, default-order
    certificates ('ref' records) and the validation jobs."""
    scan, ref = {}, {}
    for f in glob.glob(os.path.join(BA.CACHE, "speed_*.jsonl")):
        for kk, v in read_jsonl(f).items():
            k = json.loads(kk)
            if k[0] == "scan":
                scan.setdefault(float(k[1]), {})[float(k[2])] = {float(r["d"]): r for r in v["rows"]}
            elif k[0] == "ref":
                ref[kk] = v
    return dict(scan=scan, ref=ref, val=read_jsonl(os.path.join(BA.CACHE, "validate.jsonl")))


def scan_rows(stored, rpm, g):
    """Rows {depth: row} of the basin-aware scan of (rpm, g) and the scan's gain ({}, None if none)."""
    for gs, rows in stored["scan"].get(float(rpm), {}).items():
        if abs(gs - g) <= G_TOL:
            return rows, gs
    return {}, None


def range_depths(a_op):
    """The grid depths not above a_op, and a_op itself (mm)."""
    ds = [d for d in DEPTHS if d <= a_op]
    return ds + ([a_op] if a_op not in ds else [])


# ------------------------------------------------------- stored-protocol check ----
def stored_checks(ba, stored, tab, speeds, phases):
    """Is a stored critical kick the same job as here?  (Rechecked by recomputation separately.)"""
    p = ba["protocol"]
    c1 = dict(KICK_REV=C1.KICK_REV, WINDOW=C1.WINDOW, N_PHASE=C1.N_PHASE, N_BISECT=C1.N_BISECT, NREG=C1.NREG,
              DV_LO=float(C1.DV_LO), DV_HI=float(C1.DV_HI), VMAX=float(C1.VMAX))
    out = dict(zeta=bool(p["zeta"] == ZETA == BA.ZETA), clip=bool(p["vmax_V"] == VMAX == C1.VMAX),
               phases=bool(list(p["val_phases"]) == list(PHASES) and set(phases) <= set(PHASES)),
               cache_id=bool(p["cache"] == BA.PROTO), c1_constants=bool(c1 == STORED_C1),
               dv_grid=bool(np.array_equal(C1.DV_GRID, np.append(1e-4 * 2.0 ** np.arange(15), 3.0))),
               validation_json_equals_cache=True, design_points_equal=True, mismatches=[], compared=0)
    for n in speeds:
        vrow = ba["validation"].get(f"{n:.0f}")
        if vrow is None:
            continue
        for d, lab in VAL_LABEL.items():
            v = vrow.get(lab)
            if v is None:
                continue
            a_v = v["a_chk_mm"] if lab == "auth" else v["a_rob_mm"]
            z = tab.get((n, d))
            if z is not None and (z["g"] != v["g"] or z["a_op"] != a_v):
                out["design_points_equal"] = False
                out["mismatches"].append(dict(rpm=n, design=d, json=[v["g"], a_v], here=[z["g"], z["a_op"]]))
            for i, ph in enumerate(PHASES):
                sv = stored["val"].get(json.dumps(crit_key(n, v["g"], a_v, ph)))
                out["compared"] += 1
                if sv is None or not same(sv["v"] * 1e3, v["crit_kick_mmps_by_phase"][i], 1e-12):
                    out["validation_json_equals_cache"] = False
                    out["mismatches"].append(dict(rpm=n, design=d, phase=ph, cache=None if sv is None else sv["v"],
                                                  json_mmps=v["crit_kick_mmps_by_phase"][i]))
    out["ok"] = all(out[k] for k in ("zeta", "clip", "phases", "cache_id", "c1_constants", "dv_grid",
                                     "validation_json_equals_cache", "design_points_equal"))
    return out


def recheck_plan(ba, stored, speeds, phases):
    """Two stored jobs to recompute: a critical kick of ba1 at a validation speed of the run (2500 rpm
    if present) and the default-order c2.FAST certificate of ba1 at its a_rob (same speed choice)."""
    jobs = {}
    vs = [n for n in speeds if n in BA.VAL_SPEEDS and f"{n:.0f}" in ba["validation"]]
    if vs:
        n = 2500.0 if 2500.0 in vs else max(vs)
        v = ba["validation"][f"{n:.0f}"].get("1")
        k = None if v is None else crit_key(n, v["g"], v["a_rob_mm"], phases[0])
        sv = None if k is None else stored["val"].get(json.dumps(k))
        if sv is not None:
            jobs["crit"] = dict(key=["recheck_crit"] + k[1:], stored_key=k, stored=sv["v"])
    n = 2500.0 if 2500.0 in speeds else max(speeds)
    dz = ba["per_speed"][f"{n:.0f}"]["designs"]["1"]
    if dz.get("g") is not None and dz.get("a_rob_mm"):
        rk = ["ref", n, dz["g"], dz["a_rob_mm"], "FAST"]
        sv = stored["ref"].get(json.dumps(rk))
        if sv is not None:
            jobs["cert"] = dict(key=["recheck_cert", n, float(dz["g"]), float(dz["a_rob_mm"]), C2.nodes_for(n), "FAST"],
                                stored_key=rk, stored=sv)
    return jobs


def recheck_result(rc, cache):
    out = {}
    if "crit" in rc:
        v = cache.get(rc["crit"]["key"])
        out["crit"] = dict(key=rc["crit"]["stored_key"], stored_mmps=fin(rc["crit"]["stored"] * 1e3),
                           recomputed_mmps=None if v is None else fin(v["v"] * 1e3),
                           equal=None if v is None else same(v["v"], rc["crit"]["stored"]),
                           seconds=None if v is None else v["t"])
    if "cert" in rc:
        v = cache.get(rc["cert"]["key"])
        s = rc["cert"]["stored"]
        out["cert"] = dict(key=rc["cert"]["stored_key"], N=rc["cert"]["key"][4], stored_beta_mmps=s["beta"] * 1e3,
                           stored_beta_raw=s["beta_raw"],
                           recomputed_beta_mmps=None if v is None else v["v"]["beta"] * 1e3,
                           recomputed_beta_raw=None if v is None else v["v"]["beta_raw"],
                           rel_diff=None if v is None else abs(v["v"]["beta_raw"] - s["beta_raw"]) / max(s["beta_raw"], 1e-300),
                           equal=None if v is None else bool(same(v["v"]["beta_raw"], s["beta_raw"], CERT_RTOL)
                                                             and same(v["v"]["beta"], s["beta"], CERT_RTOL)),
                           bit_identical=None if v is None else bool(v["v"]["beta_raw"] == s["beta_raw"]),
                           seconds=None if v is None else v["t"], stored_seconds=s.get("t"))
    return out


# ---------------------------------------------------------------------- plan ----
def make_plan(pts, stored, phases, reuse_crit, reuse_scan, cache, ba_tight, ba):
    """Every job of the campaign (deduplicated) and the jobs still to run."""
    jobs, reused = [], dict(scan_rows=0, crit=0)
    for (n, g, a), ds in pts.items():
        Nd, Nr = C2.nodes_for(n), BA.n_red(n)
        if any(d in REF for d in ds):
            kf = cert_key(n, g, a, Nd, "FAST")
            jobs.append(kf)
            f = cache.get(kf)
            if f is not None and needs_tight(f["v"]):
                jobs.append(cert_key(n, g, a, Nd, "TIGHT"))
        elif ba_tight:                                           # ba point: TIGHT where the stored FAST < 5 mm/s
            chk = ba_check(ba, n, ds, a)
            if chk is not None and chk["tight_mmps"] is None and chk["fast_mmps"] < TIGHT_BELOW * 1e3:
                jobs.append(cert_key(n, g, a, Nd, "TIGHT"))
        rows = scan_rows(stored, n, g)[0] if reuse_scan else {}
        for d in range_depths(a):
            if d in rows:
                reused["scan_rows"] += 1
            else:
                jobs.append(cert_key(n, g, d, Nr, "FAST"))
        for ph in phases:
            k = crit_key(n, g, a, ph)
            if reuse_crit and json.dumps(k) in stored["val"] and not cache.has(k):
                reused["crit"] += 1
            else:
                jobs.append(k)
        jobs += [entry_key(n, g, a, kind) for kind in ("step", "face")]
    jobs = list({json.dumps(k): k for k in jobs}.values())
    return jobs, [k for k in jobs if not cache.has(k)], reused


def ba_check(ba, n, ds, a):
    """The stored default-order check of the (first) ba design at this point."""
    for d in ds:
        if d in BAD:
            return ba["per_speed"][f"{n:.0f}"]["designs"][BAD[d]]["default_N_checks"].get(str(a))
    return None


def job_label(k):
    if k[0] in ("cert", "recheck_cert"):
        return f"{k[0]} {'default' if k[4] == C2.nodes_for(k[1]) else 'reduced'} N={k[4]} {k[5]}"
    return f"entry {k[4]}" if k[0] == "entry" else k[0]


def print_plan(jobs, todo, reused, rc_todo, workers):
    label = job_label
    allk = rc_todo + todo
    rows = {}
    for k in allk:
        r = rows.setdefault(label(k), [0, 0.0])
        r[0] += 1
        r[1] += est_cost(k)
    print(f"plan: {len(jobs)} jobs in the campaign, {len(todo)} to run (+ {len(rc_todo)} recheck jobs); reused from "
          f"the basin-aware cache: {reused['scan_rows']} scan rows, {reused['crit']} critical kicks", flush=True)
    for lab in sorted(rows):
        print(f"  {lab:28s} {rows[lab][0]:5d} jobs  ~{rows[lab][1] / 3600:6.2f} CPU-h", flush=True)
    nd = sum(1 for k in todo if is_default_fast(k))
    tot = sum(est_cost(k) for k in allk)
    tmax = tot + sum(est_cost(cert_key(k[1], k[2], k[3], k[4], "TIGHT")) for k in todo if is_default_fast(k))
    print(f"  estimate ~{tot / 3600:.2f} CPU-h without TIGHT follow-ups, at most ~{tmax / 3600:.2f} CPU-h if all {nd} "
          f"default-order FAST kicks fall below 5 mm/s; wall ~ CPU / {workers} workers", flush=True)
    by = {}
    for k in allk:
        by[k[1]] = by.get(k[1], 0.0) + est_cost(k)
    print("  by speed (CPU-min): " + ", ".join(f"{n:.0f}: {v / 60:.0f}" for n, v in sorted(by.items())), flush=True)


# ------------------------------------------------------------------ assemble ----
def crit_value(k, cache, stored, reuse_crit):
    v = cache.get(k)
    if v is not None:
        return v["v"], "computed"
    sv = stored["val"].get(json.dumps(k)) if reuse_crit else None
    if sv is not None:
        return sv["v"], "basin_aware validate.jsonl"
    raise KeyError(f"missing job {k}")


def point_metrics(pt, ds, cache, stored, phases, reuse_crit, reuse_scan):
    """Metrics of one (rpm, g, a_op), shared by every design at it."""
    n, g, a = pt
    Nd, Nr = C2.nodes_for(n), BA.n_red(n)
    out = dict(bbar_V=BA.bbar_of(n, g, a * 1e-3), a_lim_zoa_mm=float(zoa_a_lim(n, g)) * 1e3)
    ft = [cache.get(cert_key(n, g, a, Nd, st)) for st in ("FAST", "TIGHT")]
    if ft[0] is not None or ft[1] is not None:                   # default order at a_op (computed here)
        f, t = (None if r is None else r["v"] for r in ft)
        best = max((r for r in (f, t) if r is not None), key=lambda r: r["beta"])
        out["beta_op"] = dict(beta_mmps=best["beta"] * 1e3, settings=best["settings"], N=Nd,
                              fast_mmps=None if f is None else f["beta"] * 1e3,
                              tight_mmps=None if t is None else t["beta"] * 1e3,
                              beta_raw_mmps=best.get("beta_raw", 0.0) * 1e3, bmax_c2_V=best.get("bmax_c2"),
                              rescale=best.get("fac"), note=best["note"], source="computed")
    rows, gs = scan_rows(stored, n, g) if reuse_scan else ({}, None)
    rr = []
    for d in range_depths(a):
        if d in rows:
            r = rows[d]
            rr.append(dict(d=d, bbar=r["bbar"], beta=r.get("beta", 0.0), src="scan",
                           note=r.get("note", "not computed (basin_aware.scan_gain: bbar >= 9 V or a >= a_lim)")))
        else:
            r = cache.get(cert_key(n, g, d, Nr, "FAST"))
            if r is None:
                raise KeyError(f"missing job {cert_key(n, g, d, Nr, 'FAST')}")
            rr.append(dict(d=d, bbar=r["v"]["bbar"], beta=r["v"]["beta"], src="computed", note=r["v"]["note"]))
    m = min(rr, key=lambda r: r["beta"])
    b = max(rr, key=lambda r: r["bbar"])
    out["beta_range"] = dict(beta_mmps=m["beta"] * 1e3, at_depth_mm=m["d"], N=Nr, settings="FAST",
                             depths_mm=[r["d"] for r in rr], beta_by_depth_mmps=[r["beta"] * 1e3 for r in rr],
                             bbar_by_depth_V=[r["bbar"] for r in rr], source_by_depth=[r["src"] for r in rr],
                             bbar_max_V=b["bbar"], bbar_max_depth_mm=b["d"],
                             rows_reused=sum(r["src"] == "scan" for r in rr),
                             rows_computed=sum(r["src"] == "computed" for r in rr), scan_gain=gs,
                             note_at_min=m["note"])
    vals, src = [], []
    for ph in phases:
        v, s = crit_value(crit_key(n, g, a, ph), cache, stored, reuse_crit)
        vals.append(v)
        src.append(s)
    arr = np.array(vals, float)
    eff = np.where(np.isnan(arr), 0.0, arr)                      # no settled cut: no basin
    i = int(np.argmin(eff))
    kinds = ["unsettled" if np.isnan(x) else ("inf" if np.isinf(x) else "exact") for x in arr]
    out["crit"] = dict(phases=list(phases), by_phase_mmps=[fin(x * 1e3) for x in arr], kind_by_phase=kinds,
                       worst_mmps=fin(eff[i] * 1e3), worst_kind=kinds[i], worst_phase=phases[i],
                       worst_effective_mmps=float(eff[i] * 1e3), source_by_phase=src)
    out["entries_9V"] = {kind: cache.get(entry_key(n, g, a, kind))["v"] for kind in ("step", "face")}
    return out


def design_record(n, d, z, pm, ds, ba, cache, ba_tight=False):
    rec = dict(g=z["g"], a_op_mm=z["a_op"], a_lim_mm=z["a_lim"], a_lim_zoa_mm=pm["a_lim_zoa_mm"],
               a_op_over_a_lim=z["a_op"] / z["a_lim"], source=z["source"], same_point_as=[x for x in ds if x != d],
               bbar_V=pm["bbar_V"], voltage_margin=1.0 - pm["bbar_V"] / VMAX)
    if d in BAD:
        dz = ba["per_speed"][f"{n:.0f}"]["designs"][BAD[d]]
        chk = dz["default_N_checks"].get(str(z["a_op"])) or {}
        bop = dz["beta_default_N_at_a_rob_mmps"]
        extra = {}
        tk = cache.get(cert_key(n, z["g"], z["a_op"], C2.nodes_for(n), "TIGHT")) if ba_tight else None
        if tk is not None:                                       # --ba-tight
            extra = dict(tight_added_mmps=tk["v"]["beta"] * 1e3, stored_mmps=bop)
            bop = max(bop, tk["v"]["beta"] * 1e3)
        rec.update(beta_op_mmps=bop,
                   beta_op=dict(beta_mmps=bop, N=C2.nodes_for(n), fast_mmps=chk.get("fast_mmps"),
                                tight_mmps=chk.get("tight_mmps"), note=chk.get("note"),
                                fast_only_below_5=bool(chk.get("tight_mmps") is None and (chk.get("fast_mmps") or 0) < 5.0),
                                source="basin_aware.json beta_default_N_at_a_rob_mmps (basin_aware.recert)", **extra),
                   beta_range_mmps=dz["min_beta_up_to_a_rob_mmps"],
                   beta_range=dict(beta_mmps=dz["min_beta_up_to_a_rob_mmps"],
                                   source="basin_aware.json min_beta_up_to_a_rob_mmps (default order where "
                                          "basin_aware.recert re-certified, reduced order elsewhere)",
                                   reduced_only=pm["beta_range"]),
                   bbar_stored_V=dz.get("bbar_at_a_rob_V"))
        st = dz.get("entries_9V") or {}
        rec["entries_stored"] = st
        rec["entries_equal_stored"] = bool(st and all(st.get(k) == pm["entries_9V"][k] for k in ("step", "face")))
    else:
        rec.update(beta_op_mmps=pm["beta_op"]["beta_mmps"], beta_op=pm["beta_op"],
                   beta_range_mmps=pm["beta_range"]["beta_mmps"], beta_range=pm["beta_range"])
    rec.update(crit_kick_worst_mmps=pm["crit"]["worst_mmps"], crit=pm["crit"], entries_9V=pm["entries_9V"])
    br, lin = rec["beta_range_mmps"], bool(z["a_op"] < z["a_lim"])
    volt = bool(rec["bbar_V"] <= VLIM)
    rec["admissible"] = {f"{b:g}": bool(br >= b and volt and lin) for b in LEVELS}
    rec["bbar_max_up_to_a_op_V"] = pm["beta_range"]["bbar_max_V"]          # grid depths <= a_op and a_op
    rec["admissible_strict"] = {f"{b:g}": bool(rec["admissible"][f"{b:g}"] and pm["beta_range"]["bbar_max_V"] <= VLIM
                                               and z["a_op"] <= A_PHYS) for b in LEVELS}
    return rec


def summarize(per_speed, designs, speeds):
    summ = {}
    for d in designs:
        recs = [(n, per_speed[f"{n:.0f}"][d]) for n in speeds if per_speed[f"{n:.0f}"].get(d)]
        if not recs:
            continue
        rpm = np.array([n for n, _ in recs])
        a = np.array([r["a_op_mm"] for _, r in recs])
        g = np.array([r["g"] for _, r in recs])
        bo = np.array([r["beta_op_mmps"] for _, r in recs])
        br = np.array([r["beta_range_mmps"] for _, r in recs])
        ck = np.array([r["crit"]["worst_effective_mmps"] for _, r in recs])     # unsettled = 0, inf = inf
        bb = np.array([r["bbar_V"] for _, r in recs])
        st = [r["entries_9V"]["step"]["stable"] for _, r in recs]
        fa = [r["entries_9V"]["face"]["stable"] for _, r in recs]
        s = dict(n_speeds=len(recs), rpm=rpm.tolist(),
                 a_op_mm=dict(mean=a.mean(), median=np.median(a), min=a.min(), min_at_rpm=rpm[np.argmin(a)]),
                 g_mean=g.mean(),
                 beta_op_mmps=dict(median=np.median(bo), min=bo.min(), min_at_rpm=rpm[np.argmin(bo)]),
                 beta_op_at_least={f"{b:g}": int(np.sum(bo >= b)) for b in LEVELS},
                 beta_range_mmps=dict(median=np.median(br), min=br.min(), min_at_rpm=rpm[np.argmin(br)]),
                 beta_range_at_least={f"{b:g}": int(np.sum(br >= b)) for b in LEVELS},
                 crit_kick_mmps=dict(median=fin(np.median(ck)), min=fin(ck.min()), min_at_rpm=rpm[np.argmin(ck)]),
                 crit_kick_below={f"{b:g}": int(np.sum(ck < b)) for b in LEVELS},
                 crit_kick_unsettled=int(sum(r["crit"]["worst_kind"] == "unsettled" for _, r in recs)),
                 crit_kick_inf=int(np.sum(np.isinf(ck))),
                 bbar_V=dict(median=np.median(bb), max=bb.max(), max_at_rpm=rpm[np.argmax(bb)]),
                 voltage_margin_min=1.0 - bb.max() / VMAX,
                 step_entry_stable=int(sum(st)), face_entry_stable=int(sum(fa)),
                 step_entry_chatter_rpm=[float(n) for (n, _), x in zip(recs, st) if not x],
                 face_entry_chatter_rpm=[float(n) for (n, _), x in zip(recs, fa) if not x],
                 chart_admissible_speeds={f"{b:g}": int(sum(r["admissible"][f"{b:g}"] for _, r in recs)) for b in LEVELS},
                 chart_admissible_rpm={f"{b:g}": [float(n) for n, r in recs if r["admissible"][f"{b:g}"]] for b in LEVELS},
                 chart_admissible_strict_speeds={f"{b:g}": int(sum(r["admissible_strict"][f"{b:g}"] for _, r in recs))
                                                 for b in LEVELS})
        if d in BAD:
            bro = np.array([r["beta_range"]["reduced_only"]["beta_mmps"] for _, r in recs])
            s["beta_range_reduced_only_mmps"] = dict(median=np.median(bro), min=bro.min())
            s["beta_range_reduced_only_at_least"] = {f"{b:g}": int(np.sum(bro >= b)) for b in LEVELS}
            s["beta_op_fast_only_below_5"] = int(sum(r["beta_op"]["fast_only_below_5"] for _, r in recs))
            s["entries_equal_stored"] = int(sum(r["entries_equal_stored"] for _, r in recs))
        summ[d] = s
    return summ


def print_summary(summ):
    print(f"{'design':9s} {'n':>2s} {'a_op mean/med/min':>18s} {'g mean':>7s} {'b_op med/min':>13s} {'>=1/2/5':>8s} "
          f"{'b_rng med/min':>13s} {'>=1/2/5':>8s} {'crit med/min':>14s} {'<1/2/5':>7s} {'bbar med/max':>12s} "
          f"{'step':>4s} {'face':>4s} {'adm 1/2/5':>9s}", flush=True)
    for d, s in summ.items():
        c = s["crit_kick_mmps"]
        print(f"{d:9s} {s['n_speeds']:2d} {s['a_op_mm']['mean']:6.2f}/{s['a_op_mm']['median']:5.2f}/{s['a_op_mm']['min']:5.2f} "
              f"{s['g_mean']:7.0f} {s['beta_op_mmps']['median']:6.2f}/{s['beta_op_mmps']['min']:6.2f} "
              f"{'/'.join(str(s['beta_op_at_least'][k]) for k in ('1', '2', '5')):>8s} "
              f"{s['beta_range_mmps']['median']:6.2f}/{s['beta_range_mmps']['min']:6.2f} "
              f"{'/'.join(str(s['beta_range_at_least'][k]) for k in ('1', '2', '5')):>8s} "
              f"{c['median'] if c['median'] is not None else float('inf'):7.2f}/"
              f"{c['min'] if c['min'] is not None else float('inf'):6.2f} "
              f"{'/'.join(str(s['crit_kick_below'][k]) for k in ('1', '2', '5')):>7s} "
              f"{s['bbar_V']['median']:5.2f}/{s['bbar_V']['max']:5.2f} {s['step_entry_stable']:4d} {s['face_entry_stable']:4d} "
              f"{'/'.join(str(s['chart_admissible_speeds'][k]) for k in ('1', '2', '5')):>9s}", flush=True)


def runtime_block(cache):
    """CPU seconds of the jobs this output uses, by kind (and per simulated run for the critical kicks)."""
    by = {}
    runs, tsim = 0, 0.0
    for kk, v in cache.d.items():
        k = json.loads(kk)
        r = by.setdefault(job_label(k), dict(jobs=0, cpu_s=0.0))
        r["jobs"] += 1
        r["cpu_s"] += v["t"]
        if k[0] in ("crit", "recheck_crit"):
            x = v["v"]
            if np.isnan(x):
                nr = 1
            elif np.isinf(x):
                nr = len(C1.DV_GRID)
            else:                                                # grid steps up to the first chatter, then bisections
                kgrid = int(np.searchsorted(C1.DV_GRID, x * (1 - 1e-9)))
                nr = 1 + kgrid + C1.N_BISECT
            runs += nr
            tsim += v["t"]
    for r in by.values():
        r["mean_s"] = r["cpu_s"] / r["jobs"]
    return dict(by_kind=by, cpu_s_total=sum(r["cpu_s"] for r in by.values()),
                crit_simulated_runs=runs, crit_s_per_run=(tsim / runs) if runs else None)


# ---------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--speeds", default="", help="comma-separated subset of the 21 speeds (rpm)")
    ap.add_argument("--designs", default="", help="comma-separated subset of " + ",".join(DESIGNS))
    ap.add_argument("--phases", default="", help="comma-separated subset of the kick phases 0,4,8,12")
    ap.add_argument("--out", default="fair_compare", help="output name: <out>.json and <out>_cache.jsonl here")
    ap.add_argument("--cache", default="", help="cache name (default: the output name): <cache>_cache.jsonl here; "
                    "jobs are keyed by their full definition, so runs with other options can share a cache")
    ap.add_argument("--smoke", action="store_true", help="same as --out fair_compare_smoke")
    ap.add_argument("--plan", action="store_true", help="list the jobs still to run and exit")
    ap.add_argument("--no-recheck", action="store_true", help="reuse stored results without recomputing two of them")
    ap.add_argument("--no-reuse", action="store_true", help="recompute stored critical kicks and scan rows too")
    ap.add_argument("--ba-tight", action="store_true",
                    help="also c2.TIGHT at ba design points whose stored default-order FAST kick is below 5 mm/s")
    a = ap.parse_args()
    speeds = tuple(float(x) for x in a.speeds.split(",") if x.strip()) or SPEEDS
    designs = tuple(x.strip() for x in a.designs.split(",") if x.strip()) or DESIGNS
    phases = tuple(int(x) for x in a.phases.split(",") if x.strip()) or PHASES
    if not set(speeds) <= set(SPEEDS):
        ap.error(f"speeds must be among {SPEEDS}")
    if not set(designs) <= set(DESIGNS):
        ap.error(f"designs must be among {DESIGNS}")
    if not set(phases) <= set(PHASES):
        ap.error(f"phases must be among {PHASES}")
    out_name = "fair_compare_smoke" if a.smoke else a.out
    partial = (set(speeds) != set(SPEEDS) or set(designs) != set(DESIGNS) or set(phases) != set(PHASES)
               or a.ba_tight or a.no_reuse)
    if partial and out_name == "fair_compare":
        ap.error("a run that differs from the campaign protocol (--speeds/--designs/--phases/--ba-tight/--no-reuse) "
                 "must write elsewhere: use --smoke or --out NAME")
    speeds = tuple(sorted(set(speeds)))
    designs = tuple(d for d in DESIGNS if d in designs)
    phases = tuple(p for p in PHASES if p in phases)
    out_path = os.path.join(HERE, f"{out_name}.json")
    cache = Cache(os.path.join(HERE, f"{a.cache or out_name}_cache.jsonl"))
    t0 = time.time()

    tab, fd, ba = load_inputs(speeds, designs)
    pts = points_of(tab)
    stored = load_stored()
    chk = stored_checks(ba, stored, tab, speeds, phases)
    print(f"{out_name}: {len(speeds)} speeds x {len(designs)} designs = {sum(z is not None for z in tab.values())} "
          f"design points, {len(pts)} unique (rpm, g, a_op); phases {list(phases)}; {len(cache.d)} jobs cached; "
          f"stored-protocol check {'ok' if chk['ok'] else 'FAILED'}", flush=True)
    if not chk["ok"]:
        print(json.dumps(chk, indent=1, default=float)[:4000], flush=True)

    # rechecks first: they decide whether stored results are reused
    rc = {} if (a.no_recheck or a.no_reuse) else recheck_plan(ba, stored, speeds, phases)
    rc_todo = [r["key"] for r in rc.values() if not cache.has(r["key"])]
    if a.plan:
        reuse_crit, reuse_scan = bool(not a.no_reuse and chk["ok"]), not a.no_reuse   # rechecks assumed to pass
        jobs, todo, reused = make_plan(pts, stored, phases, reuse_crit, reuse_scan, cache, a.ba_tight, ba)
        print_plan(jobs, todo, reused, rc_todo, a.workers)
        return
    if rc_todo:
        print(f"recheck of {len(rc_todo)} stored jobs", flush=True)
        failed = run_jobs(rc_todo, cache, a.workers)
        if failed:
            sys.exit("recheck failed to run; rerun to retry")
    rres = recheck_result(rc, cache)
    for kind, r in rres.items():
        print(f"recheck {kind}: {json.dumps(r, default=float)}", flush=True)
    reuse_crit = bool(not a.no_reuse and chk["ok"] and (rres.get("crit") or {}).get("equal", True))
    reuse_scan = bool(not a.no_reuse and (rres.get("cert") or {}).get("equal", True))
    if not a.no_reuse and not reuse_crit:
        print("WARNING: stored critical kicks are NOT reused (check or recheck failed)", flush=True)
    if not a.no_reuse and not reuse_scan:
        print("WARNING: basin-aware scan rows are NOT reused (certificate recheck failed)", flush=True)

    jobs, todo, reused = make_plan(pts, stored, phases, reuse_crit, reuse_scan, cache, a.ba_tight, ba)
    print_plan(jobs, todo, reused, [], a.workers)
    failed = run_jobs(todo, cache, a.workers, followup=tight_followup)
    if failed:
        print(json.dumps(failed, indent=1), flush=True)
        sys.exit(f"{len(failed)} jobs failed; rerun to retry them (finished jobs are cached)")
    # TIGHT follow-ups of FAST results that were cached before this run
    _, todo, _ = make_plan(pts, stored, phases, reuse_crit, reuse_scan, cache, a.ba_tight, ba)
    if todo:
        failed = run_jobs(todo, cache, a.workers, followup=tight_followup)
        if failed:
            sys.exit(f"{len(failed)} jobs failed; rerun to retry them")

    # assemble
    pm = {pt: point_metrics(pt, ds, cache, stored, phases, reuse_crit, reuse_scan) for pt, ds in pts.items()}
    per_speed = {}
    for n in speeds:
        row = per_speed.setdefault(f"{n:.0f}", {})
        for d in designs:
            z = tab[(n, d)]
            if z is None:
                row[d] = None
                continue
            pt = (n, z["g"], z["a_op"])
            row[d] = design_record(n, d, z, pm[pt], pts[pt], ba, cache, a.ba_tight)
    summ = summarize(per_speed, designs, speeds)
    recs = [(n, d, r) for n, row in per_speed.items() for d, r in row.items() if r]
    bad = [r for _, d, r in recs if d in BAD]
    checks = dict(
        stored_protocol=chk, recheck=rres,
        reuse=dict(crit_kicks=reuse_crit, scan_rows=reuse_scan, crit_kicks_reused=reused["crit"],
                   scan_rows_reused=reused["scan_rows"]),
        a_lim_input_vs_zoa_max_rel_diff=max(abs(r["a_lim_zoa_mm"] - r["a_lim_mm"]) / r["a_lim_mm"] for _, _, r in recs),
        bbar_vs_stored_ba_max_abs_V=max((abs(r["bbar_V"] - r["bbar_stored_V"]) for r in bad), default=None),
        entries_equal_stored_ba=dict(equal=int(sum(r["entries_equal_stored"] for r in bad)), of=len(bad)),
        ba_beta_op_fast_only_below_5=[dict(rpm=float(n), design=d, fast_mmps=r["beta_op"]["fast_mmps"])
                                      for n, d, r in recs if d in BAD and r["beta_op"]["fast_only_below_5"]],
        crit_unsettled=[dict(rpm=float(n), design=d) for n, d, r in recs if r["crit"]["worst_kind"] == "unsettled"])
    res = dict(
        note=__doc__,
        protocol=dict(zeta=ZETA, vmax_V=VMAX, rho=BA.RHO, vlim_V=VLIM, a_phys_mm=A_PHYS, depth_grid_mm=list(DEPTHS),
                      speeds_rpm=list(speeds), designs=list(designs), reference_designs=[d for d in REF if d in designs],
                      phases=list(phases), n_phase=C1.N_PHASE, kick_rev=C1.KICK_REV, window_rev=C1.WINDOW,
                      dv_grid_mps=C1.DV_GRID.tolist(), n_bisect=C1.N_BISECT, nreg=C1.NREG,
                      classifier="tdboundary.classify_run (D1)",
                      beta_op=dict(N="c2_certificate.nodes_for(rpm)", settings="FAST; TIGHT too below 5 mm/s, larger kept",
                                   ba_designs="stored (basin_aware.recert)" + (" + TIGHT below 5 mm/s" if a.ba_tight else "")),
                      beta_range=dict(N="basin_aware.n_red(rpm) = nodes_for(rpm) - 8", settings="FAST",
                                      depths="grid depths <= a_op and a_op", ba_designs="stored min_beta_up_to_a_rob_mmps"),
                      admissible="beta_range >= beta_req and bbar(a_op) <= 8.1 V and a_op < a_lim",
                      admissible_strict="also bbar <= 8.1 V at every grid depth below a_op and a_op <= 32 mm",
                      crit_kick="min over phases; unsettled (nan) = 0, inf = null with kind 'inf'",
                      levels_mmps=list(LEVELS), tight_below_mmps=TIGHT_BELOW * 1e3, scan_gain_tol=G_TOL,
                      basin_aware_cache=BA.PROTO, simulator_build=SOURCE_SHA[:12],
                      inputs=dict(fair_designs=dict(file=os.path.basename(FD_JSON), sha256_12=sha12(FD_JSON),
                                                    note=fd.get("note")),
                                  basin_aware=dict(file=os.path.basename(BA_JSON), sha256_12=sha12(BA_JSON)))),
        summary=summ, checks=checks, per_speed=per_speed,
        runtime=dict(runtime_block(cache), wall_s_this_run=time.time() - t0,
                     workers=a.workers))
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(clean(res), fh, indent=1)
    os.replace(tmp, out_path)
    print_summary(summ)
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
