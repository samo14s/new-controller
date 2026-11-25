"""Entry dependence of the basin-aware chart: the depth usable with a face entry and with a step entry.

Purpose.  The basin-aware chart (basin_aware.py) certifies the settled cut: its certified kick is a
deviation from the forced orbit of the cut at full depth, and the entry into the cut is not covered.
At its design points (n, g_rob(n; beta_req), a_rob(n; beta_req)) one simulated step entry from rest
chatters at 14, 12 and 7 of the 21 speeds for beta_req = 1, 2 and 5 mm/s, one face entry at none
(basin_aware.json summary, step_entry_chatters_at_design_rpm).  The referee asked the chart to show
the depth usable with each entry.  This script gives it per speed and beta_req, VERIFIED NUMERICALLY,
NOT CERTIFIED: one simulated entry per depth and entry kind, at the simulator's entry phase (tooth
angle 0 at t = 0), with the nominal parameters; nothing is claimed between the grid depths, for other
entry phases or for perturbed parameters.

Protocol (exact definitions; damping of every stored result: params.set_damping(gainmap_main.npz
zeta) = 1.34 %, set at import and again in every worker before any src module is used).
  designs   speeds n = 500:125:3000 rpm (21); beta_req = 1, 2, 5 mm/s; the basin-aware gain
            g = g_rob(n; beta_req) and depth a_rob(n; beta_req) of basin_aware.json
            per_speed['<n>']['designs']['1' | '2' | '5'] (keys g, a_rob_mm), unchanged.
  runs      at every grid depth d = 2, 4, ..., a_rob mm (the chart's grid), upwards, one step entry
            and one face entry: basin_aware.kick_job(('entry', n, d * 1e-3, g, 'step' | 'face')), i.e.
            tdboundary.classify(n, a, g, vmax = 9 V, nreg = 0) with edge = True for the face entry:
              step   the cut at full depth and feed from t = 0, the workpiece at rest on a fresh
                     surface; the 300 revolutions from t = 0 are judged;
              face   straight entry through a workpiece face normal to the feed (full radial
                     engagement after R / fz teeth, 40 revolutions); the 300 revolutions after the
                     entry are judged (a cut that chatters during the entry stays chatter);
            the 9 V clip on the command, the full surface memory (the chip against the deepest
            earlier pass), classifier D1 (tdboundary.classify_run).  Recorded per run: stable,
            max_x_mm (peak workpiece displacement, the run aborts at 1 cm) and max_v_V (peak |V|).
            Every grid depth up to a_rob is run, also above a first failure: with the clip the
            outcome need not be monotone in depth, and every failing depth is listed.
  depths    a_face(n; beta_req) = the largest grid depth d <= a_rob such that the face entry settles
            at every grid depth 2, 4, ..., d (0 if it chatters at 2 mm); a_step likewise for the step
            entry.  So a_face = a_rob when no face entry up to a_rob chatters.
  quiet     secondary indicator, not part of the definitions above: D1 judges the 300 revolutions,
            not the transient, and a step entry can throw the workpiece by millimetres (the tool
            leaves the cut) and still settle (stored: 7.3 mm at 1000 rpm, 26 mm, g 2000).  Settled
            runs with max_x > MAX_X_QUIET = 1 mm (20 fz; settled cuts stay below 0.06 mm) are listed,
            and a_face_quiet, a_step_quiet are the largest grid depths up to which every entry settles
            with max_x <= 1 mm.
  dedup     runs are keyed by (n, a, g, entry) (the key of basin_aware's validate.jsonl): a gain shared
            by several beta_req at one speed is run once per depth.
  check     at the design points (d = a_rob) the runs are the jobs of basin_aware's validate stage
            (same key, same simulator): they must reproduce basin_aware.json entries_9V exactly
            (stable, max_x_mm, max_v_V); the comparison is reported.
  summary   per beta_req, over the 21 speeds: the means of a_rob, a_face and a_step; the numbers (and
            lists) of speeds with a_face < a_rob and with a_step < a_rob; the step and face entries
            that chatter at the design point; the speeds whose failing depths are not a contiguous
            top band (a stable depth above a failing one); the quiet depths and the settled runs with
            a transient above 1 mm.

Output ba_entry.json: note (this docstring), protocol, per_speed['<n>']['<beta_req>'] = g, a_rob_mm,
a_face_mm, a_step_mm, failing_depths_mm {step, face}, runs {step, face: [d_mm, stable, max_x_mm,
max_v_V, t_s]}, design-point check; chart (arrays against speed for the chart: a_rob, a_face,
a_step per beta_req); summary; runtime.

Run (from ozsoy2026/; resumable: every finished run is appended to ba_entry_cache.jsonl as it
arrives and a restart skips the runs found there; 838 runs (1210 before deduplication), about 1 s
each, so about 14 CPU-min, 7 min wall with 2 workers):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup python3 verification/paper/ba_entry.py \\
        --workers 2 > verification/paper/ba_entry.log 2>&1 &
  --plan      count the runs still to do, with a CPU-time estimate, and exit (simulates nothing)
  --speeds, --betas, --depths   subsets (tests); a reduced depth list makes a_face and a_step the
              largest TESTED depth up to which every tested depth settles (flagged 'partial')
  --out NAME  write NAME.json and NAME_cache.jsonl here; --smoke = --out ba_entry_smoke.
  Smoke test (10 runs, about 10 s, one worker; delete ba_entry_smoke.json and ba_entry_smoke_cache.jsonl):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/ba_entry.py --smoke \\
        --speeds 1000 --betas 1 --depths 2,8,16,24,32 --workers 1
  (measured: 0.82-1.08 s per run, 0.09 s for a run that reaches the 1 cm abort; the two design-point
  runs reproduce basin_aware.json entries_9V exactly.)
"""
import argparse
import json
import os
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
import basin_aware as BA                              # noqa: E402  (imports no src module at load)

BA_JSON = os.path.join(HERE, "basin_aware.json")
SPEEDS = tuple(float(n) for n in BA.GRID_RPM)         # 500:125:3000 rpm
BETAS = ("1", "2", "5")                               # beta_req labels of basin_aware.json (mm/s)
KINDS = ("step", "face")
GRID = tuple(float(d) for d in BA.DEPTHS)             # 2, 4, ..., 32 mm
T_ENTRY = 1.0         # s per entry run: 1.13 s mean over the 90 entry runs of basin_aware's validate.jsonl
#                       (2 workers), 0.86 s in the smoke test (1 worker); a run reaching the 1 cm abort is cheap
MAX_X_QUIET = 1.0     # mm: a settled run whose transient exceeds it is listed (secondary indicator)


# ------------------------------------------------------------------ utilities ----
class Cache:
    """Append-only JSON-lines cache (as basin_aware.JL, with an explicit path): key (a JSON list) ->
    value.  An unfinished last line (an interrupted write) is dropped and the file truncated after the
    last complete line.  Only the main process writes."""

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

    def put(self, k, v):
        self.d[json.dumps(k)] = v
        with open(self.path, "a") as fh:
            fh.write(json.dumps(dict(k=k, v=v), default=float) + "\n")
            fh.flush()
        return v


def entry_key(rpm, g, d, kind):
    """The job key of basin_aware.kick_job and of its validate.jsonl: ['entry', rpm, a (m), g, kind]."""
    return ["entry", float(rpm), float(d) * 1e-3, float(g), kind]


def _init():
    P.set_damping(ZETA)


def run_entry(key):
    """One entry run -> (key, value or None, wall seconds, error or None)."""
    P.set_damping(ZETA)
    t = time.time()
    try:
        k, v, _ = BA.kick_job(tuple(key))
        return list(k), v, round(time.time() - t, 3), None
    except Exception as e:                                       # reported, not cached: a restart retries it
        return list(key), None, round(time.time() - t, 3), repr(e)


def clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(clean(obj), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------- designs ----
SOURCE = "chart"                                     # 'chart': basin_aware.json; 'kappa': ba_kappa.json (a <= 0.9 a_lim)
BAK_JSON = os.path.join(HERE, "ba_kappa.json")


def load_designs(speeds, betas, depth_subset):
    """(speed, beta) -> design and its depth list, from basin_aware.json; checks its protocol."""
    if SOURCE == "kappa":                           # the chart with the depth margin of ba_kappa.py
        bak = json.load(open(BAK_JSON))
        out = {}
        for n in speeds:
            for b in betas:
                dz = bak["speeds"][f"{n:.0f}"]["designs"][str(float(b) * 1e-3)]
                g, a_rob = dz.get("g"), float(dz.get("a_rob") or 0.0)
                ds = [] if g is None else [d for d in GRID if d <= a_rob + 1e-9
                                             and (depth_subset is None or d in depth_subset)]
                out[(n, b)] = dict(g=None if g is None else float(g), a_rob=a_rob, depths=ds, stored_entries=None)
        return out
    ba = json.load(open(BA_JSON))
    pr = ba["protocol"]
    assert pr["cache"] == BA.PROTO, f"basin_aware.json protocol {pr['cache']} != basin_aware.PROTO {BA.PROTO}"
    assert abs(pr["zeta"] - ZETA) < 1e-15 and pr["vmax_V"] == 9.0 and list(pr["depth_grid_mm"]) == list(GRID)
    out = {}
    for n in speeds:
        ps = ba["per_speed"][f"{n:.0f}"]
        for b in betas:
            dz = ps["designs"][b]
            g, a_rob = dz.get("g"), float(dz.get("a_rob_mm", 0.0))
            ds = [] if g is None else [d for d in GRID if d <= a_rob + 1e-9
                                         and (depth_subset is None or d in depth_subset)]
            out[(n, b)] = dict(g=None if g is None else float(g), a_rob=a_rob, depths=ds,
                               stored_entries=dz.get("entries_9V"))
    return out


def jobs_of(designs):
    keys = []
    for (n, _), dz in designs.items():
        for d in dz["depths"]:
            keys += [entry_key(n, dz["g"], d, k) for k in KINDS]
    return [json.loads(s) for s in dict.fromkeys(json.dumps(k) for k in keys)]


# ------------------------------------------------------------------ assemble ----
def largest_settled(runs, quiet=False):
    """Largest depth up to which every listed run settles (0 if the first chatters); quiet: and keeps
    its peak displacement at or below MAX_X_QUIET."""
    a = 0.0
    for r in runs:
        if not r["stable"] or (quiet and r["max_x_mm"] > MAX_X_QUIET):
            break
        a = r["d_mm"]
    return a


def assemble(designs, cache, args, partial, t_run):
    per, chart = {}, dict(rpm=[], a_rob_mm={}, a_face_mm={}, a_step_mm={}, a_face_quiet_mm={}, a_step_quiet_mm={}, g={})
    repro = dict(points=0, identical=0, differ=[])
    for (n, b), dz in sorted(designs.items()):
        rec = dict(g=dz["g"], a_rob_mm=dz["a_rob"], depths_tested_mm=dz["depths"], runs={}, failing_depths_mm={})
        complete = True
        for kind in KINDS:
            runs = []
            for d in dz["depths"]:
                c = cache.get(entry_key(n, dz["g"], d, kind))
                if c is None:
                    complete = False
                    continue
                v = c["v"]
                runs.append(dict(d_mm=d, stable=bool(v["stable"]), max_x_mm=float(v["max_x_mm"]),
                                 max_v_V=float(v["max_v_V"]), t_s=c["t"]))
            rec["runs"][kind] = runs
            rec["failing_depths_mm"][kind] = [r["d_mm"] for r in runs if not r["stable"]]
            a_k = largest_settled(runs) if complete else None
            rec[f"a_{kind}_mm"] = a_k
            rec[f"a_{kind}_quiet_mm"] = largest_settled(runs, quiet=True) if complete else None
            rec.setdefault("settled_large_transient_depths_mm", {})[kind] = [
                r["d_mm"] for r in runs if r["stable"] and r["max_x_mm"] > MAX_X_QUIET]
            fail = rec["failing_depths_mm"][kind]
            # a failing depth below a settled one: the failing depths are not one band at the top
            rec[f"{kind}_nonmonotone"] = bool(fail and any(r["stable"] and r["d_mm"] > min(fail) for r in runs))
        rec["complete"] = complete
        # design-point check against basin_aware.json entries_9V (same job key, same simulator)
        st = dz["stored_entries"]
        if st and dz["a_rob"] in dz["depths"] and complete:
            chk = {}
            for kind in KINDS:
                mine = next(r for r in rec["runs"][kind] if r["d_mm"] == dz["a_rob"])
                s = st[kind]
                same = (bool(s["stable"]) == mine["stable"] and float(s["max_x_mm"]) == mine["max_x_mm"]
                        and float(s["max_v_V"]) == mine["max_v_V"])
                chk[kind] = dict(stored=s, identical=bool(same))
                repro["points"] += 1
                if same:
                    repro["identical"] += 1
                else:
                    repro["differ"].append(dict(rpm=n, beta_req=b, kind=kind, stored=s,
                                                here={k: mine[k] for k in ("stable", "max_x_mm", "max_v_V")}))
            rec["design_point_vs_basin_aware_json"] = chk
        per.setdefault(f"{n:.0f}", dict(rpm=n))[b] = rec
    speeds = sorted({n for (n, _) in designs})
    betas = sorted({b for (_, b) in designs}, key=float)
    chart["rpm"] = speeds
    summ = {}
    for b in betas:
        rows = [per[f"{n:.0f}"][b] for n in speeds]
        chart["g"][b] = [r["g"] for r in rows]
        chart["a_rob_mm"][b] = [r["a_rob_mm"] for r in rows]
        chart["a_face_mm"][b] = [r["a_face_mm"] for r in rows]
        chart["a_step_mm"][b] = [r["a_step_mm"] for r in rows]
        chart["a_face_quiet_mm"][b] = [r["a_face_quiet_mm"] for r in rows]
        chart["a_step_quiet_mm"][b] = [r["a_step_quiet_mm"] for r in rows]
        done = [r for r in rows if r["complete"]]
        if not done:
            continue
        ar = np.array([r["a_rob_mm"] for r in done])
        af = np.array([r["a_face_mm"] for r in done])
        ast = np.array([r["a_step_mm"] for r in done])
        sp = [n for n in speeds if per[f"{n:.0f}"][b]["complete"]]
        summ[b] = dict(
            beta_req_mmps=float(b), n_speeds=len(done),
            mean_a_rob_mm=float(ar.mean()), mean_a_face_mm=float(af.mean()), mean_a_step_mm=float(ast.mean()),
            min_a_face_mm=float(af.min()), min_a_step_mm=float(ast.min()),
            n_a_face_below_a_rob=int(np.sum(af < ar)), n_a_step_below_a_rob=int(np.sum(ast < ar)),
            speeds_a_face_below_a_rob=[n for n, x, y in zip(sp, af, ar) if x < y],
            speeds_a_step_below_a_rob=[n for n, x, y in zip(sp, ast, ar) if x < y],
            mean_a_step_over_a_rob=float(np.mean(ast / ar)), mean_a_face_over_a_rob=float(np.mean(af / ar)),
            step_chatters_at_design_point_rpm=[n for n, r in zip(sp, done) if r["a_rob_mm"] in r["failing_depths_mm"]["step"]],
            face_chatters_at_design_point_rpm=[n for n, r in zip(sp, done) if r["a_rob_mm"] in r["failing_depths_mm"]["face"]],
            step_nonmonotone_rpm=[n for n, r in zip(sp, done) if r["step_nonmonotone"]],
            face_nonmonotone_rpm=[n for n, r in zip(sp, done) if r["face_nonmonotone"]],
            runs=int(sum(len(r["runs"][k]) for r in done for k in KINDS)),
            quiet=dict(max_x_mm=MAX_X_QUIET, note="secondary indicator (see the docstring), not a_face / a_step",
                       mean_a_face_quiet_mm=float(np.mean([r["a_face_quiet_mm"] for r in done])),
                       mean_a_step_quiet_mm=float(np.mean([r["a_step_quiet_mm"] for r in done])),
                       settled_runs_above={k: int(sum(len(r["settled_large_transient_depths_mm"][k]) for r in done))
                                           for k in KINDS},
                       settled_above_rpm={k: [n for n, r in zip(sp, done) if r["settled_large_transient_depths_mm"][k]]
                                          for k in KINDS}))
    keys = jobs_of(designs)
    ts = [cache.get(k)["t"] for k in keys if cache.get(k) is not None]
    out = dict(
        note=__doc__,
        protocol=dict(zeta=ZETA, vmax_V=9.0, nreg=0, classifier="D1 (tdboundary.classify_run)", window_rev=300,
                      entries=dict(step="tdboundary.classify(n, a, g, vmax=9, nreg=0): full depth from rest at t = 0",
                                   face="tdboundary.classify(n, a, g, vmax=9, nreg=0, edge=True): 40-revolution face entry"),
                      job="basin_aware.kick_job(('entry', rpm, a, g, kind))", depth_grid_mm=list(GRID),
                      designs="basin_aware.json per_speed[n].designs[beta_req]: g, a_rob_mm",
                      basin_aware_cache_id=BA.PROTO, speeds=speeds, beta_req_mmps=[float(b) for b in betas],
                      depth_subset_mm=args.depths or None, partial=bool(partial),
                      status="verified numerically (one simulated run per depth and entry), not certified"),
        per_speed=per, chart=chart, summary=summ,
        design_point_reproduction=dict(repro, note="runs at d = a_rob against basin_aware.json entries_9V "
                                                   "(same job key, same simulator): exact equality asked"),
        runtime=dict(unique_runs=len(keys), runs_done=len(ts), run_s_mean=float(np.mean(ts)) if ts else None,
                     run_s_total=float(np.sum(ts)) if ts else 0.0, wall_s_this_call=round(t_run, 1)))
    return out


# ---------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--speeds", default="", help="comma-separated subset of the 21 speeds (rpm)")
    ap.add_argument("--betas", default="", help="comma-separated subset of the beta_req labels 1,2,5 (mm/s)")
    ap.add_argument("--depths", default="", help="comma-separated subset of the grid depths (mm), tests only")
    ap.add_argument("--out", default="ba_entry", help="output name: <out>.json and <out>_cache.jsonl here")
    ap.add_argument("--smoke", action="store_true", help="same as --out ba_entry_smoke")
    ap.add_argument("--plan", action="store_true", help="count the runs still to do and exit")
    ap.add_argument("--source", choices=("chart", "kappa"), default="chart",
                    help="designs of basin_aware.json (chart) or of ba_kappa.json (kappa, a <= 0.9 a_lim)")
    ap.add_argument("--cache", default=None, help="cache name here (default <out>_cache.jsonl); entries are keyed "
                                                    "by (rpm, a, g, kind), so variants can share one cache")
    a = ap.parse_args()
    out_name = "ba_entry_smoke" if a.smoke else a.out
    speeds = tuple(float(x) for x in a.speeds.split(",") if x.strip()) or SPEEDS
    betas = tuple(x.strip() for x in a.betas.split(",") if x.strip()) or BETAS
    dsub = tuple(float(x) for x in a.depths.split(",") if x.strip()) or None
    assert all(n in SPEEDS for n in speeds) and all(b in BETAS for b in betas)
    assert dsub is None or all(d in GRID for d in dsub), "depths must be grid depths"
    partial = speeds != SPEEDS or betas != BETAS or dsub is not None
    if partial and out_name == "ba_entry" and not a.plan:
        sys.exit("a subset run must not write ba_entry.json: use --smoke or --out NAME")
    a.depths = list(dsub) if dsub else None
    out_path = os.path.join(HERE, out_name + ".json")
    global SOURCE
    SOURCE = a.source
    if SOURCE != "chart" and out_name == "ba_entry":
        sys.exit("the kappa variant must not write ba_entry.json: use --out NAME")
    cache = Cache(os.path.join(HERE, a.cache if a.cache else out_name + "_cache.jsonl"))
    designs = load_designs(speeds, betas, dsub)
    keys = jobs_of(designs)
    todo = [k for k in keys if cache.get(k) is None]
    n_all = sum(len(dz["depths"]) * len(KINDS) for dz in designs.values())
    print(f"ba_entry: {len(designs)} design points, {n_all} runs before and {len(keys)} after deduplication, "
          f"{len(keys) - len(todo)} cached, {len(todo)} to run (estimate {len(todo) * T_ENTRY / 60:.1f} CPU-min "
          f"at {T_ENTRY} s per run)", flush=True)
    if a.plan:
        return
    t0 = time.time()
    failed = []
    if todo:
        # longest first is irrelevant here (all runs take about a second); deep cuts first anyway
        todo.sort(key=lambda k: (-k[2], k[1], k[3], k[4]))
        with Pool(a.workers, initializer=_init) as pool:
            for i, (k, v, t, err) in enumerate(pool.imap_unordered(run_entry, todo)):
                if err is not None:
                    failed.append(dict(key=k, error=err))
                    print(f"  FAILED {k}: {err}", flush=True)
                    continue
                cache.put(k, dict(v=v, t=t))
                print(f"  [{i + 1}/{len(todo)}] {k[4]} entry {k[1]:.0f} rpm a {k[2] * 1e3:.0f} mm g {k[3]:.1f}: "
                      f"stable {v['stable']}, max_x {v['max_x_mm']:.3f} mm, max_v {v['max_v_V']:.2f} V ({t:.2f} s)",
                      flush=True)
    out = assemble(designs, cache, a, partial, time.time() - t0)
    out["runtime"]["failed"] = failed
    write_json(out_path, out)
    for b, s in out["summary"].items():
        print(f"beta_req {b} mm/s ({s['n_speeds']} speeds): mean a_rob {s['mean_a_rob_mm']:.2f}, a_face "
              f"{s['mean_a_face_mm']:.2f}, a_step {s['mean_a_step_mm']:.2f} mm; a_face < a_rob at "
              f"{s['n_a_face_below_a_rob']}, a_step < a_rob at {s['n_a_step_below_a_rob']} speeds", flush=True)
    r = out["design_point_reproduction"]
    print(f"design points against basin_aware.json entries_9V: {r['identical']}/{r['points']} identical", flush=True)
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
