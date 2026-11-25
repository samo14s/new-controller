"""Sensitivity of the basin-aware chart to the assumed axial-depth cap a_cap (10, 16, 32 mm).

Purpose.  basin_aware.py admits a grid depth only up to a_phys = 32 mm = 2D, "twice the cutter
diameter".  The referee noted that 32 mm is not shown to be physically available (tool stick-out,
flute length, part).  It is renamed a_cap, an ASSUMED cap, and the chart is recomputed for
a_cap = 10, 16 and 32 mm.  For a_cap = 32 mm the result must reproduce basin_aware.json's designs;
every difference is reported.  The reference designs of fair_designs.json are capped the same way.

Definitions (exactly those of basin_aware.py with a_phys replaced by a_cap).  A grid depth
a in {2, 4, ..., a_cap} mm is admissible for (n, g, beta_req) if
  (L) a < a_lim(n, g)                 zoa.a_lim;
  (V) bbar(n, a, g) <= 8.1 V          basin_aware.bbar_of (c3_periodic.orbit, rigorous bound);
  (C) beta(n, a, g) >= beta_req       basin_aware.certificate (c2_certificate.certify_kick, e = unit
                                      velocity, delta = 0.5, 9 V clip, rescaled to the rigorous bound).
a_rob_cap(n, g; beta_req) = the largest grid depth <= a_cap such that every grid depth from 2 mm up to
it is admissible (0 if 2 mm is not); a_rob_cap(n; beta_req) = max over the candidate gains, g* the
maximiser (ties: the larger smallest certified kick over the depths up to a_rob, then the smaller
gain; basin_aware.key).  beta_req = 1, 2, 5 mm/s.  binding = the constraints failing at a_rob + 2 mm
(linear limit, voltage margin, certificate), 'a_cap' when a_rob = a_cap (basin_aware.binding's
'a_phys', renamed).

Protocol (damping of every stored result: params.set_damping(gainmap_main.npz zeta) = 1.34 %, set at
import and again in every worker before any src module is used).
  speeds    the 21 speeds 500:125:3000 rpm of the chart (basin_aware.GRID_RPM).
  gains     per speed and cap: g_auth (the authority-only gain, compare record of the basin_aware
            cache, always a candidate as in basin_aware.search), EVERY coarse gain 0:250:3000 V s/m
            (basin_aware.search skipped coarse gains by branch and bound: a gain whose closed-form
            bound a_cheap could not exceed the best a_rob so far; exact for the maximum, but ties
            with the best were not explored, and the tie-break can prefer them), then for
            beta_req = 1, 2, 5 mm/s in that order, +-125 and then +-62.5 V s/m around the best gain so
            far for that cap, each skipped when min(a_cheap, a_cap) cannot exceed that beta_req's best
            (basin_aware.search's refinement rule; a_cheap is prefix-defined, so min(a_cheap, a_cap) is
            exactly the capped bound).  The candidate sets of the caps are separate.
  scan      one gain: depth rows upward from 2 mm until a depth is not admissible for 1 mm/s or a_cap
            is reached, reduced order N_red = nodes_for(n) - 8, c2.FAST (basin_aware.scan_gain).  The
            rows of basin_aware_cache/050bc4a99e/speed_<n>.jsonl ['scan', n, g] are reused where the
            gain was scanned (truncated at a_cap: a row does not depend on the cap); otherwise the rows
            are computed here by the same code path (basin_aware.bbar_of, basin_aware.certificate,
            zoa.a_lim) and cached row by row, so a larger cap extends a scan.  The coarse gains are
            scanned up to the largest cap first.  Gains are keyed by round(g, 6) as in basin_aware.
  recert    basin_aware.recert itself, re-certifying with the default order N = nodes_for(n): the
            design point, every lower grid depth whose reduced-order kick is below 1.15 beta_req, and
            (information only) the failing depth; c2.FAST, and c2.TIGHT where FAST < beta_req, the
            larger kept; a_rob lowered below a failing check, and the next candidate re-certified when
            g* falls below its reduced-order a_rob.  basin_aware's evaluate, binding and recert read
            the module constants A_PHYS and DEPTHS at call time: they are called with A_PHYS = a_cap
            and DEPTHS = the grid up to a_cap (set and restored around each call in the worker
            process; no file is changed), so the rules are basin_aware's with a_phys -> a_cap.
            Default-order certificates (['ref', n, g, d, FAST | TIGHT]) do not depend on the cap: they
            are taken from the basin_aware cache where they exist and computed and cached here
            otherwise.  The caps are processed in the order 32, 16, 10 mm (a TIGHT certificate
            computed for one cap is then used by the later ones, as basin_aware.recert uses its cache).
  check     a_cap = 32 mm: g*, a_rob and binding per speed and beta_req against basin_aware.json
            per_speed designs; each difference is listed with the facts that explain it (whether the
            gain chosen here was evaluated or skipped by basin_aware, the selection keys).
            --recheck K (default 1): per speed, the K cheapest stored reduced-order rows and the
            cheapest stored default-order c2.FAST certificate are recomputed and compared (the reuse
            of stored rows assumes the same code gives the same numbers).
  references  fair_designs.json speeds['<n>'][pub | auth0 | auth10 | auth1500]['a_chk'] capped:
            min(a_chk, a_cap).

Output ba_cap.json: note (this docstring), protocol, per_cap['<a_cap>']['<n>']['<beta_req>'] = g,
a_rob_mm, binding and the details of basin_aware.recert (reduced-order a_rob, correction, smallest
kick, default-order kicks, failing depth, tied gains, candidates re-certified); per_speed (g_auth,
gains scanned here, per cap the evaluated gains, skipped refinement gains, the authority gain under
the capped criteria, rechecks, certificates computed and their time); references; reproduction_cap32;
summary: per cap and beta_req the mean (min, max) a_rob over the speeds, the number of speeds at the
cap, the binding counts, the mean g*, and per cap the means of the capped reference depths and of
a_rob minus each of them.

Run (from ozsoy2026/; resumable: every certificate and scan row is appended to ba_cap_cache.jsonl as
it is computed, by the worker of its speed under a lock, and a restart recomputes nothing found there):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup python3 verification/paper/ba_cap.py \\
        --workers 2 > verification/paper/ba_cap.log 2>&1 &
  --plan      list per speed the coarse scans still to do with a CPU-time estimate and exit
  --speeds, --caps   subsets (tests: a subset must write elsewhere, --smoke or --out)
  --out NAME  write NAME.json and NAME_cache.jsonl here; --smoke = --out ba_cap_smoke.
  Full campaign: --plan gives 2.2 CPU-h as an upper bound (coarse rows bounded by a_cheap + 2 mm at the
  cache-mean certificate times, 4 default-order certificates per speed and cap below 32 mm); the
  500-875 rpm speeds (N_red 20-24, N 28-32) are about two thirds of it.  About 1.1 h wall with 2 workers.
  Smoke test (one speed, cap 10; delete ba_cap_smoke.json and ba_cap_smoke_cache.jsonl afterwards):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/ba_cap.py --smoke \\
        --speeds 2750 --caps 10 --workers 1
  (measured: 34 s; 20 new reduced-order certificates at N = 12, 1.24 s each, one default-order
  certificate at N = 20, 3.6 s; both rechecks reproduce the stored values exactly.)  Reproduction check
  at a speed whose 13 coarse gains are all in the basin_aware cache (no new certificate, under 1 s):
    ... ba_cap.py --out ba_cap_smoke32 --speeds 2500 --caps 32 --recheck 0 --workers 1
"""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

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
import c2_certificate as C2                           # noqa: E402
from zoa import a_lim as zoa_a_lim                    # noqa: E402

BA_JSON = os.path.join(HERE, "basin_aware.json")
FD_JSON = os.path.join(HERE, "fair_designs.json")
BA_CACHE = BA.CACHE                                   # basin_aware_cache/<protocol id>, read only here
SPEEDS = tuple(float(n) for n in BA.GRID_RPM)         # 500:125:3000 rpm
CAPS = (10.0, 16.0, 32.0)                             # mm
DEPTHS_ALL = tuple(BA.DEPTHS)                         # 2, 4, ..., 32 mm (taken before any cap is applied)
A_PHYS_STORED = float(BA.A_PHYS)                      # 32 mm
BETA_REQ = tuple(BA.BETA_REQ)                         # m/s
G_COARSE = tuple(BA.G_COARSE)                         # 0:250:3000 V s/m
REFINE, G_MAX, STEP = tuple(BA.REFINE), float(BA.G_MAX), float(BA.STEP)
REFS = ("pub", "auth0", "auth10", "auth1500")
RECHECK_RTOL = 1e-6
# cost model (--plan and the order of the speeds): s per c2.FAST certificate by order N, means of the
# basin_aware cache (same code, 2 workers); the smoke test measures one order again
CERT_S = dict(reduced={12: 1.43, 16: 3.23, 20: 7.93, 24: 13.41}, default={20: 8.82, 24: 18.48, 28: 65.86, 32: 54.14})
REF_PER_CAP = 4.0     # default-order certificates per speed for a cap below 32 mm (design points shared by
#                       the beta_req, lower depths below 1.15 beta_req, a TIGHT now and then); see the smoke test

LOCK = None           # set in the workers: appends to the shared cache file are serialised


def lab(b):
    return f"{b * 1e3:.0f}"


def rename(bind):
    return ["a_cap" if x == "a_phys" else x for x in bind]


def rel(x, y):
    x, y = float(x), float(y)
    return 0.0 if x == y else abs(x - y) / max(abs(x), abs(y))


# ------------------------------------------------------------------ caches ----
def read_records(path, rpm=None):
    """Read-only view of a JSON-lines cache: {json key: value}; only the records of one speed if rpm
    is given (key[1]); a line cut short (an interrupted or concurrent write) ends the reading."""
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if not line.endswith("\n"):
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break
                if rpm is None or float(rec["k"][1]) == rpm:
                    out[json.dumps(rec["k"])] = rec["v"]
    return out


def repair(path):
    """Truncate the cache after its last complete line (main process, before the workers start)."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        data = fh.read()
    good = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        good += len(line)
    if len(data) > good:
        with open(path, "r+b") as fh:
            fh.truncate(good)


class Store:
    """Records of one speed: the basin_aware cache (read only) and this script's cache (append-only;
    the worker of a speed is the only writer of that speed's records)."""

    def __init__(self, rpm, path):
        self.rpm, self.path = float(rpm), path
        self.old = read_records(os.path.join(BA_CACHE, f"speed_{rpm:.0f}.jsonl"))
        self.old_scans = {float(json.loads(s)[2]): v for s, v in self.old.items() if json.loads(s)[0] == "scan"}
        self.new = read_records(path, self.rpm)
        self.cost = self.zero()                                  # computed in this call

    @staticmethod
    def zero():
        return dict(reduced_n=0, reduced_s=0.0, fast_n=0, fast_s=0.0, tight_n=0, tight_s=0.0, cheap_n=0,
                    recheck_n=0, recheck_s=0.0)

    @staticmethod
    def tally(c, k, v):
        """Add one computed record to the cost counts c (certificate times as recorded by basin_aware.certificate)."""
        if k[0] == "row" and "t" in v:
            c["reduced_n"] += 1
            c["reduced_s"] += v["t"]
        elif k[0] == "ref":
            s = "fast" if v["settings"] == "FAST" else "tight"
            c[s + "_n"] += 1
            c[s + "_s"] += v["t"]
        elif k[0] == "cheap":
            c["cheap_n"] += 1
        elif k[0].startswith("recheck"):
            c["recheck_n"] += 1
            c["recheck_s"] += v["t"]

    def cost_total(self):
        """Cost of every record of this speed in this script's cache (all calls, so also after a restart)."""
        c = self.zero()
        for s, v in self.new.items():
            self.tally(c, json.loads(s), v)
        return c

    def get(self, k):
        s = json.dumps(k)
        v = self.new.get(s)
        return self.old.get(s) if v is None else v

    def put(self, k, v):
        assert float(k[1]) == self.rpm
        self.new[json.dumps(k)] = v
        line = json.dumps(dict(k=k, v=v), default=float) + "\n"
        with (LOCK if LOCK is not None else contextlib.nullcontext()):
            with open(self.path, "a") as fh:
                fh.write(line)
                fh.flush()
        self.tally(self.cost, k, v)
        return v


# ------------------------------------------------------- basin_aware with a cap ----
@contextlib.contextmanager
def capped(cap):
    """basin_aware's rules with a_phys replaced by a_cap: its evaluate, binding and recert read the module
    constants A_PHYS and DEPTHS at call time; they are set here and restored on exit (this process only)."""
    assert BA.A_PHYS == A_PHYS_STORED and tuple(BA.DEPTHS) == DEPTHS_ALL, "capped() is not re-entrant"
    assert float(cap) in DEPTHS_ALL, "a_cap must be a grid depth"
    BA.A_PHYS, BA.DEPTHS = float(cap), tuple(d for d in DEPTHS_ALL if d <= cap)
    try:
        yield
    finally:
        BA.A_PHYS, BA.DEPTHS = A_PHYS_STORED, DEPTHS_ALL


def scan_row(rpm, g, d, al, N):
    """One depth row of basin_aware.scan_gain (the same calls, in the same order)."""
    a = d * 1e-3
    lin = bool(d < al)
    bb = BA.bbar_of(rpm, g, a)
    row = dict(d=d, lin=lin, bbar=bb, volt=bool(bb <= BA.VLIM))
    if lin and bb < BA.VMAX:
        row.update(BA.certificate(rpm, g, a, N, "FAST", bb))
    return row


def scan_capped(rpm, g, cap, st):
    """basin_aware.scan_gain up to a_cap: the stored scan truncated at a_cap where it exists, else the
    rows computed (and cached) here, upward until a row is not admissible for 1 mm/s or d = a_cap."""
    g = float(round(g, 6))
    old = st.old_scans.get(g)
    if old is not None:
        return dict(rpm=rpm, g=g, a_lim=old["a_lim"], N=old["N"], rows=[r for r in old["rows"] if r["d"] <= cap],
                    source="basin_aware cache")
    al = float(zoa_a_lim(rpm, g)) * 1e3
    N = BA.n_red(rpm)
    rows = []
    for d in DEPTHS_ALL:
        if d > cap:
            break
        k = ["row", rpm, g, d]
        row = st.get(k)
        if row is None:
            row = st.put(k, scan_row(rpm, g, d, al, N))
        rows.append(row)
        if not (row["lin"] and row["volt"] and row["beta"] >= min(BETA_REQ)):
            break
    return dict(rpm=rpm, g=g, a_lim=al, N=N, rows=rows, source="computed here")


def cheap_of(rpm, g, st):
    """basin_aware.a_cheap (uncapped; the basin_aware cache where it has it)."""
    g = float(round(g, 6))
    k = ["cheap", rpm, g]
    v = st.get(k)
    if v is None:
        assert BA.A_PHYS == A_PHYS_STORED and tuple(BA.DEPTHS) == DEPTHS_ALL
        v = st.put(k, BA.a_cheap(rpm, g))
    return float(v)


def search_capped(rpm, cap, g_auth, st):
    """basin_aware.search for one cap, with every coarse gain evaluated (no branch and bound on them);
    the refinement as basin_aware.search, with the capped bound min(a_cheap, a_cap)."""
    evals = {}

    def ev(g):
        g = float(round(g, 6))
        if g not in evals:
            evals[g] = scan_capped(rpm, g, cap, st)
        return evals[g]

    def best(b):
        with capped(cap):
            return max((dict(g=g, **dict(zip(("a_rob", "min_beta"), BA.evaluate(s["rows"], b)[:2])))
                        for g, s in evals.items()), key=BA.key)

    ev(g_auth)
    for g in G_COARSE:
        ev(g)
    skipped = []
    for b in BETA_REQ:
        for step in REFINE:
            bb = best(b)
            for g in (bb["g"] - step, bb["g"] + step):
                if 0.0 <= g <= G_MAX and float(round(g, 6)) not in evals:
                    if min(cheap_of(rpm, g, st), cap) > best(b)["a_rob"]:
                        ev(g)
                    else:
                        skipped.append(float(round(g, 6)))
    return evals, sorted(set(skipped) - set(evals))


def gain_summary(s, cap):
    """a_rob, smallest kick, failing depth and binding of one evaluated gain (inside capped(cap))."""
    assert BA.A_PHYS == cap
    out = dict(a_lim_mm=s["a_lim"], source=s["source"], rows=len(s["rows"]))
    for b in BETA_REQ:
        ar, mb, fl = BA.evaluate(s["rows"], b)
        out[lab(b)] = dict(a_rob_mm=ar, min_beta_mmps=mb * 1e3, fail_depth_mm=None if fl is None else fl["d"],
                           binding=rename(BA.binding(fl, b, ar)))
    return out


def design_record(dz, evals, cap, b):
    """The design of basin_aware.recert for one beta_req, in the units of basin_aware.json (inside capped)."""
    assert BA.A_PHYS == cap
    if dz.get("g") is None:
        return dict(g=None, a_rob_mm=0.0, binding=[], candidates_recertified=dz.get("tried", []))
    ties = sorted(g for g, s in evals.items() if BA.evaluate(s["rows"], b)[0] == dz["a_rob_red"])
    nr, nb = dz["next_row"], dz["next_ref"]
    mm = (lambda x: None if x is None else x * 1e3)
    return dict(
        g=dz["g"], a_rob_mm=dz["a_rob"], binding=rename(dz["binding"]),
        a_rob_reduced_order_mm=dz["a_rob_red"], corrected_down=dz["corrected"],
        min_beta_up_to_a_rob_mmps=dz["min_beta"] * 1e3, min_beta_reduced_order_mmps=dz["min_beta_red"] * 1e3,
        a_lim_at_g_mm=dz["a_lim"], beta_default_N_at_a_rob_mmps=mm(dz["beta_ref_at_a_rob"]),
        beta_reduced_N_at_a_rob_mmps=mm(dz["beta_red_at_a_rob"]), bbar_at_a_rob_V=dz["bbar_at_a_rob"],
        upward_possible=dz["upward_possible"],
        next_depth=None if nr is None else dict(d_mm=nr["d"], linear_ok=nr["lin"], bbar_V=nr["bbar"], volt_ok=nr["volt"],
                                                beta_reduced_N_mmps=mm(nr.get("beta")),
                                                beta_default_N_mmps=None if nb is None else nb["beta"] * 1e3),
        default_N_checks={d: dict(beta_mmps=v["beta"] * 1e3, fast_mmps=v["fast"] * 1e3, tight_mmps=mm(v["tight"]),
                                  settings=v["settings"], note=v["note"]) for d, v in dz["checks"].items()},
        gains_tied_at_reduced_order=ties, candidates_recertified=dz["tried"],
        source=evals[dz["g"]]["source"])


# ------------------------------------------------------------------- checks ----
def recheck(rpm, st, k):
    """Recompute the k cheapest stored reduced-order rows and the cheapest stored default-order c2.FAST
    certificate of this speed (basin_aware cache) and compare."""
    if k <= 0:
        return []
    out = []
    rows = sorted(((r.get("t", 1e9), g, r) for g, s in st.old_scans.items() for r in s["rows"] if "beta" in r),
                  key=lambda x: (x[0], x[1], x[2]["d"]))
    for t_old, g, r in rows[:k]:
        key = ["recheck_row", rpm, g, r["d"]]
        v = st.get(key)
        if v is None:
            a = r["d"] * 1e-3
            v = st.put(key, BA.certificate(rpm, g, a, BA.n_red(rpm), "FAST", BA.bbar_of(rpm, g, a)))
        out.append(dict(kind="reduced-order row", g=g, d_mm=r["d"], N=v["N"], stored_beta_mmps=r["beta"] * 1e3,
                        beta_mmps=v["beta"] * 1e3, rel_diff=rel(v["beta"], r["beta"]),
                        rel_diff_raw=rel(v["beta_raw"], r["beta_raw"]), t_s=v["t"], stored_t_s=t_old))
    refs = sorted((v["t"], json.loads(s)) for s, v in st.old.items()
                  if json.loads(s)[0] == "ref" and json.loads(s)[4] == "FAST")
    if refs:
        t_old, kk = refs[0]
        _, _, g, d, _ = kk
        key = ["recheck_ref", rpm, g, d, "FAST"]
        v = st.get(key)
        if v is None:
            a = d * 1e-3
            v = st.put(key, BA.certificate(rpm, g, a, C2.nodes_for(rpm), "FAST", BA.bbar_of(rpm, g, a)))
        s = st.old[json.dumps(kk)]
        out.append(dict(kind="default-order certificate", g=g, d_mm=d, N=v["N"], stored_beta_mmps=s["beta"] * 1e3,
                        beta_mmps=v["beta"] * 1e3, rel_diff=rel(v["beta"], s["beta"]),
                        rel_diff_raw=rel(v["beta_raw"], s["beta_raw"]), t_s=v["t"], stored_t_s=t_old))
    return out


# ---------------------------------------------------------------- one speed ----
def _init(lock):
    global LOCK
    LOCK = lock
    P.set_damping(ZETA)


def speed_job(args):
    rpm = args[0]
    try:
        return _speed_job(*args)
    except Exception as e:                                       # reported; a restart redoes the speed from its cache
        return dict(rpm=rpm, error=repr(e), traceback=traceback.format_exc())


def _speed_job(rpm, caps, n_recheck, cache_path):
    P.set_damping(ZETA)
    t0 = time.time()
    st = Store(rpm, cache_path)
    cmp_ = st.get(["compare", rpm])
    if cmp_ is None:
        raise RuntimeError(f"no 'compare' record for {rpm} rpm in {BA_CACHE}")
    g_auth = float(round(cmp_["g1500"]["g"], 6))               # as basin_aware.speed_job
    checks = recheck(rpm, st, n_recheck)
    # every coarse gain (and g_auth) scanned up to the largest cap
    coarse_here = [g for g in dict.fromkeys(float(round(x, 6)) for x in (g_auth,) + G_COARSE)
                   if scan_capped(rpm, g, max(caps), st)["source"] != "basin_aware cache"]
    t_coarse = time.time() - t0
    out = {}
    for cap in sorted(caps, reverse=True):
        t1, c_before = time.time(), dict(st.cost)
        evals, skipped = search_capped(rpm, cap, g_auth, st)
        with capped(cap):
            designs = BA.recert(rpm, evals, st)
            rec = {lab(b): design_record(designs[str(b)], evals, cap, b) for b in BETA_REQ}
            ev_sum = {f"{g}": gain_summary(s, cap) for g, s in sorted(evals.items())}
        out[f"{cap:.0f}"] = dict(designs=rec, evaluated=ev_sum, skipped_refinement=skipped,
                                 refinement_gains=sorted(g for g in evals if g != g_auth and g not in G_COARSE),
                                 auth_gain_under_capped_criteria=dict(g=g_auth, **{b: v for b, v in ev_sum[f"{g_auth}"].items()
                                                                                   if b in ("1", "2", "5")}),
                                 cost={k: st.cost[k] - c_before[k] for k in st.cost}, t_s=round(time.time() - t1, 1))
    return dict(rpm=rpm, n_red=BA.n_red(rpm), n_default=C2.nodes_for(rpm), g_auth=g_auth,
                coarse_gains_scanned_here=coarse_here, caps=out, recheck=checks, cost=st.cost,
                cost_total=st.cost_total(), t_coarse_s=round(t_coarse, 1), t_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------------- plan ----
def plan_speed(rpm, caps, cache_path, n_recheck):
    """Coarse scan rows still to compute (upper bound: up to min(a_cheap + 2 mm, largest cap)), and a
    CPU-time estimate with the cost model."""
    st = Store(rpm, cache_path)
    cap_hi = max(caps)
    cmp_ = st.get(["compare", rpm])
    g_auth = float(round(cmp_["g1500"]["g"], 6))
    gains, n_rows = [], 0
    for g in dict.fromkeys(float(round(x, 6)) for x in (g_auth,) + G_COARSE):
        if g in st.old_scans:
            continue
        ch = st.get(["cheap", rpm, g])
        lim = cap_hi if ch is None else min(cap_hi, float(ch) + STEP)
        cnt = 0
        for d in DEPTHS_ALL:
            if d > lim:
                break
            r = st.get(["row", rpm, g, d])
            if r is None:
                cnt += 1
            elif not (r["lin"] and r["volt"] and r["beta"] >= min(BETA_REQ)):
                break
        if cnt:
            gains.append(g)
            n_rows += cnt
    nr, nd = BA.n_red(rpm), C2.nodes_for(rpm)
    done_refs = sum(1 for s in st.new if json.loads(s)[0] == "ref")
    n_refs = max(0.0, REF_PER_CAP * sum(1 for c in caps if c < A_PHYS_STORED) - done_refs)
    n_chk = 0 if n_recheck <= 0 or any(json.loads(s)[0] == "recheck_ref" for s in st.new) else 1
    est = (n_rows + n_recheck * (n_chk > 0)) * CERT_S["reduced"][nr] + (n_refs + n_chk) * CERT_S["default"][nd]
    return dict(rpm=rpm, n_red=nr, n_default=nd, coarse_gains_to_scan=gains, coarse_rows_upper_bound=n_rows,
                default_certs_estimate=n_refs, est_s=est)


# ------------------------------------------------------------------ assemble ----
def assemble(results, caps, speeds, partial, args, t_run):
    ba = json.load(open(BA_JSON))
    fd = json.load(open(FD_JSON))
    done = [n for n in speeds if n in results and "error" not in results[n]]
    ckeys = [f"{c:.0f}" for c in sorted(caps)]
    per_cap = {c: {f"{n:.0f}": results[n]["caps"][c]["designs"] for n in done} for c in ckeys}
    refs = {}
    for n in done:
        fs = fd["speeds"][f"{n:.0f}"]
        refs[f"{n:.0f}"] = {t: dict(g=fs[t]["g"], a_chk_mm=fs[t]["a_chk"],
                                    capped_mm={c: min(fs[t]["a_chk"], float(c)) for c in ckeys}) for t in REFS}
    # reproduction of basin_aware.json at a_cap = 32 mm
    repro = None
    if "32" in ckeys and done:
        pts, diffs, same_a = 0, [], 0
        for n in done:
            ps = ba["per_speed"][f"{n:.0f}"]
            here_all = results[n]["caps"]["32"]
            for b in ("1", "2", "5"):
                s, h = ps["designs"][b], here_all["designs"][b]
                pts += 1
                sb = rename(s.get("binding", []))
                ok_g = (s.get("g") is None and h["g"] is None) or (s.get("g") is not None and h["g"] is not None
                                                                     and abs(s["g"] - h["g"]) < 1e-6)
                ok = ok_g and s.get("a_rob_mm") == h["a_rob_mm"] and sb == h["binding"]
                same_a += int(s.get("a_rob_mm") == h["a_rob_mm"])
                if not ok:
                    ev = here_all["evaluated"]
                    sg = None if s.get("g") is None else f"{float(round(s['g'], 6))}"
                    diffs.append(dict(
                        rpm=n, beta_req=b,
                        stored=dict(g=s.get("g"), a_rob_mm=s.get("a_rob_mm"), binding=sb,
                                    min_beta_mmps=s.get("min_beta_up_to_a_rob_mmps")),
                        here=dict(g=h["g"], a_rob_mm=h["a_rob_mm"], binding=h["binding"],
                                  min_beta_mmps=h.get("min_beta_up_to_a_rob_mmps")),
                        here_g_evaluated_by_basin_aware=None if h["g"] is None else f"{h['g']}" in ps["evaluated"],
                        here_g_skipped_by_basin_aware=None if h["g"] is None else h["g"] in ps["skipped_by_bound"],
                        stored_g_under_this_protocol=None if sg is None else ev.get(sg, {}).get(b)))
        repro = dict(points=pts, identical=pts - len(diffs), same_a_rob=same_a, differences=diffs,
                     note="g*, a_rob and binding against basin_aware.json per_speed designs (binding a_phys = a_cap)")
    # summary
    summ = {}
    for c in ckeys:
        cs = {}
        rc = {t: np.array([refs[f"{n:.0f}"][t]["capped_mm"][c] for n in done]) for t in REFS}
        cs["ref_capped_mean_mm"] = {t: float(rc[t].mean()) for t in REFS} if done else {}
        for b in ("1", "2", "5"):
            d = [per_cap[c][f"{n:.0f}"][b] for n in done]
            if not d:
                continue
            ar = np.array([x["a_rob_mm"] for x in d])
            gs = np.array([np.nan if x["g"] is None else x["g"] for x in d])
            prim = [x["binding"][0] if x["binding"] else "none" for x in d]
            cs[b] = dict(beta_req_mmps=float(b), n_speeds=len(d), mean_a_rob_mm=float(ar.mean()),
                         min_a_rob_mm=float(ar.min()), max_a_rob_mm=float(ar.max()),
                         n_at_cap=int(np.sum(ar >= float(c))), mean_g=float(np.nanmean(gs)),
                         primary_binding_counts={k: prim.count(k) for k in ("a_cap", "linear limit", "voltage margin",
                                                                             "certificate", "none")},
                         corrected_down_rpm=[n for n, x in zip(done, d) if x.get("corrected_down")],
                         a_rob_minus_ref_capped_mean_mm={t: float(np.mean(ar - rc[t])) for t in REFS},
                         n_a_rob_below_ref_capped={t: int(np.sum(ar < rc[t] - 1e-9)) for t in REFS},
                         a_rob_by_rpm={f"{n:.0f}": x["a_rob_mm"] for n, x in zip(done, d)},
                         g_by_rpm={f"{n:.0f}": x["g"] for n, x in zip(done, d)})
        summ[c] = cs
    rch = [dict(rpm=n, **x) for n in done for x in results[n]["recheck"]]
    cost = {k: float(sum(results[n]["cost"][k] for n in done)) for k in results[done[0]]["cost"]} if done else {}
    cost_tot = {k: float(sum(results[n]["cost_total"][k] for n in done)) for k in results[done[0]]["cost_total"]} if done else {}
    out = dict(
        note=__doc__,
        protocol=dict(zeta=ZETA, vmax_V=BA.VMAX, vlim_V=BA.VLIM, caps_mm=[float(c) for c in sorted(caps)],
                      depth_grid_mm=list(DEPTHS_ALL), beta_req_mmps=[b * 1e3 for b in BETA_REQ],
                      coarse_gains=list(G_COARSE), coarse="every coarse gain (no branch and bound) and g_auth",
                      refine_steps=list(REFINE), refine_rule="basin_aware.search: skipped when min(a_cheap, a_cap) <= best",
                      margin=BA.MARGIN, n_red="nodes_for(rpm) - 8", scan_settings="c2.FAST",
                      recert="basin_aware.recert with A_PHYS = a_cap and DEPTHS = grid up to a_cap",
                      cap_order=[float(c) for c in sorted(caps, reverse=True)],
                      basin_aware_cache=os.path.relpath(BA_CACHE, ROOT), basin_aware_cache_id=BA.PROTO,
                      references="fair_designs.json speeds[n][pub|auth0|auth10|auth1500].a_chk, capped min(a_chk, a_cap)",
                      speeds=list(speeds), partial=bool(partial), recheck_per_speed=args.recheck),
        per_cap=per_cap,
        per_speed={f"{n:.0f}": dict(rpm=n, n_red=results[n]["n_red"], n_default=results[n]["n_default"],
                                    g_auth=results[n]["g_auth"], coarse_gains_scanned_here=results[n]["coarse_gains_scanned_here"],
                                    caps={c: {k: v for k, v in results[n]["caps"][c].items() if k != "designs"} for c in ckeys},
                                    recheck=results[n]["recheck"], cost_this_call=results[n]["cost"],
                                    cost_all_calls=results[n]["cost_total"], t_coarse_s=results[n]["t_coarse_s"], t_s=results[n]["t_s"]) for n in done},
        references=refs,
        reproduction_cap32=repro,
        summary=dict(per_cap=summ, n_speeds_done=len(done),
                     uncapped_ref_mean_mm=fd["summary"]["mean_a_chk_mm"],
                     basin_aware_json_mean_a_rob_mm={b: ba["summary"][b]["a_rob_mm_grid21"]["mean"] for b in ("1", "2", "5")},
                     reproduction_cap32=None if repro is None else {k: repro[k] for k in ("points", "identical", "same_a_rob")},
                     recheck=dict(n=len(rch), max_rel_diff=max((x["rel_diff"] for x in rch), default=None),
                                  all_within_rtol=bool(all(x["rel_diff"] <= RECHECK_RTOL for x in rch)), rtol=RECHECK_RTOL),
                     errors={f"{n:.0f}": results[n]["error"] for n in speeds if n in results and "error" in results[n]},
                     missing_speeds=[n for n in speeds if n not in results]),
        runtime=dict(new_certificates_this_call=cost, certificates_all_calls=cost_tot,
                     note="certificate CPU seconds as recorded by basin_aware.certificate; all_calls = every record "
                          "of these speeds in this cache (also across restarts)", wall_s_this_call=round(t_run, 1),
                     t_s_by_rpm={f"{n:.0f}": results[n]["t_s"] for n in done}))
    return out


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


# ---------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--speeds", default="", help="comma-separated subset of the 21 speeds (rpm)")
    ap.add_argument("--caps", default="", help="comma-separated subset of the caps 10,16,32 (mm; grid depths)")
    ap.add_argument("--recheck", type=int, default=1, help="stored reduced-order rows recomputed per speed (and one "
                                                           "default-order certificate if > 0)")
    ap.add_argument("--out", default="ba_cap", help="output name: <out>.json and <out>_cache.jsonl here")
    ap.add_argument("--smoke", action="store_true", help="same as --out ba_cap_smoke")
    ap.add_argument("--plan", action="store_true", help="list the coarse scans still to do, with a CPU estimate, and exit")
    a = ap.parse_args()
    out_name = "ba_cap_smoke" if a.smoke else a.out
    speeds = tuple(float(x) for x in a.speeds.split(",") if x.strip()) or SPEEDS
    caps = tuple(float(x) for x in a.caps.split(",") if x.strip()) or CAPS
    assert all(n in SPEEDS for n in speeds), "speeds must be chart speeds"
    assert all(c in DEPTHS_ALL for c in caps), "caps must be grid depths (2 ... 32 mm)"
    partial = set(speeds) != set(SPEEDS) or set(caps) != set(CAPS)
    if partial and out_name == "ba_cap" and not a.plan:
        sys.exit("a subset run must not write ba_cap.json: use --smoke or --out NAME")
    ba = json.load(open(BA_JSON))
    assert ba["protocol"]["cache"] == BA.PROTO and os.path.isdir(BA_CACHE), "basin_aware cache does not match basin_aware.json"
    out_path = os.path.join(HERE, out_name + ".json")
    cache_path = os.path.join(HERE, out_name + "_cache.jsonl")
    repair(cache_path)
    plans = {n: plan_speed(n, caps, cache_path, a.recheck) for n in speeds}
    tot = sum(p["est_s"] for p in plans.values())
    for n, p in plans.items():
        print(f"plan {n:6.0f} rpm (N_red {p['n_red']}, N {p['n_default']}): {len(p['coarse_gains_to_scan'])} coarse gains to "
              f"scan, <= {p['coarse_rows_upper_bound']} rows; about {p['default_certs_estimate']:.0f} default-order "
              f"certificates; estimate {p['est_s'] / 60:.1f} CPU-min", flush=True)
    print(f"plan: {len(speeds)} speeds, caps {list(caps)}: estimate {tot / 3600:.2f} CPU-h "
          f"(upper bound on the coarse rows; recert {REF_PER_CAP:.0f} certificates per cap below 32 mm)", flush=True)
    if a.plan:
        return
    order = sorted(speeds, key=lambda n: -plans[n]["est_s"])         # longest first
    t0 = time.time()
    results = {}
    lock = mp.Lock()
    with mp.Pool(a.workers, initializer=_init, initargs=(lock,)) as pool:
        for r in pool.imap_unordered(speed_job, [(n, caps, a.recheck, cache_path) for n in order]):
            n = r["rpm"]
            results[n] = r
            if "error" in r:
                print(f"{n:6.0f} rpm: FAILED {r['error']}\n{r['traceback']}", flush=True)
                continue
            line = "; ".join(f"cap {c}: " + ", ".join(f"{b}: g {r['caps'][c]['designs'][b]['g']} a_rob "
                                                      f"{r['caps'][c]['designs'][b]['a_rob_mm']:.0f} "
                                                      f"({'/'.join(r['caps'][c]['designs'][b]['binding'])})"
                                                      for b in ("1", "2", "5")) for c in r["caps"])
            cs = r["cost"]
            print(f"{n:6.0f} rpm: {line}  [{len(r['coarse_gains_scanned_here'])} coarse gains scanned here; new "
                  f"certificates: {cs['reduced_n']} reduced ({cs['reduced_s']:.0f} s), {cs['fast_n']} FAST + "
                  f"{cs['tight_n']} TIGHT default ({cs['fast_s'] + cs['tight_s']:.0f} s); {r['t_s']:.0f} s]", flush=True)
            write_json(out_path, assemble(results, caps, speeds, partial, a, time.time() - t0))
    out = assemble(results, caps, speeds, partial, a, time.time() - t0)
    write_json(out_path, out)
    s = out["summary"]
    for c, cs in s["per_cap"].items():
        print(f"a_cap {c} mm: " + "; ".join(f"beta {b}: mean a_rob {cs[b]['mean_a_rob_mm']:.2f} mm ({cs[b]['n_at_cap']} at the cap)"
                                            for b in ("1", "2", "5") if b in cs)
              + "; refs capped: " + ", ".join(f"{t} {v:.2f}" for t, v in cs["ref_capped_mean_mm"].items()), flush=True)
    if s["reproduction_cap32"]:
        print(f"a_cap 32 against basin_aware.json: {s['reproduction_cap32']}", flush=True)
    print(f"recheck: {s['recheck']}", flush=True)
    print(f"wrote {out_path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
