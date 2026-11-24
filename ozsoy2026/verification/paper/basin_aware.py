"""Basin-aware gain scheduling of clipped DVF: the deepest cut with a certified kick basin.

Purpose.  The authority-only schedule of Section 3 (Eq. feas) picks, per speed, the largest
gain whose steady forced command fits the 9 V clip up to 0.9 a_lim; its checked depth a_chk
rests on one simulated step entry and lies inside the bistable band (Sections 3-4).  The
manuscript left open "a gain map that keeps a certified kick at every depth up to kappa a_lim"
(Section 7).  This script computes it: per speed n, the largest depth a_rob(n; beta_req)
that some DVF gain g keeps robust, and that gain g*.

Definitions.  A grid depth a in {2, 4, ..., 32} mm is admissible for (n, g, beta_req) if
  (L) a < a_lim(n, g)                         ZOA linear limit (zoa.a_lim);
  (V) bbar(n, a, g) <= (1 - rho) Vmax         rho = 0.1, Vmax = 9 V: the rigorous bound on the
                                              forced command (c3_periodic.orbit, largest of its
                                              16 segment bounds, harmonics above the 40th
                                              included), i.e. 10 % of the voltage kept free;
  (P) a <= a_phys = 32 mm                     twice the cutter diameter;
  (C) beta(n, a, g) >= beta_req               certified kick of the averaged collocated model
                                              (c2_certificate.certify_kick: e = unit velocity,
                                              delta = 0.5, the 9 V clip), rescaled to the
                                              rigorous bound as in the manuscript:
                                              beta = beta_c2 min(1, (9 - bbar) / (9 - bmax_c2)).
a_rob(n, g; beta_req) is the largest grid depth such that every grid depth from 2 mm up to it
is admissible (0 if 2 mm is not); a_rob(n; beta_req) = max over the candidate gains, g* the
maximiser (ties: the larger smallest certified kick over the depths up to a_rob, then the
smaller gain).  beta_req = 1, 2 and 5 mm/s.  The binding constraint is the one that fails at
the next grid depth a_rob + 2 mm ('a_phys' when a_rob = 32 mm).

Protocol (damping of every stored result: params.set_damping(gainmap_main.npz zeta), 1.34 %).
  speeds   the 21 speeds 500:125:3000 rpm; the C1 basin speeds 1090, 1420, 2500, 2870 are
           represented by the grid speeds 1125, 1375, 2500, 2875 (to keep the campaign within
           its budget on a shared machine).
  compare  per speed: a_lim(n, 0); the authority-only design of the manuscript, g_auth =
           feas_g1500.npz g_best and a_chk = feas_cert_g1500.npz a_cert (off the stored 10 rpm
           grid recomputed with feasibility.edge_gain(g_hi = 1500) and the step-entry check of
           src/run_feas_certify.py; also at two stored speeds, as a check); the same with
           g <= 3000 (feas_g3000 / feas_cert_g3000); the published-condition depth
           gainmap_main.npz a_best (off grid: the two neighbouring stored speeds, flagged).
  scan     candidate gains: G_COARSE = 0:250:3000 V s/m, then, for each beta_req, +-125 and
           then +-62.5 V s/m around the best gain so far.  Branch and bound: a_cheap(n, g),
           the largest grid depth up to which (L), (V) and (P) hold (closed form), bounds
           a_rob(n, g; beta_req) from above for every beta_req; coarse gains are taken in
           decreasing order of a_cheap and skipped when a_cheap cannot exceed the best a_rob
           found so far for any beta_req, refinement gains when it cannot exceed that
           beta_req's best (so ties with the best are not explored).  g_auth is always
           evaluated, for the comparison.  One gain: depths upward from 2 mm until a depth
           is not admissible for beta_req = 1 mm/s (at a (V) failure with bbar < 9 V the
           certificate is still computed, for the report).  Certificates with the reduced
           order N_red = nodes_for(n) - 8 (24 at 500-625 rpm, 20 at 750-875, 16 at 1000-1875,
           12 at 2000-3000; basin_aware_nconv.json: its linear limit is within 1.2e-4 of the
           ZOA limit, where N = 10-14 fails at low speed) and the solver defaults (c2.FAST).
  recert   per speed and beta_req, with the default order N = nodes_for(n): the design point
           (n, g*, a_rob), every lower grid depth whose reduced-order kick is below
           MARGIN = 1.15 beta_req, and the failing depth a_rob + 2 mm.  Each default-order
           certificate is computed with c2.FAST and, at the design point and the lower depths,
           if below beta_req also with c2.TIGHT; the larger is kept (every candidate is checked
           by c2.check whatever the solver settings).  At the failing depth c2.FAST only.  If one falls below beta_req, a_rob is lowered to the grid
           depth below it and the check repeated; if g* then falls below another gain's
           reduced-order a_rob, that gain is re-certified the same way and the larger
           corrected a_rob wins.  A pass at the failing depth is flagged ('could be one step
           higher'), not applied.
  validate at VAL_SPEEDS (750, 1125, 1375, 2000, 2500, 2875 rpm), per beta_req: the simulated critical kick (c1_unsafe_zone.
           _critical_kick: face entry, full surface memory, |V| <= 9 V, kick at revolution
           100, 300-revolution window, D1 classifier) at the kick phases 0, 4, 8, 12 of 16 at
           the design point, and at the phases 0 and 8 at the failing depth a_rob + 2 mm;
           kicks of exactly beta_req at all 16 phases at the design point; the critical kick
           (same 4 phases) of the authority-only design at its checked depth a_chk.  At every
           speed and design point: one step entry from rest and one face entry with the 9 V
           clip and the full surface memory (tdboundary.classify), which the certificate does
           not cover (it certifies kicks on the settled cut).
  assemble basin_aware.json (numbers; every quoted value has a key) and basin_aware.npz
           (arrays for the 'basin-aware stability chart': depth against speed for a_lim(0),
           a_lim(g_auth), a_chk, the published-condition depth, a_lim(g*) and a_rob for the
           three beta_req; plus the dense stored curves over 251 speeds and, per speed and
           coarse gain, bbar and the reduced-order kick on the depth grid).

How to run (from the repository root ozsoy2026/; about 2 h with 2 workers):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup python3 verification/paper/basin_aware.py \
        all --workers 2 > verification/paper/basin_aware.log 2>&1 &
  Stages: scan (includes compare and recert), validate, assemble; 'all' runs them in order.
  Every unit of work is appended to verification/paper/basin_aware_cache/<protocol hash>/ as
  it finishes, so an interrupted run resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)
RES = os.path.join(HERE, "..", "..", "results")
ZETA = float(np.load(os.path.join(RES, "gainmap_main.npz"))["zeta"])

VMAX = 9.0                                    # F_max / K_a = 27 N / 3 N/V
RHO = 0.1
VLIM = (1.0 - RHO) * VMAX                     # 8.1 V
A_PHYS = 32.0                                 # mm, twice the cutter diameter
DEPTHS = tuple(float(d) for d in np.arange(2.0, A_PHYS + 0.01, 2.0))      # mm
STEP = 2.0
BETA_REQ = (1e-3, 2e-3, 5e-3)                 # m/s
GRID_RPM = tuple(float(n) for n in np.arange(500.0, 3001.0, 125.0))
C1_RPM = (1090.0, 1420.0, 2500.0, 2870.0)
SPEEDS = GRID_RPM                             # the C1 basin speeds are represented by the nearest grid speeds
G_COARSE = tuple(float(g) for g in np.arange(0.0, 3001.0, 250.0))
REFINE = (125.0, 62.5)
G_MAX = 3000.0
MARGIN = 1.15                                 # reduced-order kicks below MARGIN beta_req are re-certified
VAL_SPEEDS = (750.0, 1125.0, 1375.0, 2000.0, 2500.0, 2875.0)     # 1125, 1375, 2875: nearest to C1's 1090, 1420, 2870
VAL_PHASES = (0, 4, 8, 12)                    # of c1_unsafe_zone.N_PHASE = 16
NEXT_PHASES = (0, 8)                          # at the failing depth (information only)
PROTO = hashlib.sha1(repr((ZETA, VLIM, A_PHYS, DEPTHS, BETA_REQ, G_COARSE, REFINE, MARGIN, "Nd-8", "FAST",
                           "v1")).encode()).hexdigest()[:10]
CACHE = os.path.join(HERE, "basin_aware_cache", PROTO)


# ------------------------------------------------------------------ utilities ----
def _setup():
    import params as P
    P.set_damping(ZETA)


class JL:
    """Append-only JSON-lines cache: key (a JSON-serialisable list) -> value."""

    def __init__(self, name):
        os.makedirs(CACHE, exist_ok=True)
        self.path = os.path.join(CACHE, name)
        self.d = {}
        if os.path.exists(self.path):
            good = 0
            with open(self.path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:              # a line cut short by an interruption
                        break
                    self.d[json.dumps(rec["k"])] = rec["v"]
                    good += len(line)
            if os.path.getsize(self.path) > good:
                with open(self.path, "r+") as fh:
                    fh.truncate(good)

    def get(self, k):
        return self.d.get(json.dumps(k))

    def put(self, k, v):
        self.d[json.dumps(k)] = v
        with open(self.path, "a") as fh:
            fh.write(json.dumps(dict(k=k, v=v), default=float) + "\n")
            fh.flush()
        return v


def n_red(rpm):
    import c2_certificate as c2
    return c2.nodes_for(rpm) - 8


def bbar_of(rpm, g, a):
    """Rigorous bound on |b(t)| = |g x_p'(t)| (V) over the tooth period."""
    if g == 0.0:
        return 0.0
    import c3_periodic as c3
    return float(c3.orbit(rpm, g, a)[2].max())


def certificate(rpm, g, a, N, settings_name, bbar):
    """Certified kick (m/s) rescaled to the rigorous command bound, and its record."""
    import c2_certificate as c2
    st = dict(FAST=c2.FAST, TIGHT=c2.TIGHT)[settings_name]
    t = time.time()
    beta_raw, note = c2.certify_kick(rpm, a, g, N=N, settings=st)
    bm = float(c2.orbit(rpm, g, a, N)[2]) if g > 0 else 0.0
    fac = 1.0 if g == 0.0 else (min(1.0, (VMAX - bbar) / (VMAX - bm)) if bm < VMAX else 0.0)
    return dict(beta=float(beta_raw) * max(fac, 0.0), beta_raw=float(beta_raw), bmax_c2=bm, fac=fac, N=int(N),
                settings=settings_name, note=str(note), t=round(time.time() - t, 2))


# ----------------------------------------------------------- one gain: depth scan ----
def scan_gain(rpm, g):
    """Rows of the upward depth scan of one (speed, gain) with the reduced order."""
    from zoa import a_lim
    al = a_lim(rpm, g) * 1e3
    N = n_red(rpm)
    rows = []
    for d in DEPTHS:
        a = d * 1e-3
        lin = bool(d < al)
        bb = bbar_of(rpm, g, a)
        row = dict(d=d, lin=lin, bbar=bb, volt=bool(bb <= VLIM))
        if lin and bb < VMAX:
            row.update(certificate(rpm, g, a, N, "FAST", bb))
        rows.append(row)
        if not (lin and row["volt"] and row["beta"] >= min(BETA_REQ)):
            break
    return dict(rpm=rpm, g=g, a_lim=al, N=N, rows=rows)


def evaluate(rows, beta_req):
    """a_rob (mm), smallest kick up to it, and the first failing row for one beta_req."""
    a_rob, mins, fail = 0.0, np.inf, None
    for r in rows:
        ok = r["lin"] and r["volt"] and r.get("beta", 0.0) >= beta_req
        if not ok:
            fail = r
            break
        a_rob, mins = r["d"], min(mins, r["beta"])
    if fail is None and a_rob < A_PHYS:
        raise RuntimeError("scan stopped without a failing row")       # cannot happen: the scan stops at a failure
    return a_rob, (mins if a_rob > 0 else 0.0), fail


def binding(fail, beta_req, a_rob):
    """Constraints failing at the next grid depth, primary first."""
    if a_rob >= A_PHYS:
        return ["a_phys"]
    out = []
    if not fail["lin"]:
        out.append("linear limit")
    if not fail["volt"]:
        out.append("voltage margin")
    if fail["lin"] and fail.get("beta", 0.0) < beta_req:
        out.append("certificate")
    return out


def key(ev):
    """Selection key of a candidate: a_rob, then the smallest kick up to it, then the smaller gain."""
    return (ev["a_rob"], ev["min_beta"], -ev["g"])


# ---------------------------------------------------------- comparison designs ----
def compare(rpm, cache):
    """Authority-only designs (g <= 1500 and g <= 3000), a_lim(0), the published-condition depth."""
    k = ["compare", rpm]
    v = cache.get(k)
    if v is not None:
        return v
    import feasibility as F
    from zoa import a_lim
    from tdboundary import boundary
    out = dict(rpm=rpm, a_lim0=a_lim(rpm, 0.0) * 1e3)
    for tag, gmax in (("g1500", 1500.0), ("g3000", 3000.0)):
        d = np.load(os.path.join(RES, f"feas_{tag}.npz"))
        c = np.load(os.path.join(RES, f"feas_cert_{tag}.npz"))
        i = np.nonzero(np.isclose(d["rpm"], rpm))[0]
        t = time.time()
        g_edge = float(F.edge_gain(rpm, g_hi=gmax))
        rec = dict(g_edge=g_edge)
        recompute = (not i.size) or (rpm in (2000.0, 2500.0) and tag == "g1500")
        if i.size:
            rec.update(g=float(d["g_best"][i[0]]), a_lim=float(c["a_lim"][i[0]]), a_chk=float(c["a_cert"][i[0]]),
                       a_td=float(c["a_td_clipped"][i[0]]), source="stored")
        if recompute:
            g = float(d["g_best"][i[0]]) if i.size else g_edge
            al = a_lim(rpm, g)
            td = boundary(rpm, g, vmax=VMAX if g > 0 else 0.0, a_lo=0.25 * 0.9 * al, a_hi=0.9 * al,
                          growth=1.08, nreg=F.NREG)
            r2 = dict(g=g, a_lim=al * 1e3, a_chk=min(0.9 * al, td) * 1e3, a_td=td * 1e3)
            if i.size:
                rec["recomputed"] = r2
            else:
                rec.update(r2, source="recomputed (edge_gain and the step-entry check of run_feas_certify)")
        rec["t"] = round(time.time() - t, 1)
        out[tag] = rec
    gm = np.load(os.path.join(RES, "gainmap_main.npz"))
    i = np.nonzero(np.isclose(gm["rpm"], rpm))[0]
    if i.size:
        out["published"] = dict(g=float(gm["g_best"][i[0]]), a=float(gm["a_best"][i[0]]), source="stored")
    else:
        j = np.searchsorted(gm["rpm"], rpm)
        out["published"] = dict(g=None, a=float(0.5 * (gm["a_best"][j - 1] + gm["a_best"][j])),
                                neighbours=[[float(gm["rpm"][j - 1]), float(gm["a_best"][j - 1]), float(gm["g_best"][j - 1])],
                                            [float(gm["rpm"][j]), float(gm["a_best"][j]), float(gm["g_best"][j])]],
                                source="mean of the two neighbouring stored speeds (published map not computed here)")
    return cache.put(k, out)


# ------------------------------------------------------------ one speed: search ----
def a_cheap(rpm, g):
    """Largest grid depth up to which every grid depth satisfies (L), (V) and (P): an upper
    bound on a_rob(n, g; beta_req) for every beta_req (closed form, no certificate)."""
    from zoa import a_lim
    al = a_lim(rpm, g) * 1e3
    last = 0.0
    for d in DEPTHS:
        if not (d < al and bbar_of(rpm, g, d * 1e-3) <= VLIM):
            break
        last = d
    return last


def search(rpm, cache, g_auth):
    """Branch and bound over the gains.  Candidates: G_COARSE in decreasing order of a_cheap
    (then decreasing g), each skipped when a_cheap(g) cannot exceed the best a_rob found so far
    for any beta_req; then, per beta_req, +-125 and +-62.5 V s/m around the best gain, skipped
    when a_cheap cannot exceed that beta_req's best.  g_auth is always evaluated (comparison)."""
    evals, cheap, skipped = {}, {}, []

    def ac(g):
        g = float(round(g, 6))
        if g not in cheap:
            k = ["cheap", rpm, g]
            v = cache.get(k)
            cheap[g] = v if v is not None else cache.put(k, a_cheap(rpm, g))
        return cheap[g]

    def ev(g):
        g = float(round(g, 6))
        if g in evals:
            return evals[g]
        k = ["scan", rpm, g]
        s = cache.get(k)
        if s is None:
            s = cache.put(k, scan_gain(rpm, g))
        evals[g] = s
        return s

    def best(b):
        return max((dict(g=g, **dict(zip(("a_rob", "min_beta"), evaluate(s["rows"], b)[:2]))) for g, s in evals.items()),
                   key=key)

    ev(g_auth)
    for g in sorted(G_COARSE, key=lambda x: (-ac(x), -x)):
        if ac(g) <= min(best(b)["a_rob"] for b in BETA_REQ):
            skipped.append(g)
            continue
        ev(g)
    for b in BETA_REQ:
        for step in REFINE:
            bb = best(b)
            for g in (bb["g"] - step, bb["g"] + step):
                if 0.0 <= g <= G_MAX and float(round(g, 6)) not in evals:
                    if ac(g) > best(b)["a_rob"]:
                        ev(g)
                    else:
                        skipped.append(float(round(g, 6)))
    return evals, cheap, sorted(set(skipped) - set(evals))


# ------------------------------------------------------- one speed: re-certify ----
def recert(rpm, evals, cache):
    """Default-order certificates near the selected designs; corrected a_rob per beta_req."""
    import c2_certificate as c2
    Nd = c2.nodes_for(rpm)

    def ref(g, d, need):
        """Best default-order kick at (g, d): FAST, and TIGHT if FAST < need."""
        a = d * 1e-3
        bb = bbar_of(rpm, g, a)
        out = {}
        for st in ("FAST", "TIGHT"):
            k = ["ref", rpm, g, d, st]
            v = cache.get(k)
            if v is None and (st == "FAST" or out["FAST"]["beta"] < need):
                v = cache.put(k, certificate(rpm, g, a, Nd, st, bb))
            if v is not None:
                out[st] = v
        best = max(out.values(), key=lambda r: r["beta"])
        return dict(best, fast=out["FAST"]["beta"], tight=out.get("TIGHT", {}).get("beta"))

    designs = {}
    for b in BETA_REQ:
        cands = sorted(((dict(g=g, **dict(zip(("a_rob", "min_beta"), evaluate(s["rows"], b)[:2]))), s)
                        for g, s in evals.items()), key=lambda x: key(x[0]), reverse=True)
        best, tried = None, []
        for e, s in cands:
            if e["a_rob"] <= 0:
                break
            if best is not None and e["a_rob"] <= best["a_rob"]:
                break
            rows = {r["d"]: r for r in s["rows"]}
            a = e["a_rob"]
            checks = {}
            while a >= DEPTHS[0]:
                todo = [a] + [d for d in DEPTHS if d < a and rows[d]["beta"] < MARGIN * b]
                fails = []
                for d in todo:
                    if d not in checks:
                        checks[d] = ref(e["g"], d, b)
                    if checks[d]["beta"] < b:
                        fails.append(d)
                if not fails:
                    break
                a = min(fails) - STEP
            a = max(a, 0.0)
            nxt = a + STEP
            nrow = rows.get(nxt)
            # the failing depth: re-certified where the certificate can be what binds there
            # (below the linear limit and inside the voltage margin)
            if a < A_PHYS and nrow is not None and nrow["lin"] and nrow["volt"] and nxt not in checks:
                checks[nxt] = ref(e["g"], nxt, 0.0)                   # c2.FAST only (information)
            # smallest kick up to a_rob, default order where computed, reduced order elsewhere
            mins = min(checks[d]["beta"] if d in checks else rows[d]["beta"] for d in DEPTHS if d <= a) if a > 0 else 0.0
            res = dict(g=e["g"], a_rob_red=e["a_rob"], a_rob=a, min_beta=mins, min_beta_red=e["min_beta"],
                       checks={str(d): checks[d] for d in sorted(checks)})
            tried.append(res)
            if best is None or (res["a_rob"], res["min_beta"], -res["g"]) > (best["a_rob"], best["min_beta"], -best["g"]):
                best = res
        if best is None:
            designs[str(b)] = dict(g=None, a_rob=0.0, tried=[dict(g=t["g"], a_rob_red=t["a_rob_red"], a_rob=t["a_rob"])
                                                             for t in tried])
            continue
        s = evals[best["g"]]
        rows = {r["d"]: r for r in s["rows"]}
        a = best["a_rob"]
        nxt = a + STEP
        if a >= A_PHYS:
            bind, nb = ["a_phys"], None
        else:
            nrow = dict(rows[nxt])
            nb = best["checks"].get(str(nxt))
            if nb is not None:
                nrow["beta"] = nb["beta"]                             # default-order kick at the failing depth
            bind = binding(nrow, b, a)
        best = dict(best)                                             # a copy: 'tried' holds the original
        best.update(binding=bind, a_lim=s["a_lim"], next_row=rows.get(nxt), next_ref=nb,
                    upward_possible=bool(nb is not None and nb["beta"] >= b and rows[nxt]["lin"] and rows[nxt]["volt"]),
                    corrected=bool(best["a_rob"] < best["a_rob_red"]),
                    tried=[dict(g=t["g"], a_rob_red=t["a_rob_red"], a_rob=t["a_rob"], min_beta=t["min_beta"]) for t in tried],
                    bbar_at_a_rob=rows[a]["bbar"] if a > 0 else None,
                    beta_ref_at_a_rob=best["checks"].get(str(a), {}).get("beta") if a > 0 else None,
                    beta_red_at_a_rob=rows[a]["beta"] if a > 0 else None)
        designs[str(b)] = best
    return designs


def speed_job(rpm):
    _setup()
    cache = JL(f"speed_{rpm:.0f}.jsonl")
    t0 = time.time()
    cmp_ = compare(rpm, cache)
    g_auth = float(round(cmp_["g1500"]["g"], 6))
    evals, cheap, skipped = search(rpm, cache, g_auth)
    t1 = time.time()
    designs = recert(rpm, evals, cache)
    # the authority-only gain under the same criteria (scan order, no re-certification)
    auth = {str(b): dict(zip(("a_rob", "min_beta"), evaluate(evals[g_auth]["rows"], b)[:2])) for b in BETA_REQ}
    for b in BETA_REQ:
        auth[str(b)]["binding"] = binding(evaluate(evals[g_auth]["rows"], b)[2], b, auth[str(b)]["a_rob"]) \
            if auth[str(b)]["a_rob"] < A_PHYS else ["a_phys"]
    return dict(rpm=rpm, compare=cmp_, g_auth=g_auth, n_red=n_red(rpm), evals={str(g): s for g, s in sorted(evals.items())},
                a_cheap={str(g): v for g, v in sorted(cheap.items())}, skipped=skipped,
                designs=designs, auth_eval=auth, t_scan=round(t1 - t0, 1), t_recert=round(time.time() - t1, 1))


def run_scan(workers):
    order = sorted(SPEEDS, key=lambda n: (-n_red(n), n))             # largest orders (slowest) first
    out = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(speed_job, order):
            out[r["rpm"]] = r
            d = r["designs"]
            print(f"{r['rpm']:6.0f} rpm (N_red {r['n_red']}): " + "; ".join(
                f"beta {float(b)*1e3:.0f}: g* {d[b]['g']} a_rob {d[b]['a_rob']:.0f} mm ({'/'.join(d[b].get('binding', []))})"
                for b in d) + f"  [scan {r['t_scan']:.0f} s, recert {r['t_recert']:.0f} s, "
                f"{len(r['evals'])} gains]", flush=True)
    p = os.path.join(CACHE, "speeds.json")
    old = {float(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else {}
    old.update(out)                                                   # runs on subsets of the speeds accumulate
    json.dump({str(k): v for k, v in sorted(old.items())}, open(p + ".tmp", "w"), default=float)
    os.replace(p + ".tmp", p)
    return out


# ------------------------------------------------------------------ validation ----
def kick_job(args):
    """('crit', rpm, a, g, phase) -> critical kick (m/s); ('kick', rpm, a, g, dv, phase) -> chatter."""
    _setup()
    import c1_unsafe_zone as C
    t = time.time()
    if args[0] == "crit":
        _, rpm, a, g, ph = args
        v = float(C._critical_kick((rpm, a, g, C.VMAX, ph, ZETA)))
    elif args[0] == "entry":
        # one entry with |V| <= 9 V and the full surface memory, judged by tdboundary.classify (D1)
        from tdboundary import classify
        _, rpm, a, g, kind = args
        c = classify(rpm, a, g, vmax=C.VMAX, nreg=0, **({} if kind == "step" else {"edge": True}))
        v = dict(stable=bool(c["stable"]), max_x_mm=float(c["run"]["max_x"] * 1e3), max_v_V=float(c["run"]["max_v"]))
    else:
        _, rpm, a, g, dv, ph = args
        ks = C.KICK_REV * C.SPR + (ph * C.ND) // C.N_PHASE
        v = bool(C._kick_chatters(rpm, a, g, C.VMAX, ks, dv))
    return list(args), v, round(time.time() - t, 1)


def val_jobs(speeds_res, vspeeds):
    jobs = []
    for n in vspeeds:
        r = speeds_res[n]
        for b in BETA_REQ:
            dz = r["designs"][str(b)]
            if dz["g"] is None or dz["a_rob"] <= 0:
                continue
            g, a = dz["g"], dz["a_rob"] * 1e-3
            jobs += [("crit", n, a, g, p) for p in VAL_PHASES]
            jobs += [("kick", n, a, g, b, p) for p in range(16)]
            nxt = dz["a_rob"] + STEP
            if dz["a_rob"] < A_PHYS and nxt < dz["a_lim"]:
                jobs += [("crit", n, nxt * 1e-3, g, p) for p in NEXT_PHASES]
        c = r["compare"]["g1500"]
        jobs += [("crit", n, c["a_chk"] * 1e-3, c["g"], p) for p in VAL_PHASES]
    # step entry from rest and face entry at every design point of every speed (one run each)
    for n, r in speeds_res.items():
        for b in BETA_REQ:
            dz = r["designs"][str(b)]
            if dz["g"] is not None and dz["a_rob"] > 0:
                jobs += [("entry", n, dz["a_rob"] * 1e-3, dz["g"], k) for k in ("step", "face")]
    return list(dict.fromkeys(jobs))


def run_validate(speeds_res, workers, vspeeds):
    cache = JL("validate.jsonl")
    jobs = val_jobs(speeds_res, vspeeds)
    todo = [j for j in jobs if cache.get(list(j)) is None]
    todo.sort(key=lambda j: 0 if j[0] == "crit" else 1)
    print(f"validate: {len(jobs)} jobs, {len(jobs) - len(todo)} cached", flush=True)
    if todo:
        with Pool(workers) as pool:
            for i, (k, v, t) in enumerate(pool.imap_unordered(kick_job, todo)):
                cache.put(k, dict(v=v, t=t))
                if k[0] == "crit":
                    print(f"  [{i+1}/{len(todo)}] crit {k[1]:.0f} rpm a {k[2]*1e3:.2f} mm g {k[3]:.1f} phase {k[4]}: "
                          f"{v*1e3:.3f} mm/s ({t:.0f} s)", flush=True)
                elif k[0] == "entry":
                    print(f"  [{i+1}/{len(todo)}] {k[4]} entry {k[1]:.0f} rpm a {k[2]*1e3:.0f} mm g {k[3]:.1f}: {v}", flush=True)
    return {json.dumps(list(j)): cache.get(list(j))["v"] for j in jobs}


# -------------------------------------------------------------------- assemble ----
def fine_bbar_max(rpm, g, a_rob):
    """Largest rigorous command bound over depths 0.25 mm ... a_rob in 0.25 mm steps."""
    if g is None or a_rob <= 0:
        return None, None
    ds = np.arange(0.25, a_rob + 1e-9, 0.25)
    v = np.array([bbar_of(rpm, g, d * 1e-3) for d in ds])
    return float(v.max()), float(ds[int(np.argmax(v))])


def assemble(speeds_res, val, vspeeds):
    _setup()
    from zoa import a_lim
    rpm = np.array(SPEEDS)
    nb = len(BETA_REQ)
    T = dict(a_rob=np.zeros((len(rpm), nb)), a_rob_red=np.zeros((len(rpm), nb)), g_star=np.full((len(rpm), nb), np.nan),
             a_lim_star=np.full((len(rpm), nb), np.nan), beta_at_a_rob=np.full((len(rpm), nb), np.nan),
             min_beta=np.full((len(rpm), nb), np.nan), bbar_at_a_rob=np.full((len(rpm), nb), np.nan),
             a_rob_auth=np.zeros((len(rpm), nb)))
    binding_codes = {"a_phys": 0, "linear limit": 1, "voltage margin": 2, "certificate": 3}
    T["binding"] = np.full((len(rpm), nb), -1)
    per = []
    for i, n in enumerate(rpm):
        r = speeds_res[n]
        c = r["compare"]
        row = dict(rpm=float(n), on_grid_125=bool(n in GRID_RPM), c1_speed=bool(n in C1_RPM), n_red=r["n_red"],
                   n_default=r["n_red"] + 8, a_lim0_mm=c["a_lim0"],
                   auth_g1500=dict(g=c["g1500"]["g"], a_lim_mm=c["g1500"]["a_lim"], a_chk_mm=c["g1500"]["a_chk"],
                                   kappa_a_lim_mm=0.9 * c["g1500"]["a_lim"], source=c["g1500"]["source"],
                                   recomputed=c["g1500"].get("recomputed"), g_edge=c["g1500"]["g_edge"]),
                   auth_g3000=dict(g=c["g3000"]["g"], a_lim_mm=c["g3000"]["a_lim"], a_chk_mm=c["g3000"]["a_chk"],
                                   source=c["g3000"]["source"]),
                   published=c["published"], designs={}, auth_under_basin_criteria={})
        for j, b in enumerate(BETA_REQ):
            dz = r["designs"][str(b)]
            lab = f"{b*1e3:.0f}"
            ae = r["auth_eval"][str(b)]
            T["a_rob_auth"][i, j] = ae["a_rob"]
            row["auth_under_basin_criteria"][lab] = dict(a_rob_mm=ae["a_rob"], binding=ae["binding"],
                                                         min_beta_mmps=ae["min_beta"] * 1e3, note="reduced order, not re-certified")
            if dz["g"] is None:
                row["designs"][lab] = dict(g=None, a_rob_mm=0.0)
                continue
            fb, fd = fine_bbar_max(n, dz["g"], dz["a_rob"])
            T["a_rob"][i, j], T["a_rob_red"][i, j], T["g_star"][i, j] = dz["a_rob"], dz["a_rob_red"], dz["g"]
            T["a_lim_star"][i, j] = dz["a_lim"]
            T["beta_at_a_rob"][i, j] = dz["beta_ref_at_a_rob"] if dz["beta_ref_at_a_rob"] is not None else np.nan
            T["min_beta"][i, j] = dz["min_beta"]
            T["bbar_at_a_rob"][i, j] = dz["bbar_at_a_rob"] if dz["bbar_at_a_rob"] is not None else np.nan
            T["binding"][i, j] = binding_codes[dz["binding"][0]] if dz["binding"] else -1
            ties = sorted(float(g) for g, s in r["evals"].items() if evaluate(s["rows"], b)[0] == dz["a_rob_red"])
            row["designs"][lab] = dict(
                g=dz["g"], a_rob_mm=dz["a_rob"], a_rob_reduced_order_mm=dz["a_rob_red"], corrected_down=dz["corrected"],
                a_lim_at_g_mm=dz["a_lim"], a_rob_over_a_lim_at_g=dz["a_rob"] / dz["a_lim"],
                binding=dz["binding"], upward_possible=dz["upward_possible"],
                beta_default_N_at_a_rob_mmps=None if dz["beta_ref_at_a_rob"] is None else dz["beta_ref_at_a_rob"] * 1e3,
                beta_reduced_N_at_a_rob_mmps=None if dz["beta_red_at_a_rob"] is None else dz["beta_red_at_a_rob"] * 1e3,
                min_beta_up_to_a_rob_mmps=dz["min_beta"] * 1e3, bbar_at_a_rob_V=dz["bbar_at_a_rob"],
                bbar_max_fine_0p25mm_V=fb, bbar_max_fine_depth_mm=fd,
                voltage_margin_violated_between_grid_depths=bool(fb is not None and fb > VLIM),
                next_depth=None if dz["next_row"] is None else dict(
                    d_mm=dz["next_row"]["d"], linear_ok=dz["next_row"]["lin"], bbar_V=dz["next_row"]["bbar"],
                    beta_reduced_N_mmps=None if "beta" not in dz["next_row"] else dz["next_row"]["beta"] * 1e3,
                    beta_default_N_mmps=None if dz["next_ref"] is None else dz["next_ref"]["beta"] * 1e3),
                gains_tied_at_reduced_order=ties,
                default_N_checks={d: dict(beta_mmps=v["beta"] * 1e3, fast_mmps=v["fast"] * 1e3,
                                          tight_mmps=None if v["tight"] is None else v["tight"] * 1e3, note=v["note"])
                                  for d, v in dz["checks"].items()},
                candidates_recertified=[dict(g=t["g"], a_rob_red=t["a_rob_red"], a_rob=t["a_rob"]) for t in dz["tried"]])
        row["skipped_by_bound"] = r["skipped"]
        row["a_cheap_mm"] = r["a_cheap"]
        def gain_summary(s, b):
            ar, mb, fl = evaluate(s["rows"], b)
            return dict(a_rob_mm=ar, min_beta_mmps=mb * 1e3, fail_depth_mm=None if fl is None else fl["d"],
                        binding=binding(fl, b, ar) if ar < A_PHYS else ["a_phys"],
                        fail_bbar_V=None if fl is None else fl["bbar"],
                        fail_beta_mmps=None if fl is None or "beta" not in fl else fl["beta"] * 1e3)
        row["evaluated"] = {g: dict(a_lim_mm=s["a_lim"], **{f"beta_{b*1e3:.0f}": gain_summary(s, b) for b in BETA_REQ})
                            for g, s in r["evals"].items()}
        # what keeps a larger gain from doing better: the nearest evaluated gain above g*
        for b in BETA_REQ:
            lab = f"{b*1e3:.0f}"
            dz = row["designs"].get(lab)
            if not dz or dz.get("g") is None:
                continue
            # nearest larger gain that was considered (evaluated, or skipped by the bound: then its
            # cheap constraints, (L) or (V), fail at a_cheap + 2 mm, at or below a_rob + 2 mm)
            up = sorted(float(g) for g in r["a_cheap"] if float(g) > dz["g"] + 1e-9)
            if up:
                gu = up[0]
                if str(gu) in r["evals"]:
                    dz["next_higher_gain"] = dict(g=gu, evaluated=True, a_lim_mm=r["evals"][str(gu)]["a_lim"],
                                                  **gain_summary(r["evals"][str(gu)], b))
                else:
                    ac_ = r["a_cheap"][str(gu)]
                    d_ = ac_ + STEP
                    al_ = a_lim(n, gu) * 1e3
                    if ac_ >= A_PHYS:
                        bl = ["a_phys"]
                    else:
                        bb_ = bbar_of(n, gu, d_ * 1e-3)
                        bl = (["linear limit"] if d_ >= al_ else []) + (["voltage margin"] if bb_ > VLIM else [])
                    dz["next_higher_gain"] = dict(g=gu, evaluated=False, a_lim_mm=al_, a_cheap_mm=ac_,
                                                  fail_depth_mm=None if ac_ >= A_PHYS else d_, binding=bl)
            dz["gain_limited_by"] = None if not up else dz["next_higher_gain"]["binding"]
        per.append(row)
    # entries at every design point
    for p_ in per:
        n = p_["rpm"]
        for b in BETA_REQ:
            dz = speeds_res[n]["designs"][str(b)]
            lab = f"{b*1e3:.0f}"
            if dz["g"] is not None and dz["a_rob"] > 0:
                p_["designs"][lab]["entries_9V"] = {k: val.get(json.dumps(["entry", n, dz["a_rob"] * 1e-3, dz["g"], k]))
                                                     for k in ("step", "face")}
    # validation
    vrows = {}
    for n in vspeeds:
        r = speeds_res[n]
        vr = vrows.setdefault(f"{n:.0f}", {})
        for b in BETA_REQ:
            dz = r["designs"][str(b)]
            if dz["g"] is None or dz["a_rob"] <= 0:
                continue
            g, a = dz["g"], dz["a_rob"] * 1e-3
            # np.min: a nan (the unkicked cut did not settle) propagates; inf = no chatter up to 3 m/s
            crit = np.array([val[json.dumps(["crit", n, a, g, p])] for p in VAL_PHASES], float)
            kick = [val[json.dumps(["kick", n, a, g, b, p])] for p in range(16)]
            nxt = dz["a_rob"] + STEP
            critn = np.array([val[json.dumps(["crit", n, nxt * 1e-3, g, p])] for p in NEXT_PHASES], float) \
                if dz["a_rob"] < A_PHYS and nxt < dz["a_lim"] else None
            w = float(np.min(crit))
            vr[f"{b*1e3:.0f}"] = (dict(rpm=n, beta_req_mmps=b * 1e3, g=g, a_rob_mm=dz["a_rob"],
                              beta_cert_mmps=None if dz["beta_ref_at_a_rob"] is None else dz["beta_ref_at_a_rob"] * 1e3,
                              crit_kick_mmps_by_phase=(crit * 1e3).tolist(), crit_kick_worst_mmps=w * 1e3,
                              unkicked_cut_settles=bool(np.all(np.isfinite(crit) | np.isinf(crit))),
                              above_beta_req=bool(w > b), td_over_cert=(w / dz["beta_ref_at_a_rob"])
                              if dz["beta_ref_at_a_rob"] else None,
                              kick_beta_req_16_phases_chatter=int(sum(kick)),
                              next_depth_mm=nxt if critn is not None else None,
                              next_depth_crit_kick_mmps_by_phase=None if critn is None else (critn * 1e3).tolist(),
                              next_depth_crit_kick_worst_mmps=None if critn is None else float(np.min(critn)) * 1e3,
                              next_depth_above_beta_req=None if critn is None else bool(np.min(critn) > b)))
        c = r["compare"]["g1500"]
        crit = np.array([val[json.dumps(["crit", n, c["a_chk"] * 1e-3, c["g"], p])] for p in VAL_PHASES], float)
        vr["auth"] = dict(rpm=n, design="authority-only (g_auth, a_chk)", g=c["g"], a_chk_mm=c["a_chk"],
                          crit_kick_mmps_by_phase=(crit * 1e3).tolist(), crit_kick_worst_mmps=float(np.min(crit)) * 1e3)
    vsum = {}
    for b in BETA_REQ:
        lab = f"{b*1e3:.0f}"
        rr = [v[lab] for v in vrows.values() if lab in v]
        if not rr:
            continue
        w = np.array([x["crit_kick_worst_mmps"] for x in rr])
        tc = np.array([x["td_over_cert"] for x in rr if x["td_over_cert"] is not None], float)
        nx = [x for x in rr if x["next_depth_crit_kick_worst_mmps"] is not None]
        vsum[lab] = dict(points=len(rr), all_above_beta_req=bool(all(x["above_beta_req"] for x in rr)),
                         all_settle=bool(all(x["unkicked_cut_settles"] for x in rr)),
                         worst_crit_kick_mmps=dict(min=float(np.min(w)), max=float(np.max(w))),
                         worst_crit_kick_over_beta_req_min=float(np.min(w) / (b * 1e3)),
                         td_over_cert=dict(min=float(np.min(tc)), max=float(np.max(tc)), median=float(np.median(tc)))
                         if tc.size else None,
                         kicks_of_beta_req_chattering=int(sum(x["kick_beta_req_16_phases_chatter"] for x in rr)),
                         kicks_of_beta_req_total=16 * len(rr),
                         next_depth_points=len(nx),
                         next_depth_above_beta_req=int(sum(1 for x in nx if x["next_depth_above_beta_req"])))
    # summaries
    summ = {}
    for j, b in enumerate(BETA_REQ):
        lab = f"{b*1e3:.0f}"
        ar = T["a_rob"][:, j]
        chk = np.array([p["auth_g1500"]["a_chk_mm"] for p in per])
        al0 = np.array([p["a_lim0_mm"] for p in per])
        alauth = np.array([p["auth_g1500"]["a_lim_mm"] for p in per])
        gs = T["g_star"][:, j]
        ga = np.array([p["auth_g1500"]["g"] for p in per])
        grid = np.array([n in GRID_RPM for n in rpm])
        bind = [per[i]["designs"][lab].get("binding", []) for i in range(len(rpm))]
        prim = [x[0] if x else "none" for x in bind]
        summ[lab] = dict(
            a_rob_mm_grid21=dict(mean=float(ar[grid].mean()), min=float(ar[grid].min()), max=float(ar[grid].max()),
                                 min_at_rpm=float(rpm[grid][np.argmin(ar[grid])])),
            a_chk_mm_grid21_mean=float(chk[grid].mean()),
            a_rob_over_a_chk_grid21=dict(mean=float(np.mean(ar[grid] / chk[grid])), min=float(np.min(ar[grid] / chk[grid])),
                                         max=float(np.max(ar[grid] / chk[grid])), median=float(np.median(ar[grid] / chk[grid]))),
            a_chk_minus_a_rob_mm_grid21=dict(mean=float(np.mean(chk[grid] - ar[grid])), max=float(np.max(chk[grid] - ar[grid])),
                                             min=float(np.min(chk[grid] - ar[grid]))),
            a_rob_below_a_chk_speeds=int(np.sum(ar[grid] < chk[grid] - 1e-9)),
            a_rob_over_a_lim0_grid21_median=float(np.median(ar[grid] / al0[grid])),
            a_rob_over_a_lim_auth_grid21_median=float(np.median(ar[grid] / alauth[grid])),
            primary_binding_counts_grid21={k: int(sum(1 for p, gflag in zip(prim, grid) if gflag and p == k))
                                           for k in ("a_phys", "linear limit", "voltage margin", "certificate", "none")},
            primary_binding_by_rpm={f"{n:.0f}": p for n, p in zip(rpm, prim)},
            g_star_by_rpm={f"{n:.0f}": (None if not np.isfinite(g) else float(g)) for n, g in zip(rpm, gs)},
            g_star_over_g_auth_grid21=dict(median=float(np.nanmedian(gs[grid] / ga[grid])),
                                           min=float(np.nanmin(gs[grid] / ga[grid])), max=float(np.nanmax(gs[grid] / ga[grid]))),
            g_star_below_g_auth_speeds=int(np.sum(gs[grid] < ga[grid] - 1e-6)),
            g_star_above_g_auth_speeds=int(np.sum(gs[grid] > ga[grid] + 1e-6)),
            corrected_down_speeds=[float(n) for n, p in zip(rpm, per) if p["designs"][lab].get("corrected_down")],
            upward_possible_speeds=[float(n) for n, p in zip(rpm, per) if p["designs"][lab].get("upward_possible")],
            voltage_violated_between_grid_depths_speeds=[float(n) for n, p in zip(rpm, per)
                                                         if p["designs"][lab].get("voltage_margin_violated_between_grid_depths")],
            a_rob_auth_gain_grid21_mean=float(T["a_rob_auth"][grid, j].mean()),
            gain_limited_by_primary_grid21={k: int(sum(1 for i in range(len(rpm)) if grid[i] and (per[i]["designs"][lab].get(
                "gain_limited_by") or ["none"])[0] == k)) for k in ("a_phys", "linear limit", "voltage margin", "certificate",
                                                                   "none")},
            gain_limited_by_by_rpm={f"{n:.0f}": (per[i]["designs"][lab].get("gain_limited_by") or ["none"])
                                    for i, n in enumerate(rpm)},
            step_entry_chatters_at_design_rpm=[float(n) for i, n in enumerate(rpm)
                                               if (per[i]["designs"][lab].get("entries_9V") or {}).get("step")
                                               and not per[i]["designs"][lab]["entries_9V"]["step"]["stable"]],
            face_entry_chatters_at_design_rpm=[float(n) for i, n in enumerate(rpm)
                                               if (per[i]["designs"][lab].get("entries_9V") or {}).get("face")
                                               and not per[i]["designs"][lab]["entries_9V"]["face"]["stable"]],
            a_rob_over_a_lim_at_g_star_grid21=dict(
                median=float(np.nanmedian(ar[grid] / T["a_lim_star"][grid, j])),
                min=float(np.nanmin(ar[grid] / T["a_lim_star"][grid, j])),
                max=float(np.nanmax(ar[grid] / T["a_lim_star"][grid, j]))))
    # chart arrays: dense stored curves
    f15 = np.load(os.path.join(RES, "feas_g1500.npz"))
    c15 = np.load(os.path.join(RES, "feas_cert_g1500.npz"))
    gm = np.load(os.path.join(RES, "gainmap_main.npz"))
    npz = dict(rpm=rpm, on_grid_125=np.array([n in GRID_RPM for n in rpm]), beta_req=np.array(BETA_REQ),
               a_lim0=np.array([p["a_lim0_mm"] for p in per]), g_auth=np.array([p["auth_g1500"]["g"] for p in per]),
               a_lim_auth=np.array([p["auth_g1500"]["a_lim_mm"] for p in per]),
               a_chk=np.array([p["auth_g1500"]["a_chk_mm"] for p in per]),
               g_auth3000=np.array([p["auth_g3000"]["g"] for p in per]),
               a_chk3000=np.array([p["auth_g3000"]["a_chk_mm"] for p in per]),
               a_pub=np.array([p["published"]["a"] for p in per]),
               a_pub_interpolated=np.array([p["published"]["source"] != "stored" for p in per]),
               binding_code_legend=np.array(["a_phys", "linear limit", "voltage margin", "certificate"]),
               dense_rpm=f15["rpm"], dense_a_lim0=f15["a_unc"], dense_g_auth=f15["g_best"], dense_a_lim_auth=c15["a_lim"],
               dense_a_chk=c15["a_cert"], dense_a_pub=gm["a_best"], dense_g_pub=gm["g_best"],
               depth_grid=np.array(DEPTHS), coarse_gains=np.array(G_COARSE), vlim=VLIM, a_phys=A_PHYS, zeta=ZETA, **T)
    # per speed and coarse gain: bbar on the full depth grid, reduced-order kick where computed
    bb = np.full((len(rpm), len(G_COARSE), len(DEPTHS)), np.nan)
    be = np.full_like(bb, np.nan)
    alg = np.full((len(rpm), len(G_COARSE)), np.nan)
    ach = np.full((len(rpm), len(G_COARSE)), np.nan)
    for i, n in enumerate(rpm):
        for k, g in enumerate(G_COARSE):
            alg[i, k] = a_lim(n, g) * 1e3
            ach[i, k] = speeds_res[n]["a_cheap"].get(str(g), np.nan)
            for m, d in enumerate(DEPTHS):
                bb[i, k, m] = bbar_of(n, g, d * 1e-3)
            s = speeds_res[n]["evals"].get(str(g))           # None: skipped by the branch and bound
            for row in (s["rows"] if s else []):
                if "beta" in row:
                    be[i, k, DEPTHS.index(row["d"])] = row["beta"]
    npz.update(coarse_bbar=bb, coarse_beta_red=be, coarse_a_lim=alg, coarse_a_cheap=ach)
    np.savez_compressed(os.path.join(HERE, "basin_aware.npz"), **npz)
    # reduced-order kicks (raw, c2.orbit bound) against the stored default-order kicks of
    # results/c4_gainmap.npz (DVF, c2.FAST, default N) at common (speed, depth, gain) points,
    # over every scan record in the cache; and the default-order recomputations against it
    G4 = np.load(os.path.join(RES, "c4_gainmap.npz"))
    g4, d4, n4 = list(G4["g1_dvf"]), list(G4["depth"]), list(G4["rpm"])
    rat, rat_ref = [], []
    for n in rpm:
        if n not in n4:
            continue
        cj = JL(f"speed_{n:.0f}.jsonl")
        for kk, v in cj.d.items():
            k = json.loads(kk)
            if k[0] == "scan" and k[2] in g4:
                for row in v["rows"]:
                    if "beta_raw" in row and row["d"] in d4:
                        ref4 = G4["kick_dvf"][n4.index(n), d4.index(row["d"]), g4.index(k[2])]
                        if ref4 > 0 and row["beta_raw"] > 0:
                            rat.append((n, row["d"], k[2], row["beta_raw"] / ref4))
            if k[0] == "ref" and k[4] == "FAST" and k[2] in g4 and k[3] in d4:
                ref4 = G4["kick_dvf"][n4.index(n), d4.index(k[3]), g4.index(k[2])]
                if ref4 > 0:
                    rat_ref.append((n, k[3], k[2], v["beta_raw"] / ref4))
    # and the reduced-order kicks against every default-order c2.FAST recomputation of the recert stage
    rat_all = []
    for n in rpm:
        cj = JL(f"speed_{n:.0f}.jsonl")
        scans = {json.loads(kk)[2]: v for kk, v in cj.d.items() if json.loads(kk)[0] == "scan"}
        for kk, v in cj.d.items():
            k = json.loads(kk)
            if k[0] == "ref" and k[4] == "FAST" and k[2] in scans:
                rw = {r["d"]: r for r in scans[k[2]]["rows"]}.get(k[3])
                if rw is not None and "beta_raw" in rw and v["beta_raw"] > 0:
                    rat_all.append((n, k[3], k[2], rw["beta_raw"] / v["beta_raw"]))
    ra = np.array([x[3] for x in rat_all])
    rr = np.array([x[3] for x in rat])
    nred_check = dict(
        points=len(rat), ratio_reduced_over_default=dict(min=float(rr.min()), max=float(rr.max()), median=float(np.median(rr)),
                                                         above_1p10=int(np.sum(rr > 1.10)), above_1p15=int(np.sum(rr > 1.15)),
                                                         below_1=int(np.sum(rr < 1.0))) if rr.size else None,
        worst=[dict(rpm=x[0], d_mm=x[1], g=x[2], ratio=x[3]) for x in sorted(rat, key=lambda y: -y[3])[:5]],
        default_recomputed_over_stored=dict(points=len(rat_ref), min=float(min(x[3] for x in rat_ref)),
                                            max=float(max(x[3] for x in rat_ref))) if rat_ref else None,
        note="ratio of the reduced-order (N = nodes_for - 8) c2.FAST kick to results/c4_gainmap.npz kick_dvf (default N, "
             "c2.FAST), both before the rescaling to the rigorous bound",
        recert_points=len(rat_all),
        recert_ratio_reduced_over_default_fast=dict(min=float(ra.min()), max=float(ra.max()), median=float(np.median(ra)),
                                                    above_1p10=int(np.sum(ra > 1.10)), above_1p15=int(np.sum(ra > 1.15)))
        if ra.size else None,
        recert_worst=[dict(rpm=x[0], d_mm=x[1], g=x[2], ratio=x[3]) for x in sorted(rat_all, key=lambda y: -y[3])[:5]])
    nconv = os.path.join(HERE, "basin_aware_nconv.json")
    out = dict(note=__doc__, protocol=dict(zeta=ZETA, vmax_V=VMAX, rho=RHO, vlim_V=VLIM, a_phys_mm=A_PHYS,
                                           depth_grid_mm=list(DEPTHS), beta_req_mmps=[b * 1e3 for b in BETA_REQ],
                                           coarse_gains=list(G_COARSE), refine_steps=list(REFINE), margin=MARGIN,
                                           n_red="nodes_for(rpm) - 8", scan_settings="c2.FAST",
                                           val_speeds=list(vspeeds), val_phases=list(VAL_PHASES), cache=PROTO),
               summary=summ, nred_vs_default=nred_check,
               per_speed={f"{p['rpm']:.0f}": p for p in per}, validation=vrows, validation_summary=vsum,
               nconv_file=os.path.basename(nconv) if os.path.exists(nconv) else None,
               runtime=dict(scan_recert_s_by_rpm={f"{n:.0f}": [speeds_res[n]["t_scan"], speeds_res[n]["t_recert"]] for n in rpm}))
    json.dump(out, open(os.path.join(HERE, "basin_aware.json"), "w"), indent=1, default=float)
    return out


def load_speeds():
    p = os.path.join(CACHE, "speeds.json")
    return {float(k): v for k, v in json.load(open(p)).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "validate", "assemble", "all"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vspeeds", default=",".join(f"{n:.0f}" for n in VAL_SPEEDS))
    ap.add_argument("--speeds", default="", help="scan: comma-separated subset of SPEEDS (for tests)")
    a = ap.parse_args()
    vspeeds = tuple(float(x) for x in a.vspeeds.split(",") if x.strip())
    t0 = time.time()
    global SPEEDS
    if a.speeds:
        SPEEDS = tuple(float(x) for x in a.speeds.split(","))
    if a.stage in ("scan", "all"):
        res = run_scan(a.workers)
        print(f"scan + recert done in {time.time() - t0:.0f} s", flush=True)
    if a.stage in ("validate", "all"):
        res = load_speeds()
        run_validate(res, a.workers, vspeeds)
        print(f"validate done at {time.time() - t0:.0f} s", flush=True)
    if a.stage in ("assemble", "all"):
        res = load_speeds()
        val = run_validate(res, a.workers, vspeeds)
        out = assemble(res, val, vspeeds)
        print(json.dumps(out["summary"], indent=1, default=float)[:8000])
        for v in out["validation"]:
            print(v)
        print(f"assemble done at {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
