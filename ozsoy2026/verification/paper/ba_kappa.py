"""Basin-aware chart with a depth margin below the linear limit (review round 2, like-for-like comparison).

Purpose.  The reference schedules operate at most at kappa a_lim(g), kappa = 0.9 (Section 3), whereas
the basin-aware chart of Section 6 only requires a < a_lim(n, g), so that its design points lie at
0.85-0.997 of their own linear limit.  Under errors in the model the linear limit moves, so a fair
comparison of depth and of robustness needs the same margin on both sides.  This script recomputes
the chart with the admissibility of basin_aware.py plus
  (K)  a <= kappa a_lim(n, g),   kappa = 0.9,
everything else unchanged: the grid of depths 2:2:32 mm (the cap a_cap = 32 mm), the voltage margin
bbar <= 8.1 V, the reduced-order certificate >= beta_req for every grid depth up to a_rob, the same
branch-and-bound search over the gains (coarse 0:250:3000 V s/m, refinement +-125 and +-62.5 V s/m,
with the bound a_cheap replaced by min(a_cheap, largest grid depth <= kappa a_lim)), the same tie-break
and the same default-order re-certification (basin_aware.search and basin_aware.recert are called
unchanged).

Implementation.  A wrapper cache serves basin_aware's stored scans, closed-form bounds and default-order
certificates (basin_aware_cache/<proto>/speed_<rpm>.jsonl, read only) after applying (K): in a scan,
the first grid depth above kappa a_lim and every deeper one are made inadmissible (marked kappa_cut),
and a_cheap is lowered to the largest grid depth <= kappa a_lim.  New scans (gains the original
search skipped) stop at the first depth above kappa a_lim.  New results go to
basin_aware_cache/kappa_<proto>/speed_<rpm>.jsonl.  The binding constraint 'linear limit' is
reported as 'depth margin' where the failing grid depth is below a_lim but above kappa a_lim.

Run:  cd ozsoy2026 && OMP_NUM_THREADS=1 python3 verification/paper/ba_kappa.py [--workers 2] [--kappa 0.9]
Writes verification/paper/ba_kappa.json.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import basin_aware as BA  # noqa: E402

KAPPA = 0.9
OUT = os.path.join(HERE, "ba_kappa.json")


def kappa_floor(al_mm, kappa):
    """Largest grid depth <= kappa * a_lim (mm), 0 if none."""
    ok = [d for d in BA.DEPTHS if d <= kappa * al_mm]
    return max(ok) if ok else 0.0


def kappa_rows(scan, kappa):
    """The scan with (K) applied: rows above kappa a_lim are cut, the first of them failing."""
    al = scan["a_lim"]
    rows = []
    for r in scan["rows"]:
        r = dict(r)
        if r["d"] > kappa * al:
            r["lin_orig"] = bool(r.get("lin_orig", r["lin"]))
            r["lin"] = False
            r["kappa_cut"] = True
            rows.append(r)
            break
        rows.append(r)
    return dict(scan, rows=rows, kappa=kappa)


class OwnJL(BA.JL):
    """basin_aware.JL in another directory."""

    def __init__(self, directory, name):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, name)
        self.d = {}
        if os.path.exists(self.path):
            good = 0
            with open(self.path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    self.d[json.dumps(rec["k"])] = rec["v"]
                    good += len(line)
            if os.path.getsize(self.path) > good:
                with open(self.path, "r+") as fh:
                    fh.truncate(good)


class KCache:
    """Read basin_aware's cache, apply (K), write new entries to an own cache."""

    def __init__(self, rpm, kappa):
        self.rpm, self.kappa = rpm, kappa
        name = f"speed_{rpm:.0f}.jsonl"
        self.base = BA.JL(name)          # read only below (never put)
        self.own = OwnJL(os.path.join(os.path.dirname(BA.CACHE), f"kappa{kappa:g}_{BA.PROTO}"), name)
        self._al = {}

    def a_lim_mm(self, g):
        if g not in self._al:
            from zoa import a_lim
            self._al[g] = a_lim(self.rpm, g) * 1e3
        return self._al[g]

    def _view(self, k, v):
        if v is None:
            return None
        if k[0] == "scan":
            return kappa_rows(v, self.kappa)
        if k[0] == "cheap":
            return min(float(v), kappa_floor(self.a_lim_mm(float(k[2])), self.kappa))
        return v

    def get(self, k):
        v = self.own.get(k)
        if v is None:
            v = self.base.get(k)
        return self._view(k, v)

    def put(self, k, v):
        self.own.put(k, v)
        return self._view(k, v)


def scan_gain_kappa(rpm, g, kappa):
    """basin_aware.scan_gain, stopped at the first grid depth above kappa a_lim (no certificate there)."""
    from zoa import a_lim
    al = a_lim(rpm, g) * 1e3
    N = BA.n_red(rpm)
    rows = []
    for d in BA.DEPTHS:
        a = d * 1e-3
        if d > kappa * al:
            bb = BA.bbar_of(rpm, g, a)
            rows.append(dict(d=d, lin=False, lin_orig=bool(d < al), kappa_cut=True, bbar=bb, volt=bool(bb <= BA.VLIM)))
            break
        lin = bool(d < al)
        bb = BA.bbar_of(rpm, g, a)
        row = dict(d=d, lin=lin, bbar=bb, volt=bool(bb <= BA.VLIM))
        if lin and bb < BA.VMAX:
            row.update(BA.certificate(rpm, g, a, N, "FAST", bb))
        rows.append(row)
        if not (lin and row["volt"] and row.get("beta", 0.0) >= min(BA.BETA_REQ)):
            break
    return dict(rpm=rpm, g=g, a_lim=al, N=N, rows=rows)


def speed_job(args):
    rpm, kappa = args
    BA._setup()
    t0 = time.time()
    BA.scan_gain = lambda r, g: scan_gain_kappa(r, g, kappa)      # used by basin_aware.search for new gains
    cache = KCache(rpm, kappa)
    g_auth = float(round(cache.base.get(["compare", rpm])["g1500"]["g"], 6))
    evals, cheap, skipped = BA.search(rpm, cache, g_auth)
    designs = BA.recert(rpm, evals, cache)
    out = {}
    for b, des in designs.items():
        des = dict(des)
        nr = des.get("next_row") or {}
        if nr.get("kappa_cut") and nr.get("lin_orig") and "linear limit" in des.get("binding", []):
            des["binding"] = ["depth margin" if x == "linear limit" else x for x in des["binding"]]
        al = des.get("a_lim")
        des["a_rob_over_a_lim"] = (des["a_rob"] / al) if al else None
        des.pop("next_row", None)
        out[b] = des
    return dict(rpm=rpm, kappa=kappa, designs=out, n_evaluated=len(evals), skipped=skipped,
                evaluated={str(g): dict(a_lim=s["a_lim"], a_rob={str(b): BA.evaluate(s["rows"], b)[0] for b in BA.BETA_REQ})
                           for g, s in sorted(evals.items())},
                seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    a = ap.parse_args()
    speeds = sorted(BA.SPEEDS, key=lambda n: (-BA.n_red(n), n))
    res = {}
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(speed_job, [(n, a.kappa) for n in speeds]):
            res[f"{r['rpm']:.0f}"] = r
            d = r["designs"]
            print(f"{r['rpm']:.0f} rpm: " + ", ".join(f"{float(b) * 1e3:g} mm/s g {d[b]['g']} a_rob {d[b]['a_rob']}"
                                                         f" {d[b].get('binding')}" for b in sorted(d)) +
                  f" ({r['n_evaluated']} gains, {r['seconds']} s)", flush=True)
    ba = json.load(open(os.path.join(HERE, "basin_aware.json")))
    out = dict(note=__doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1].split("\n")[0], kappa=a.kappa,
               protocol=dict(base_cache=BA.CACHE, depths=list(BA.DEPTHS), v_lim=BA.VLIM, beta_req=list(BA.BETA_REQ)),
               speeds={k: res[k] for k in sorted(res, key=float)})
    summ = {}
    for b in BA.BETA_REQ:
        key = str(b)
        rows = [out["speeds"][k]["designs"][key] for k in out["speeds"]]
        base = [ba["per_speed"][k]["designs"][f"{b * 1e3:g}"] for k in out["speeds"]]
        summ[f"{b * 1e3:g}"] = dict(
            mean_a_rob_mm=float(np.mean([r["a_rob"] for r in rows])),
            mean_a_rob_kappa1_mm=float(np.mean([r["a_rob_mm"] for r in base])),
            median_a_rob_over_a_lim=float(np.median([r["a_rob_over_a_lim"] for r in rows if r["a_rob_over_a_lim"]])),
            binding_counts={x: sum(1 for r in rows if (r.get("binding") or ["none"])[0] == x)
                            for x in sorted({(r.get("binding") or ["none"])[0] for r in rows})},
            corrected_down=[k for k in out["speeds"] if out["speeds"][k]["designs"][key].get("corrected")],
            upward_possible=[k for k in out["speeds"] if out["speeds"][k]["designs"][key].get("upward_possible")])
    out["summary"] = summ
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
