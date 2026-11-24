"""Merge the N-convergence runs of the C2 certified kick into one table (certs_nconv.json).

Sources: certs_nconv_extra.json (intermediate N where the 2 N check failed), certs_nconv_a.json, certs_nconv_b.json, certs_nconv_c.json (certs_checks.py nconv, run in
three processes: 1090/2500 rpm, 1420 rpm, 2870 rpm; at 0.5 and 0.98 a_lim; N = c2.nodes_for(rpm) and
2 N) and revision_runs.json beta_vs_N (0.9 a_lim at 1090 and 2500 rpm, 0.3 a_lim at 2870 rpm, with
1.5 N as well; and the DVF + D comb point, kind nconv_comb).  The certified kick is the C2 size along
the velocity direction, in mm/s, with the command bound of the C2 solve (before the rescaling to the
rigorous bound b-bar, which changes it by factors 0.997-1.000 at these depths).

Run:  cd ozsoy2026 && python3 verification/paper/certs_nconv_summary.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    pts = {}
    for tag in ("a", "b", "c"):
        for k, v in load(f"certs_nconv_{tag}.json").get("points", {}).items():
            if len(v.get("beta_mmps", [])) == 2:
                pts[k] = dict(rpm=v["rpm"], frac=v["frac"], a_mm=v["a_mm"], N=v["N"], beta_mmps=v["beta_mmps"],
                              source=f"certs_nconv_{tag}.json")
    for k, v in load("certs_nconv_extra.json").get("points", {}).items():
        if k in pts:
            d = dict(zip(pts[k]["N"], pts[k]["beta_mmps"]))
            d.update({int(N): r["beta_mmps"] for N, r in v["runs"].items()})
            Ns = sorted(d)
            pts[k]["N"], pts[k]["beta_mmps"] = Ns, [d[N] for N in Ns]
            pts[k]["source"] += " + certs_nconv_extra.json"
    rr = load("revision_runs.json").get("beta_vs_N", [])
    stored = {}
    for v in rr:
        if v.get("kind") == "nconv":
            n, fr, N = v["args"]
            stored.setdefault((float(n), float(fr)), {})[int(N)] = v["beta_mmps"]
    for (n, fr), d in stored.items():
        Ns = sorted(d)
        pts[f"{n:.0f}_{fr}"] = dict(rpm=n, frac=fr, N=Ns, beta_mmps=[d[N] for N in Ns],
                                   source="revision_runs.json beta_vs_N")
    rows = []
    for k, v in sorted(pts.items(), key=lambda kv: (kv[1]["rpm"], kv[1]["frac"])):
        b = v["beta_mmps"]
        N0 = v["N"][0]
        d = dict(zip(v["N"], b))
        b2 = d.get(2 * N0)
        failed = [N for N in v["N"] if d[N] <= 0]
        passing = [N for N in v["N"] if d[N] > 0]
        rows.append(dict(key=k, rpm=v["rpm"], frac=v["frac"], N=v["N"], beta_mmps=b,
                         rel_2N=(b2 - b[0]) / b[0] if b2 is not None and b2 > 0 else None,
                         failed_N=failed, largest_passing_N=max(passing),
                         rel_largest_passing=(d[max(passing)] - b[0]) / b[0], source=v["source"]))
    ok = [r for r in rows if r["rel_2N"] is not None]
    out = dict(note=__doc__.split("\n\n")[0], points=rows,
               summary=dict(n_points=len(rows), n_with_2N=len(ok),
                            rel_2N_min=min(r["rel_2N"] for r in ok), rel_2N_max=max(r["rel_2N"] for r in ok),
                            largest_drop=min(ok, key=lambda r: r["rel_2N"])["key"],
                            largest_rise=max(ok, key=lambda r: r["rel_2N"])["key"],
                            n_drop_over_1pct=sum(r["rel_2N"] < -0.01 for r in ok),
                            points_with_failed_check=[dict(key=r["key"], failed_N=r["failed_N"],
                                                           largest_passing_N=r["largest_passing_N"],
                                                           rel_largest_passing=r["rel_largest_passing"])
                                                      for r in rows if r["failed_N"]]))
    with open(os.path.join(HERE, "certs_nconv.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(f"{r['key']:>10s} N {r['N']} beta {[round(x, 4) for x in r['beta_mmps']]} rel_2N {'failed' if r['rel_2N'] is None else f"{100 * r['rel_2N']:+.2f} %"}  ({r['source']})")
    print(out["summary"])


if __name__ == "__main__":
    main()
