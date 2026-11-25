"""Tables of the Monte Carlo comparison of the basin-aware chart with the reference (review round 2, 4.2).

Purpose.  mc_designs.py simulated, on the same 30 parameter samples per speed, the reference design
(authority rule with the 10 % voltage margin at its checked depth) and the basin-aware chart, with the
depth margin a <= 0.9 a_lim (mc_designs_kappa.json: own depth and common depth) and without it
(mc_designs_chart.json: own depth).  This script collects, per speed and beta_req, the depths, the
ratios a / a_lim, the failures out of n and their breakdown (unsettled: the cut does not settle before
the kick; kick: a kick of beta_req ends in chatter), and pools them over the six speeds, with the
paired statistics computed by mc_designs.py (McNemar, sign test over the samples).  It computes no
new statistic.  Both files must be complete.

Run:  cd ozsoy2026 && python3 verification/paper/mc_designs_table.py   (writes mc_designs_table.json)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = ("1", "2", "5")


def main():
    kap = json.load(open(os.path.join(HERE, "mc_designs_kappa.json")))
    cha = json.load(open(os.path.join(HERE, "mc_designs_chart.json")))
    for name, d in (("kappa", kap), ("chart", cha)):
        if not d.get("complete"):
            raise SystemExit(f"mc_designs_{name}.json is not complete: {d.get('progress')}")
    speeds = sorted(kap["per_speed"], key=float)
    rows = []
    for k in speeds:
        for b in BETAS:
            pk = kap["per_speed"][k]["b"][b]
            pc = cha["per_speed"][k]["b"][b]
            own, com, ch = pk["own"], pk["common"], pc["own"]
            assert own["samples_used"] == ch["samples_used"] == com["samples_used"]
            assert abs(own["ref"]["a_mm"] - ch["ref"]["a_mm"]) < 1e-9
            rk = kap["per_speed"][k]
            rows.append(dict(
                rpm=float(k), beta_req=float(b), n=own["ref"]["n"],
                ref=dict(g=own["ref"]["g"], a_mm=own["ref"]["a_mm"], a_over_a_lim=rk["ref"]["a_over_a_lim"],
                         fail=own["ref"]["failures"], unsettled=own["ref"]["unsettled"], kick=own["ref"]["kick"]),
                ba_kappa=dict(g=own["BA"]["g"], a_mm=own["BA"]["a_mm"], a_over_a_lim=rk["BA"][b]["a_over_a_lim"],
                              fail=own["BA"]["failures"], unsettled=own["BA"]["unsettled"], kick=own["BA"]["kick"]),
                ba_chart=dict(g=ch["BA"]["g"], a_mm=ch["BA"]["a_mm"],
                              a_over_a_lim=cha["per_speed"][k]["BA"][b]["a_over_a_lim"],
                              fail=ch["BA"]["failures"], unsettled=ch["BA"]["unsettled"], kick=ch["BA"]["kick"]),
                common=dict(a_mm=com["ref"]["a_mm"], ref_fail=com["ref"]["failures"], ba_fail=com["BA"]["failures"],
                            ref_kick=com["ref"]["kick"], ba_kick=com["BA"]["kick"]),
                paired_own_kappa=own["paired"], paired_common=com["paired"], paired_own_chart=ch["paired"]))
    pooled = {}
    for b in BETAS:
        sk, sc = kap["summary"][b], cha["summary"][b]
        pooled[b] = dict(
            own_kappa=dict(ref=sk["own"]["ref"], BA=sk["own"]["BA"], paired=sk["own"]["paired"],
                           sign=sk["own"]["sign_test_over_samples"]),
            common=dict(ref=sk["common"]["ref"], BA=sk["common"]["BA"], paired=sk["common"]["paired"],
                        sign=sk["common"]["sign_test_over_samples"]),
            own_chart=dict(ref=sc["own"]["ref"], BA=sc["own"]["BA"], paired=sc["own"]["paired"],
                           sign=sc["own"]["sign_test_over_samples"]))
    out = dict(note=__doc__.split("\n\n")[0], rows=rows, pooled=pooled)
    with open(os.path.join(HERE, "mc_designs_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print(f"{r['rpm']:5.0f} {r['beta_req']:3.0f}  ref {r['ref']['a_mm']:5.1f} ({r['ref']['a_over_a_lim']:.2f}) "
              f"{r['ref']['fail']:2d}/{r['n']}  BAk {r['ba_kappa']['a_mm']:5.1f} ({r['ba_kappa']['a_over_a_lim']:.2f}) "
              f"{r['ba_kappa']['fail']:2d}  BA {r['ba_chart']['a_mm']:5.1f} ({r['ba_chart']['a_over_a_lim']:.2f}) "
              f"{r['ba_chart']['fail']:2d}  common {r['common']['a_mm']:5.1f} {r['common']['ref_fail']:2d}/{r['common']['ba_fail']:2d}"
              f"  kicks ref {r['ref']['kick']} BAk {r['ba_kappa']['kick']} BA {r['ba_chart']['kick']}")
    for b, p in pooled.items():
        for c in ("own_kappa", "common", "own_chart"):
            q = p[c]
            print(f"b {b} {c:9s} ref {q['ref']['failures']}/{q['ref']['n']} ({q['ref']['P_fail']:.3f} "
                  f"[{q['ref']['cp95'][0]:.3f}, {q['ref']['cp95'][1]:.3f}]; unsettled {q['ref']['unsettled']}, kick {q['ref']['kick']})"
                  f"  BA {q['BA']['failures']}/{q['BA']['n']} ({q['BA']['P_fail']:.3f} "
                  f"[{q['BA']['cp95'][0]:.3f}, {q['BA']['cp95'][1]:.3f}]; unsettled {q['BA']['unsettled']}, kick {q['BA']['kick']})"
                  f"  McNemar p {q['paired']['mcnemar_exact_p_pooled']:.3g} (ref only {q['paired']['ref_fails_BA_passes']}, "
                  f"BA only {q['paired']['BA_fails_ref_passes']})  sign p {q['sign']['p_exact_two_sided']:.3g}")


if __name__ == "__main__":
    main()
