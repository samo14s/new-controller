"""Numbers of Table 6 of the manuscript (basin-aware chart against the reference schedules).

Purpose.  Table 6 compares, at the 21 speeds 500:125:3000 rpm, the reference designs (published
condition, authority rule with no and with 10 % voltage margin, the former g <= 1500 schedule) with the
basin-aware chart (beta_req = 1, 2, 5 mm/s) and with the chart under the depth margin a <= 0.9 a_lim.
This script collects every entry from the stored results; it computes nothing new.

Columns (per design; depths are means, kicks and voltages medians over the speeds, with the minimum or
maximum in brackets):
  depth        operating depth: a_chk = min(0.9 a_lim, a_TD) for the references (fair_designs.json),
               a_rob for the chart (basin_aware.json) and its margin variant (ba_kappa.json);
  step         the largest depth up to which the step entry from rest also settles: a_chk for the
               references (a_chk contains the step-entry check), ba_entry.json / ba_entry_kappa.json
               (mean_a_step_mm) for the chart;
  certifiable  the depth up to which the design's own gain meets the chart's conditions for 1, 2 and
               5 mm/s (ref_under_ba.json; ref_under_ba_kappa0.9.json with the depth margin);
  bbar         rigorous bound on the forced command at the operating depth, V (fair_compare.json
               summary bbar_V; bak_crit.json for the margin variant);
  beta         certified kick (C2, default collocation order) at the operating depth, mm/s
               (fair_compare.json summary beta_op_mmps; ba_kappa.json designs[b].beta_ref_at_a_rob);
  crit         simulated critical kick at the operating depth, smallest over the phases 0, 4, 8, 12
               of 16, mm/s (fair_compare.json summary crit_kick_mmps; bak_crit.json): a phase whose
               unkicked cut does not settle counts as 0, no chatter up to 3 m/s as infinity (both
               enter the median as such).

Run:  cd ozsoy2026 && python3 verification/paper/table6.py   (writes verification/paper/table6.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = ("pub", "auth0", "auth10", "auth1500")
BAS = ("ba1", "ba2", "ba5")
BKEY = {"ba1": "0.001", "ba2": "0.002", "ba5": "0.005"}


def load(name):
    return json.load(open(os.path.join(HERE, name)))


def med_min(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), min=float(np.min(x)), n=int(x.size), n_inf=int(np.isinf(x).sum()))


def med_max(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), max=float(np.max(x)), n=int(x.size))


def main():
    fd = load("fair_designs.json")
    fc = load("fair_compare.json")
    ru = load("ref_under_ba.json")
    ruk = load("ref_under_ba_kappa0.9.json")
    be = load("ba_entry.json")
    bek = load("ba_entry_kappa.json")
    bk = load("ba_kappa.json")
    bc = load("bak_crit.json")
    fs = fc["summary"]
    out = dict(note=__doc__.split("\n\n")[0], rows={})
    for d in REFS + BAS:
        s = fs[d]
        row = dict(depth_mean_mm=s["a_op_mm"]["mean"], bbar_V=dict(median=s["bbar_V"]["median"], max=s["bbar_V"]["max"]),
                   beta_mmps=dict(median=s["beta_op_mmps"]["median"], min=s["beta_op_mmps"]["min"]),
                   crit_mmps=dict(median=s["crit_kick_mmps"]["median"], min=s["crit_kick_mmps"]["min"],
                                  n_inf=s["crit_kick_inf"], n_unsettled=s["crit_kick_unsettled"]),
                   step_entry_stable=s["step_entry_stable"], face_entry_stable=s["face_entry_stable"],
                   n_speeds=s["n_speeds"])
        if d in REFS:
            row["step_mean_mm"] = fd["summary"]["mean_a_chk_mm"][d]
        out["rows"][d] = row
    out["certifiable"] = dict(kappa1=ru.get("summary"), kappa09=ruk.get("summary"))
    out["step_ba"] = {b: be["summary"][b]["mean_a_step_mm"] for b in be["summary"] if isinstance(be["summary"][b], dict)
                      and "mean_a_step_mm" in be["summary"][b]}
    out["step_ba_kappa"] = {b: bek["summary"][b]["mean_a_step_mm"] for b in bek["summary"]
                            if isinstance(bek["summary"][b], dict) and "mean_a_step_mm" in bek["summary"][b]}
    # margin variant: certified kick from ba_kappa.json, critical kick and bbar from bak_crit.json
    kap = {}
    for d in BAS:
        b = BKEY[d]
        beta, crit, bbar, depth = [], [], [], []
        for k, v in bk["speeds"].items():
            z = v["designs"][b]
            depth.append(float(z["a_rob"]))
            beta.append(float(z["beta_ref_at_a_rob"]) * 1e3)
            r = bc["speeds"][f"{float(k):.0f}"][b]
            kind = r["crit_kind"]
            crit.append(np.inf if kind == "inf" else (0.0 if kind == "unsettled" else float(r["crit_mmps"])))
            bbar.append(float(r["bbar_V"]))
        kap[d] = dict(depth_mean_mm=float(np.mean(depth)), beta_mmps=med_min(beta), crit_mmps=med_min(crit),
                      bbar_V=med_max(bbar), n_speeds=len(depth))
    out["rows_kappa"] = kap
    with open(os.path.join(HERE, "table6.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for d, r in list(out["rows"].items()) + [(f"{d}_kappa", r) for d, r in kap.items()]:
        print(f"{d:12s} depth {r['depth_mean_mm']:6.2f}  bbar {r['bbar_V']['median']:5.2f} (max {r['bbar_V']['max']:5.2f})  "
              f"beta {r['beta_mmps']['median']:6.2f} (min {r['beta_mmps']['min']:6.2f})  "
              f"crit {r['crit_mmps']['median']} (min {r['crit_mmps']['min']})")


if __name__ == "__main__":
    main()
