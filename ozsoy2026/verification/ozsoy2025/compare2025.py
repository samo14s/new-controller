"""Model predictions against the measurements of Ozsoy et al. 2025 (second data set, same rig).

Inputs: sim2025_fig10.json and sim2025_fig6.json (sim2025.py: closed form and simulator at every
digitised test point) and the digitised measurements carried in them (points_fig10.json,
points_fig6.json; digitise_notes.md says how they were read).

Saturation tests (the article's Figs. 10 and 12; re-clamped structure; DVF):
  forced command   measured amplitude of the command (N) where not saturated, against the settled forced
                   demand of the model: the closed-form peak K_a g max|x_p'| and, as alternative readings
                   of 'amplitude', its tooth-harmonic amplitude and sqrt(2) RMS (forced_map.py); ratio,
                   correlation, and the error by depth.
  saturation       measured 'Sat' against the predicted saturation of the settled cut (closed-form
                   demand > 27 N), and against a clip reached at any time of the simulated cut (entry
                   included), face entry.
  outcome          measured chatter / marginal / stable against the simulated outcome (face entry, 9 V
                   clip, full surface memory; step entry reported too) and against the linear limit.
  mechanism        at every point the position relative to the two linear limits (below a_lim(0), between
                   a_lim(0) and a_lim(g), above a_lim(g)), with the measured outcome: chatter between the
                   limits is chatter below the linear limit of the controlled cut.
Stability lobe diagram (the article's Fig. 6; nominal structure; g = 253 V s/m): every experimental
  marker against the linear limits and the simulated outcome; the article's own simulated saturation
  onsets (open squares) against the closed-form demand reaching 27 N.

Run:  cd ozsoy2026 && python3 verification/ozsoy2025/compare2025.py   (writes compare2025.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FMAX = 27.0


def pos(a, a0, ag):
    if a < a0:
        return "below a_lim(0)"
    if a < ag:
        return "between a_lim(0) and a_lim(g)"
    return "above a_lim(g)"


def fig10():
    d = json.load(open(os.path.join(HERE, "sim2025_fig10.json")))
    rows, num = [], []
    for p in d["points"]:
        cf, fa, st = p["closed_form"], p["face"], p["step"]
        meas_sat = str(p["value"]).strip().lower() == "sat"
        meas_val = None if meas_sat else float(p["value"])
        outcome = p.get("outcome", "none") or "none"
        r = dict(rpm=p["rpm"], depth_mm=p["depth_mm"], measured=p["value"], measured_outcome=outcome,
                 position=pos(p["depth_mm"], cf["a_lim0_mm"], cf["a_lim_g_mm"]),
                 demand_N=cf["demand_N"], h1_N=cf["h1_N"], rms2_N=cf["rms2_N"], residual_authority_N=cf["residual_authority_N"],
                 pred_sat_settled=bool(cf["demand_N"] > FMAX), sim_face_chatter=fa["chatter"], sim_step_chatter=st["chatter"],
                 sim_face_run_clipped=fa["run_clipped"], sim_face_settled_clip_fraction=fa["settled_clip_fraction"],
                 sim_face_settled_demand_N=fa["settled_peak_demand_N"], linear_unstable=bool(p["depth_mm"] >= cf["a_lim_g_mm"]))
        rows.append(r)
        if meas_val is not None:
            num.append((meas_val, cf["demand_N"], cf["h1_N"], cf["rms2_N"], p["depth_mm"], p["rpm"], outcome))
    out = dict(n_points=len(rows), rows=rows)
    if num:
        m = np.array([x[0] for x in num])
        for j, name in ((1, "peak"), (2, "h1"), (3, "rms2")):
            pr = np.array([x[j] for x in num])
            ok = m > 0
            ratio = pr[ok] / m[ok]
            out[f"amplitude_{name}"] = dict(n=int(ok.sum()), median_ratio_pred_over_meas=float(np.median(ratio)),
                                            ratio_q25_q75=[float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))],
                                            corr=float(np.corrcoef(m[ok], pr[ok])[0, 1]),
                                            rms_error_N=float(np.sqrt(np.mean((pr[ok] - m[ok]) ** 2))),
                                            median_abs_error_N=float(np.median(np.abs(pr[ok] - m[ok]))))
        stable_num = [x for x in num if x[6] == "none"]
        if stable_num:
            m2 = np.array([x[0] for x in stable_num]); p2 = np.array([x[1] for x in stable_num])
            out["amplitude_peak_stable_only"] = dict(n=len(stable_num), median_ratio=float(np.median(p2 / m2)),
                                                     corr=float(np.corrcoef(m2, p2)[0, 1]))
        by_depth = {}
        for x in num:
            by_depth.setdefault(round(x[4], 1), []).append(x[1] / x[0])
        out["peak_ratio_by_depth"] = {str(k): dict(n=len(v), median=float(np.median(v))) for k, v in sorted(by_depth.items())}
    # saturation contingency
    def table(pred_key, meas_fn):
        t = dict(both=0, meas_only=0, pred_only=0, neither=0)
        for r in rows:
            ms, ps = meas_fn(r), bool(r[pred_key])
            t["both" if ms and ps else "meas_only" if ms else "pred_only" if ps else "neither"] += 1
        return t
    is_sat = lambda r: str(r["measured"]).strip().lower() == "sat"   # noqa: E731
    out["saturation_settled_closed_form"] = table("pred_sat_settled", is_sat)
    out["saturation_any_time_face_entry"] = table("sim_face_run_clipped", is_sat)
    # outcome contingency (chatter or marginal counted as not stable)
    unstable = lambda r: r["measured_outcome"] in ("chatter", "marginal")   # noqa: E731
    out["outcome_face"] = table("sim_face_chatter", unstable)
    out["outcome_step"] = table("sim_step_chatter", unstable)
    out["outcome_linear"] = table("linear_unstable", unstable)
    # mechanism: measured outcome by position relative to the linear limits
    mech = {}
    for r in rows:
        k = r["position"]
        mech.setdefault(k, dict(n=0, chatter=0, marginal=0, stable=0, sat=0))
        mech[k]["n"] += 1
        mech[k][r["measured_outcome"] if r["measured_outcome"] in ("chatter", "marginal") else "stable"] += 1
        mech[k]["sat"] += int(is_sat(r))
    out["by_position"] = mech
    return out


def fig6():
    p = os.path.join(HERE, "sim2025_fig6.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rows = []
    for q in d["points"]:
        cf, fa, st = q["closed_form"], q["face"], q["step"]
        rows.append(dict(type=q["type"], rpm=q["rpm"], depth_mm=q["depth_mm"], label=q.get("label"),
                         position=pos(q["depth_mm"], cf["a_lim0_mm"], cf["a_lim_g_mm"]), demand_N=cf["demand_N"],
                         sim_face_chatter=fa["chatter"], sim_step_chatter=st["chatter"], linear_unstable=bool(q["depth_mm"] >= cf["a_lim_g_mm"]),
                         linear_unstable_uncontrolled=bool(q["depth_mm"] >= cf["a_lim0_mm"])))
    out = dict(n_points=len(rows), rows=rows, by_type={})
    for t in sorted({r["type"] for r in rows}):
        rr = [r for r in rows if r["type"] == t]
        out["by_type"][t] = dict(n=len(rr), linear_unstable=sum(r["linear_unstable"] for r in rr),
                                 linear_unstable_uncontrolled=sum(r["linear_unstable_uncontrolled"] for r in rr),
                                 sim_face_chatter=sum(r["sim_face_chatter"] for r in rr),
                                 sim_step_chatter=sum(r["sim_step_chatter"] for r in rr),
                                 demand_over_27N=sum(r["demand_N"] > FMAX for r in rr),
                                 positions={k: sum(r["position"] == k for r in rr) for k in sorted({r["position"] for r in rr})})
    return out


def main():
    out = dict(note=__doc__.split("\n\n")[0], fig10=fig10(), fig6=fig6())
    with open(os.path.join(HERE, "compare2025.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    f10 = out["fig10"]
    for k in ("amplitude_peak", "amplitude_h1", "amplitude_rms2", "amplitude_peak_stable_only", "peak_ratio_by_depth",
              "saturation_settled_closed_form", "saturation_any_time_face_entry", "outcome_face", "outcome_step",
              "outcome_linear", "by_position"):
        print(k, json.dumps(f10.get(k)))
    if out["fig6"]:
        print("fig6 by type", json.dumps(out["fig6"]["by_type"], indent=1))


if __name__ == "__main__":
    main()
