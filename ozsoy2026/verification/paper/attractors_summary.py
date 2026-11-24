"""What lies outside the basin of the forced orbit: comparison with the
manuscript's band edge and the terminology the runs support (no simulation).

Purpose.  Collect, per basin speed (1090, 1420, 2500, 2870 rpm, g = 1500 V s/m,
9 V clip), what attractors_kick.py (long runs after kicks of 2-3 dv* at the 48
basin points) and attractors_bifurcation.py (depth sweep from a_lim(0) to
1.05 a_lim from four initial conditions) found, and compare it with the stored
C1 results the manuscript quotes:
  results/c1_band.npz   lower edge of the 'bistable band' (every 20 rpm; at 1090
                        and 2870 rpm, off that grid, the two neighbouring speeds
                        and the linear interpolation are given)
  results/c1_map.npz    step-entry and face-entry limits (D1, 300 revolutions)
  results/c1_basin.npz  smallest of the 12 depths with a finite critical kick
Terminology test, per speed: 'bistable' is demonstrated at a depth where the
face entry ends on the forced orbit and some other initial condition ends in
bounded, stationary chatter (a second bounded attractor); 'divergent' where some
initial condition reached the 1 cm abort; 'loss of attraction only' where some
initial condition left the forced orbit but none ended in bounded chatter.
The high branch is the clipped chatter (clipped fraction of the command >= 0.5);
bounded chatter with the command unclipped (clipped fraction < 0.5) is listed
apart as 'small_unclipped_chatter'.  Also collected: the twin-run test of
attractors_chaos.py, the band spot check of attractors_band_spot.py and its
diverged runs rerun with a 10 cm abort (attractors_abort.py).

Run (after the two campaigns, a few seconds):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/attractors_summary.py
Writes verification/paper/attractors_summary.json.
"""
import json
import os

import numpy as np

import attractors_common as AC

HERE = AC.HERE


def band_edge(rpm):
    """Lower band edge (mm) of results/c1_band.npz at rpm: value on the grid, or the
    two neighbours and the linear interpolation."""
    b = np.load(os.path.join(AC.RES, "c1_band.npz"))
    r, v, g = b["rpm"], b["a_band_low"], b["g"]
    if np.any(np.isclose(r, rpm)):
        i = int(np.nonzero(np.isclose(r, rpm))[0][0])
        return dict(on_grid=True, value_mm=float(v[i]), g=float(g[i]))
    j = int(np.searchsorted(r, rpm))
    lo, hi = j - 1, j
    t = (rpm - r[lo]) / (r[hi] - r[lo])
    return dict(on_grid=False, neighbours={f"{r[lo]:.0f}": float(v[lo]), f"{r[hi]:.0f}": float(v[hi])},
                g_neighbours=[float(g[lo]), float(g[hi])], value_mm=float(v[lo] + t * (v[hi] - v[lo])))


def main():
    kick = json.load(open(os.path.join(HERE, "attractors_kick.json")))
    bif = json.load(open(os.path.join(HERE, "attractors_bifurcation.json")))
    m = np.load(os.path.join(AC.RES, "c1_map.npz"))
    bas = np.load(os.path.join(AC.RES, "c1_basin.npz"))
    out = dict(note=__doc__, zeta=AC.ZETA, sources=dict(
        kick="verification/paper/attractors_kick.json", bifurcation="verification/paper/attractors_bifurcation.json",
        band="results/c1_band.npz", map="results/c1_map.npz", basin="results/c1_basin.npz"),
        build_kick=kick["build"], build_bifurcation=bif["build"], speeds={})
    for d in bif["speeds"]:
        n = d["rpm"]
        key = f"{n:.0f}"
        al, al0 = d["a_lim_mm"], d["a_lim0_mm"]
        i = int(np.argmin(np.abs(m["rpm"] - n)))
        assert np.isclose(m["rpm"][i], n) and abs(m["a_lim"][i] - al) < 1e-6 * al
        be = band_edge(n)
        # c1_basin: smallest depth with a finite critical kick, and the largest depth below it without
        sel = [k for k in range(len(bas["rpm"])) if np.isclose(bas["rpm"][k], n) and bas["vmax"][k] > 0]
        fin = [bas["frac"][k] for k in sel if np.any(np.isfinite(bas["dv"][k]))]
        nofin = [bas["frac"][k] for k in sel if not np.any(np.isfinite(bas["dv"][k])) and bas["frac"][k] < min(fin)]
        # bifurcation digest
        rows = d["depth_rows"]
        ics = bif["protocol"]["ics"]
        states = [[row["runs"][ic]["state"] for ic in ics] for row in rows]
        a_mm = [row["a_mm"] for row in rows]
        bist = [a for a, s in zip(a_mm, states) if s[ics.index("face")] == "forced" and "chatter" in s]
        left = [a for a, s in zip(a_mm, states) if any(x != "forced" for x in s)]
        only_loss = [a for a, s in zip(a_mm, states)
                     if any(x != "forced" for x in s) and "chatter" not in s]
        div = [a for a, s in zip(a_mm, states) if "diverged" in s]
        # clipped chatter (the high branch) and small, unclipped chatter (clipped fraction < 0.5)
        ch_runs = [row["runs"][ic] for row in rows for ic in ics
                   if row["runs"][ic]["state"] == "chatter" and row["runs"][ic]["clip_frac"] >= 0.5]
        small = [dict(a_mm=row["a_mm"], a_over_alim=row["a_over_alim"], ic=ic, pp_x_um=row["runs"][ic]["pp_x"] * 1e6,
                      clip_frac=row["runs"][ic]["clip_frac"], f_chatter=row["runs"][ic]["f_chatter"],
                      res_tooth_um=row["runs"][ic]["res_tooth"] * 1e6)
                 for row in rows for ic in ics
                 if row["runs"][ic]["state"] == "chatter" and row["runs"][ic]["clip_frac"] < 0.5]

        def rng(key2, scale=1.0, runs=ch_runs):
            v = [r[key2] * scale for r in runs if r.get(key2) is not None]
            return [float(min(v)), float(max(v))] if v else None

        ref = d["high_branch_refined_mm"]
        ref_mid = 0.5 * (ref[0] + ref[1]) if ref else None
        # the chatter run that ended the bisection at the upper side of the bracket: amplitude at the fold
        end_run = [r for r in d["refine_runs"] if r["state"] == "chatter" and ref and abs(r["a"] * 1e3 - ref[1]) < 1e-9]
        # kick campaign at this speed
        kp = [p for p in kick["points"] if np.isclose(p["rpm"], n)]
        k_out = {f"{p['frac']:.2f}": ([r["outcome"] for r in p["runs"]] if not p["skipped"] else "skipped")
                 for p in kp}
        k_b = [r for p in kp if not p["skipped"] for r in p["runs"] if r["outcome"].startswith("bounded")]
        s = dict(
            a_lim0_mm=al0, a_lim_mm=al, a_lim0_over_alim=al0 / al,
            band_edge_c1_band=be, band_edge_over_alim=be["value_mm"] / al,
            high_branch_end_grid_mm=d["high_branch_grid_end_mm"],
            high_branch_end_grid_below_mm=d["high_branch_grid_below_mm"],
            high_branch_end_refined_bracket_mm=ref, high_branch_end_refined_mm=ref_mid,
            high_branch_end_over_alim=ref_mid / al if ref_mid else None,
            high_branch_end_over_alim0=ref_mid / al0 if ref_mid else None,
            high_branch_end_minus_band_edge_mm=ref_mid - be["value_mm"] if ref_mid else None,
            high_branch_end_over_band_edge=ref_mid / be["value_mm"] if ref_mid else None,
            high_branch_holes_mm=d["high_branch_holes_mm"],
            high_branch_pp_x_at_end_mm=end_run[0]["pp_x"] * 1e3 if end_run else None,
            high_branch_end_reached_by=end_run[0]["ic"] if end_run else None,
            bistable_range_refined_mm=[ref[1], d["low_branch_top_mm"]] if ref else None,
            bistable_range_refined_over_alim=[ref[1] / al, d["low_branch_top_mm"] / al] if ref else None,
            basin_first_finite_frac=float(min(fin)) if fin else None,
            basin_first_finite_mm=float(min(fin) * al) if fin else None,
            basin_last_infinite_frac_below=float(max(nofin)) if nofin else None,
            step_entry_limit_c1_map_mm=float(m["a_step_clip"][i]),
            step_entry_bracket_bifurcation_mm=[d["step_forced_last_mm"], d["step_chatter_first_mm"]],
            face_entry_limit_c1_map_mm=float(m["a_edge_clip"][i]),
            face_low_branch_bracket_bifurcation_mm=[d["low_branch_top_mm"], d["low_branch_first_unstable_mm"]],
            bistable_demonstrated_grid_mm=bist,
            bistable_demonstrated_range_mm=[min(bist), max(bist)] if bist else None,
            bistable_demonstrated_range_over_alim=[min(bist) / al, max(bist) / al] if bist else None,
            depths_where_some_ic_left_forced_orbit_mm=left,
            loss_of_attraction_only_mm=only_loss,
            diverged_depths_mm=div, diverged_detail=d["diverged_depths_mm"],
            transient_detail=d["transient_depths_mm"],
            small_unclipped_chatter=small,
            high_branch_pp_x_mm=rng("pp_x", 1e3), high_branch_rms_V=rng("rms_V"),
            high_branch_clip_frac=rng("clip_frac"), high_branch_f_chatter_hz=rng("f_chatter"),
            d1_vs_final_disagreements=d["d1_vs_final_disagreements"],
            kick_outcomes_by_frac=k_out,
            kick_bounded_Q4_pp_x_mm=rng("pp_x", 1e3, [r["Q4"] for r in k_b]),
            kick_bounded_Q4_rms_V=rng("rms_V", 1.0, [r["Q4"] for r in k_b]),
            kick_bounded_Q4_clip_frac=rng("clip_frac", 1.0, [r["Q4"] for r in k_b]),
            kick_bounded_Q4_f_chatter_hz=rng("f_chatter", 1.0, [r["Q4"] for r in k_b]),
        )
        out["speeds"][key] = s
    chaos = json.load(open(os.path.join(HERE, "attractors_chaos.json")))
    spot = json.load(open(os.path.join(HERE, "attractors_band_spot.json")))
    abort = json.load(open(os.path.join(HERE, "attractors_abort.json")))
    out["sources"].update(chaos="verification/paper/attractors_chaos.json",
                          band_spot="verification/paper/attractors_band_spot.json",
                          abort="verification/paper/attractors_abort.json")
    ks = kick["summary"]
    allb = [sp for sp in out["speeds"].values()]
    out["overall"] = dict(
        kick_outcome_counts=ks["outcome_counts"], kick_runs=ks["n_runs"], kick_points_run=ks["n_points_run"],
        kick_points_skipped=ks["n_skipped"], kick_d1_chatter_300rev=ks["d1_chatter_300rev_count"],
        kick_bounded_Q4_pp_x_mm=ks["bounded_Q4_pp_x_mm"], kick_bounded_Q4_clip_frac=ks["bounded_Q4_clip_frac"],
        kick_bounded_Q4_f_chatter_hz=ks["bounded_Q4_f_chatter_hz"], kick_transient_peak_mm=ks["transient_peak_mm"],
        long_runs=[{k: lr.get(k) for k in ("rpm", "frac", "outcome", "rel_change_rms_x", "rel_change_rms_V",
                                           "change_f_chatter_hz")} for lr in kick["long_runs"]],
        bifurcation_state_counts=dict(zip(*np.unique(
            [r["state"] for d in bif["speeds"] for row in d["depth_rows"] for r in row["runs"].values()],
            return_counts=True))),
        bifurcation_small_unclipped_chatter={k: sp["small_unclipped_chatter"] for k, sp in out["speeds"].items()
                                             if sp["small_unclipped_chatter"]},
        bifurcation_max_x_mm=max(r["max_x_mm"] for d in bif["speeds"] for row in d["depth_rows"]
                                 for r in row["runs"].values()),
        any_divergence_bifurcation=any(sp["diverged_depths_mm"] for sp in allb),
        any_divergence_kick=ks["outcome_counts"].get("diverged", 0) > 0,
        high_branch_end_over_band_edge=[sp["high_branch_end_over_band_edge"] for sp in allb],
        bistable_range_over_alim={k: sp["bistable_demonstrated_range_over_alim"] for k, sp in out["speeds"].items()},
        chaos_verdicts=[dict(kind=c["kind"], rpm=c["rpm"], g=c["g"], a_mm=c["a_mm"], verdict=c["verdict"],
                             sep_final_over_pp=c.get("sep_final_over_pp")) for c in chaos["cases"]],
        band_spot_summary=spot["summary"],
        band_spot_by_speed={f"{p['rpm']:.0f}": dict(g=p["g"], a_lim_mm=p["a_lim_mm"], band_edge_mm=p["band_edge_mm"],
                                                     states=p["states"]) for p in spot["speeds"]},
        abort_10cm_summary=abort["summary"],
        abort_10cm_outcomes={f"{r['rpm']:.0f} rpm {r['a_mm']:.2f} mm {r['ic']}": dict(state=r["state"],
                                                                                     max_x_mm=r["max_x_mm"])
                             for r in abort["runs"]},
    )
    json.dump(AC.jsonable(out), open(os.path.join(HERE, "attractors_summary.json"), "w"), indent=1)
    for k, sp in out["speeds"].items():
        print(f"{k} rpm: a_lim0 {sp['a_lim0_mm']:.2f}, a_lim {sp['a_lim_mm']:.2f}; band edge {sp['band_edge_c1_band']['value_mm']:.2f}; "
              f"high branch end {sp['high_branch_end_refined_bracket_mm']} (grid {sp['high_branch_end_grid_mm']}); "
              f"bistable {sp['bistable_demonstrated_range_mm']}; diverged {sp['diverged_depths_mm']}; "
              f"step {sp['step_entry_limit_c1_map_mm']:.2f} vs {sp['step_entry_bracket_bifurcation_mm']}; "
              f"face {sp['face_entry_limit_c1_map_mm']:.2f} vs {sp['face_low_branch_bracket_bifurcation_mm']}")
    print(json.dumps(AC.jsonable(out["overall"]), indent=1))


if __name__ == "__main__":
    main()
