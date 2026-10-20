"""Headline numbers of C4, computed from the stored results.

    python c4_summary.py      -> ../results/c4_summary.json
The report quotes these numbers; tests/test_c4_results.py pins them.
"""
import json
import os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VMAX = 9.0


def r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def stats(x, nd=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(min=r(x.min(), nd), median=r(np.median(x), nd), max=r(x.max(), nd), n=int(x.size)) if x.size else None


def runout_tolerance(cmd, runout):
    """Largest runout (um) with the worst orbit command below the limit
    (linear interpolation on the sweep; capped at the sweep's end)."""
    if cmd[0] >= VMAX:
        return 0.0
    if cmd[-1] < VMAX:
        return float(runout[-1] * 1e6)
    return float(np.interp(VMAX, cmd, runout) * 1e6)


def build():
    L, E, G, V, R = (np.load(os.path.join(RES, n)) for n in
                     ("c4_limits.npz", "c4_entry.npz", "c4_gainmap.npz", "c4_validate.npz", "c4_robust.npz"))
    out = {}
    # ---- linear limits
    names = list(L["feas_names"])
    af, gf = L["feas_a_lim"], L["feas_g1"]
    dvf = af[:, names.index("dvf")]
    lim_auth = gf[:, names.index("dvf")] < L["g1_max"] - 1          # DVF gain held below the bound by authority
    out["limits"] = dict(
        speeds=int(len(L["rpm"])), g1_max=float(L["g1_max"]),
        dvf_authority_limited_speeds=int(lim_auth.sum()),
        paper_gain_ratio={f"{g2:.0f}": stats(L["a_lim"][:, j] / L["a_lim"][:, 0]) for j, g2 in enumerate(L["g2"])},
        feasible={n: dict(g2=float(L["feas_g2"][j]), kd=float(L["feas_kd"][j]), a_lim_mm=stats(af[:, j]),
                          ratio_to_dvf=stats(af[:, j] / dvf),
                          ratio_authority_limited=stats(af[lim_auth, j] / dvf[lim_auth]),
                          ratio_gain_bounded=stats(af[~lim_auth, j] / dvf[~lim_auth]))
                  for j, n in enumerate(names)},
        paper_orbit_saturated_fraction=r(np.mean(L["bmax"] >= VMAX), 4))
    if "uncapped_a_lim" in L.files:
        ua = L["uncapped_a_lim"]
        m = np.isfinite(ua[:, 0])
        out["limits"]["uncapped"] = dict(speeds=int(m.sum()), dvf_g1=stats(L["uncapped_g1"][m, 0], 0),
                                         dvf_a_lim_mm=stats(ua[m, 0]), d2000_a_lim_mm=stats(ua[m, 1]),
                                         d2000_over_dvf=stats(ua[m, 1] / ua[m, 0]))
    # ---- simulated entries
    des, ent = list(E["designs"]), list(E["names"])
    A = E["a_entry"]
    e = {}
    for k, name in enumerate(ent):
        e[name] = {d: dict(mm=stats(A[:, j, k]), over_a_lim=stats(A[:, j, k] / E["a_lim"][:, j], 3))
                   for j, d in enumerate(des)}
        e[name]["d2000_over_dvf"] = stats(A[:, des.index("feas_d2000"), k] / A[:, des.index("feas_dvf"), k])
    out["entry"] = dict(speeds=int(len(E["rpm"])), **e)
    # ---- certified gain map
    kick = G["kick"] * 1e3
    kdvf = G["kick_dvf"] * 1e3 if "kick_dvf" in G.files else kick[:, :, :, 0]
    g2 = np.asarray(G["g2"])
    gm = {}
    for i, n in enumerate(G["rpm"]):
        d_best = np.maximum(kick[i, :, :, 0].max(axis=1), kdvf[i].max(axis=1))   # DVF: coarse and fine g1 grid
        c_best = kick[i, :, :, 1:].reshape(len(G["depth"]), -1).max(axis=1)
        c_2000 = kick[i][:, :, (g2 > 0) & (g2 <= 2000)].reshape(len(G["depth"]), -1).max(axis=1)
        both = (d_best > 0) & (c_best > 0)
        reach, cont = {}, {}
        for D in (1.0, 2.0, 4.0):
            for lab, y in (("dvf", d_best), ("comb", c_best), ("comb_g2le2000", c_2000)):
                ok = np.where(y >= D)[0]
                reach[f"{lab}_{D:g}"] = float(G["depth"][ok].max()) if ok.size else None
                # every grid depth up to this one certified at >= D (no gap)
                bad = np.where(y < D)[0]
                last = (bad[0] - 1) if bad.size else len(y) - 1
                cont[f"{lab}_{D:g}"] = float(G["depth"][last]) if last >= 0 else None
        gm[f"{n:.0f}"] = dict(deepest_certified_mm=reach, certified_without_gap_mm=cont, comb_over_dvf=stats(c_best[both] / d_best[both]),
                              dvf_last_mm=float(G["depth"][d_best > 0].max()) if (d_best > 0).any() else None,
                              comb_last_mm=float(G["depth"][c_best > 0].max()) if (c_best > 0).any() else None)
    out["gainmap"] = dict(points=int(kick.size + (kdvf.size if "kick_dvf" in G.files else 0)),
                          checked=int(np.sum(kick > 0) + (np.sum(kdvf > 0) if "kick_dvf" in G.files else 0)),
                          dvf_g1_grid=int(kdvf.shape[-1]), by_rpm=gm)
    # ---- validation
    kind = V["kind"]
    comb = kind == "comb"
    cert, td, mk = V["cert"], V["td_min"], V["model_kick"]
    okt = np.isfinite(td) & (cert > 0)
    okm = np.isfinite(mk) & (cert > 0)
    pairs = []
    for n in sorted(set(V["rpm"].tolist())):
        for d in sorted(set(V["depth"].tolist())):
            sd = (V["rpm"] == n) & (V["depth"] == d)
            if np.sum(sd & ~comb) == 1 and np.sum(sd & comb) == 1:
                pairs.append((n, d, float(td[sd & ~comb][0]), float(td[sd & comb][0]),
                              float(mk[sd & ~comb][0]), float(mk[sd & comb][0]),
                              float(cert[sd & ~comb][0]), float(cert[sd & comb][0])))
    pr = np.array(pairs) if pairs else np.zeros((0, 8))
    np.seterr(invalid="ignore")                   # inf / inf (no divergence up to 3 m/s) -> nan, dropped
    fin = np.isfinite(pr[:, 2]) & np.isfinite(pr[:, 3]) & (pr[:, 2] > 0) if len(pr) else np.zeros(0, bool)
    out["validate"] = dict(
        points=int(len(cert)), unsettled=int(V["unsettled"].sum()),
        cert_below_td=int(np.sum(cert[okt] <= td[okt])), td_points=int(okt.sum()),
        cert_below_model=int(np.sum(cert[okm] <= mk[okm])), model_points=int(okm.sum()),
        td_over_cert=stats(td[okt] / cert[okt]), proof_vmax=r(np.nanmax(V["proof_vmax"]), 4),
        offaxis_bad=int(V["offaxis_bad"][comb].sum()), offaxis_runs=int(V["offaxis_runs"][comb].sum()),
        paired_depths=int(len(pr)),
        td_comb_over_dvf=stats(pr[fin, 3] / pr[fin, 2]) if fin.any() else None,
        model_comb_over_dvf=stats(pr[:, 5] / np.where(pr[:, 4] > 0, pr[:, 4], np.nan)) if len(pr) else None,
        cert_comb_over_dvf=stats(pr[:, 7] / pr[:, 6]) if len(pr) else None,
        td_comb_better=int(np.sum(pr[fin, 3] > pr[fin, 2])) if fin.any() else 0,
        td_pairs=int(fin.sum()))
    # ---- supplementary validation (fine-grid DVF, same-g1 delayed design, deep points)
    f2 = os.path.join(RES, "c4_validate2.npz")
    if os.path.exists(f2):
        V2 = np.load(f2)
        k2 = V2["kind"]
        allc = np.concatenate([V["cert"], V2["cert"]])
        allt = np.concatenate([V["td_min"], V2["td_min"]])
        allm = np.concatenate([V["model_kick"], V2["model_kick"]])
        ok_t = np.isfinite(allt) & (allc > 0)
        ok_m = np.isfinite(allm) & (allc > 0)

        def at(arr_kind, R_, kind, n, d):
            m = (R_["rpm"] == n) & (R_["depth"] == d) & (arr_kind == kind)
            return float(R_["td_min"][m][0]) if m.any() else None
        rows = []
        for n in sorted(set(V["rpm"].tolist()) | set(V2["rpm"].tolist())):
            for d in sorted(set(V["depth"].tolist()) | set(V2["depth"].tolist())):
                dv = at(k2, V2, "dvf_fine", n, d)
                dv = at(V["kind"], V, "dvf", n, d) if dv is None else dv
                cb = at(V["kind"], V, "comb", n, d)
                cb = at(k2, V2, "comb", n, d) if cb is None else cb
                sg = at(k2, V2, "same_g1", n, d)
                rows.append((n, d, np.nan if dv is None else dv, np.nan if cb is None else cb,
                             np.nan if sg is None else sg))
        rw = np.array(rows)
        fc = np.isfinite(rw[:, 2]) & np.isfinite(rw[:, 3]) & (rw[:, 2] > 0)
        fs = np.isfinite(rw[:, 2]) & np.isfinite(rw[:, 4]) & (rw[:, 2] > 0)
        deep = (V2["kind"] == "comb") & np.isin(V2["depth"], (30.0, 34.0))
        okd = deep & np.isfinite(V2["td_min"]) & (V2["cert"] > 0)
        out["validate2"] = dict(
            points=int(len(k2)), kinds={k: int(np.sum(k2 == k)) for k in ("dvf_fine", "same_g1", "comb")},
            unsettled=int(V2["unsettled"].sum()),
            all_cert_below_td=int(np.sum(allc[ok_t] <= allt[ok_t])), all_td_points=int(ok_t.sum()),
            all_cert_below_model=int(np.sum(allc[ok_m] <= allm[ok_m])), all_model_points=int(ok_m.sum()),
            all_td_over_cert=stats(allt[ok_t] / allc[ok_t]),
            offaxis_bad=int(V2["offaxis_bad"].sum()), offaxis_runs=int(V2["offaxis_runs"].sum()),
            deep_td_over_cert=stats(V2["td_min"][okd] / V2["cert"][okd]), deep_points=int(okd.sum()),
            comb_over_dvf=stats(rw[fc, 3] / rw[fc, 2]), comb_better=int(np.sum(rw[fc, 3] > rw[fc, 2])),
            same_g1_over_dvf=stats(rw[fs, 4] / rw[fs, 2]), same_g1_better=int(np.sum(rw[fs, 4] > rw[fs, 2])),
            pairs_mmps=[[r(x * 1e3, 2) if i > 1 else float(x) for i, x in enumerate(row)] for row in rw])  # rpm, mm, DVF, best delayed, same-g1 delayed
    # ---- robustness
    labels = list(R["labels"])
    rob = {}
    for i, n in enumerate(R["rpm"]):
        row = {}
        for j, lab in enumerate(labels):
            d = dict(g1=r(R["g1"][i, j], 0), g2=r(R["g2"][i, j], 0), a_lim_mm=r(R["a_lim"][i, j]),
                     runout_tolerance_um=r(runout_tolerance(R["runout_cmd"][i, j], R["runout"])),
                     tau_a_lim_mm={("one_mode" if np.isnan(m[0]) else f"mode_{m[0]:.0f}Hz"):
                                   [r(x) for x in R["tau_a_lim"][i, j, k]] for k, m in enumerate(R["mode2"])})
            if R["g2"][i, j] > 0:
                d["eps_a_lim_mm"] = [r(x) for x in R["eps_a_lim"][i, j]]
                d["eps_worst_cmd_V"] = [r(x) for x in R["eps_cmd"][i, j]]
            row[lab] = d
        rob[f"{n:.0f}"] = row
    tdo = R["td_orbit"]
    tdl = R["td_limit"]
    out["robust"] = dict(eps=R["eps"].tolist(), runout_um=(R["runout"] * 1e6).tolist(), tau_us=(R["tau"] * 1e6).tolist(),
                         by_rpm=rob,
                         td_orbit_ratio=stats(tdo[:, 0] / tdo[:, 1], 3), td_limit_ratio=stats(tdl[:, 0] / tdl[:, 1], 4),
                         td_face_eps={f"{e:+.2f}": [r(a), r(b)] for e, (a, b) in zip(R["td_face_eps"], R["td_face"])})
    return out


def main():
    out = build()
    json.dump(out, open(os.path.join(RES, "c4_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
