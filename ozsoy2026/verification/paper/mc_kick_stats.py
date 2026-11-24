"""Statistics of the simulated Monte Carlo of the critical kick (mc_kick.json) as quoted in the paper.

NOT A ROBUSTNESS GUARANTEE (see mc_kick.py): 25 random samples per speed bound nothing.

For each speed: the number of samples, how many did not settle before the kick ("unsettled": the
cut chattered without a kick) and why (closed-form linear instability at the sampled parameters,
forced command at the sampled clip), how many showed no chatter for any kick up to 3 m/s ("inf"),
and over the samples with a finite worst-phase critical kick ("finite"): range, median, ratios to
the nominal value, counts below the nominal value and below half of it, and Spearman rank
correlations of the critical kick with the margin to the sampled linear limit (a_lim - a, closed
form) and with the residual authority u0 of the sampled forced orbit (worst runout phase and the
sample's own runout phase).  Clopper-Pearson 95 % upper bound on the probability of "unsettled".

Run:  cd ozsoy2026 && python3 verification/paper/mc_kick_stats.py   (reads mc_kick.json, writes mc_kick_stats.json)
"""
import json
import os

import numpy as np
from scipy.stats import beta, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def main():
    d = json.load(open(os.path.join(HERE, "mc_kick.json")))
    nom = {r: d["nominal"][r]["worst4_rev_mmps"] for r in d["nominal"]}
    by = {}
    for s in d["samples"]:
        by.setdefault(f"{s['rpm']:.0f}", []).append(s)
    out = dict(note=("Statistics of verification/paper/mc_kick.json; not a robustness guarantee. "
                     "finite = samples with a finite worst-phase critical kick (4 phases); "
                     "margin = closed-form a_lim of the sample minus the depth (0.9 nominal a_lim)."),
               speeds={})
    tot = dict(n=0, unsettled=0, inf=0, finite=0, below_half=0)
    for r, L in sorted(by.items(), key=lambda kv: float(kv[0])):
        F = [s for s in L if s["worst4_kind"] == "exact"]
        U = [s for s in L if s["worst4_kind"] == "unsettled"]
        k = np.array([s["worst4_mmps"] for s in F])
        m = np.array([s["cf"]["a_lim_mm"] - s["a_mm"] for s in F])

        def rho(key):
            u = np.array([np.nan if s["cf"][key] is None else s["cf"][key] for s in F])
            ok = np.isfinite(u)
            return float(spearmanr(k[ok], u[ok])[0]), int(ok.sum())

        rw, nw = rho("u0_worst_psi_V")
        rs, ns = rho("u0_sample_psi_V")
        reasons = []
        for s in U:
            cf = s["cf"]
            if not cf["stable"]:
                reasons.append("linearly unstable (closed form)")
            elif cf["u0_worst_psi_V"] is not None and cf["u0_worst_psi_V"] <= 0:
                reasons.append("forced command at the clip (closed form)")
            else:
                reasons.append("other")
        rec = dict(n=len(L), a_mm=L[0]["a_mm"], nominal_mmps=nom[r],
                   unsettled=len(U), unsettled_reasons=reasons,
                   unsettled_pre_kick_clipped=[bool(s["pre_kick_clipped"]) for s in U],
                   inf=sum(s["worst4_kind"] == "inf" for s in L), finite=len(F),
                   finite_min_mmps=float(k.min()), finite_max_mmps=float(k.max()),
                   finite_median_mmps=float(np.median(k)),
                   ratio_min=float(k.min() / nom[r]), ratio_max=float(k.max() / nom[r]),
                   ratio_median=float(np.median(k) / nom[r]),
                   below_nominal=int((k < nom[r]).sum()), below_half_nominal=int((k < 0.5 * nom[r]).sum()),
                   spearman_kick_vs_alim_margin=float(spearmanr(k, m)[0]),
                   spearman_kick_vs_u0_worst_psi=rw, n_u0_worst_psi=nw,
                   spearman_kick_vs_u0_sample_psi=rs, n_u0_sample_psi=ns,
                   cp95_upper_unsettled=cp_upper(len(U), len(L)),
                   cf_unstable_but_settled=[s["i"] for s in L if not s["cf"]["stable"] and s["worst4_kind"] != "unsettled"])
        out["speeds"][r] = rec
        tot["n"] += rec["n"]
        tot["unsettled"] += rec["unsettled"]
        tot["inf"] += rec["inf"]
        tot["finite"] += rec["finite"]
        tot["below_half"] += rec["below_half_nominal"]
    S = out["speeds"].values()
    out["all"] = dict(tot, ratio_min=min(v["ratio_min"] for v in S), ratio_max=max(v["ratio_max"] for v in S),
                      ratio_median_range=[min(v["ratio_median"] for v in S), max(v["ratio_median"] for v in S)],
                      spearman_margin_range=[min(v["spearman_kick_vs_alim_margin"] for v in S),
                                             max(v["spearman_kick_vs_alim_margin"] for v in S)],
                      spearman_u0_worst_range=[min(v["spearman_kick_vs_u0_worst_psi"] for v in S),
                                               max(v["spearman_kick_vs_u0_worst_psi"] for v in S)],
                      cp95_upper_unsettled_range=[min(v["cp95_upper_unsettled"] for v in S),
                                                  max(v["cp95_upper_unsettled"] for v in S)])
    with open(os.path.join(HERE, "mc_kick_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["all"], indent=1))
    for r, v in out["speeds"].items():
        print(r, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})


if __name__ == "__main__":
    main()
