"""Headline numbers of C2, computed from the stored results.

    python c2_summary.py      -> ../results/c2_summary.json
The report quotes these numbers; tests/test_c2_results.py pins them.
"""
import json
import os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def main():
    v, s, f = (np.load(os.path.join(RES, n)) for n in ("c2_validate.npz", "c2_step.npz", "c2_surface.npz"))
    m = np.load(os.path.join(RES, "c1_map.npz"))
    out = {}
    fin_m = np.isfinite(v["model_kick"]) & (v["cert"] > 0)
    fin_t = np.isfinite(v["td_min"]) & (v["cert"] > 0)
    both = np.isfinite(v["model_kick"]) & np.isfinite(v["td_min"])
    near = fin_t & (v["frac"] >= 0.7)
    out["validate"] = dict(
        points=int(len(v["cert"])), checked=int(np.sum(v["cert"] > 0)), delta=float(v["delta"]),
        below_model=int(np.sum(v["cert"][fin_m] <= v["model_kick"][fin_m])), model_points=int(fin_m.sum()),
        below_td=int(np.sum(v["cert"][fin_t] <= v["td_min"][fin_t])), td_points=int(fin_t.sum()),
        td_over_cert_min=r(np.min(v["td_min"][fin_t] / v["cert"][fin_t])),
        td_over_cert_max=r(np.max(v["td_min"][fin_t] / v["cert"][fin_t])),
        td_over_cert_median_ge07=r(np.median(v["td_min"][near] / v["cert"][near])),
        model_over_td_min=r(np.min(v["model_kick"][both] / v["td_min"][both])),
        model_over_td_max=r(np.max(v["model_kick"][both] / v["td_min"][both])),
        model_over_td_median=r(np.median(v["model_kick"][both] / v["td_min"][both])),
        proof_vmax=r(np.nanmax(v["proof_vmax"]), 4),
        offaxis_bad=int(v["offaxis_bad"].sum()), offaxis_runs=int(v["offaxis_runs"].sum()))
    step = {}
    for i, n in enumerate(s["rpm"]):
        j = int(np.argmin(np.abs(m["rpm"] - n)))
        cert = s["frac"][s["beta"][i] >= 1]
        step[f"{n:.0f}"] = dict(beta=[r(b, 3) for b in s["beta"][i]], certified_fracs=cert.tolist(),
                                td_step_limit_frac=r(m["a_step_clip"][j] / m["a_lim"][j], 3))
    out["step"] = dict(fracs=s["frac"].tolist(), by_rpm=step, beta_max=r(np.max(s["beta"]), 3))
    c = f["cert"]
    out["surface"] = dict(speeds=int(len(f["rpm"])), depths=int(len(f["frac"])), checked=int(np.sum(c > 0)),
                          zero=int(np.sum(c <= 0)),
                          kick_090_min_mmps=r(np.min(c[:, list(f["frac"]).index(0.9)]) * 1e3),
                          kick_090_median_mmps=r(np.median(c[:, list(f["frac"]).index(0.9)]) * 1e3),
                          kick_090_max_mmps=r(np.max(c[:, list(f["frac"]).index(0.9)]) * 1e3))
    json.dump(out, open(os.path.join(RES, "c2_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
