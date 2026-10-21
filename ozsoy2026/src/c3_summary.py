"""Headline numbers of C3, computed from the stored results.

    python c3_summary.py      -> ../results/c3_summary.json
The report quotes these numbers; tests/test_c3_results.py pins them.

It also corrects C2's and C4's certificates for the error of their command
bound: feasibility's slice-force harmonics come from a 4096-point DFT and the
forced orbit stops at N_HARM harmonics, so the true peak command is up to a
few mV above the b_max they used.  A certificate at level u0 stays valid at
u0 - d after scaling (W, Y, S) by ((u0 - d) / u0)^2 (every LMI is
homogeneous, the inclusion scales with it), so its size scales by
(u0 - d) / u0.  d is taken from c3_periodic.orbit (closed-form harmonics plus
the bound on the rest).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
VMAX = 9.0


def r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def stats(x, nd=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(min=r(x.min(), nd), median=r(np.median(x), nd), max=r(x.max(), nd), n=int(x.size)) if x.size else None


def load(name):
    p = os.path.join(RES, name)
    return np.load(p) if os.path.exists(p) else None


def _fmin(rows):
    return np.array([np.min(x[np.isfinite(x)]) if np.isfinite(x).any() else np.inf for x in np.atleast_2d(rows)])


def floquet(F):
    out = {}
    for i, ae in enumerate(F["ae"]):
        q = F["a_floquet"][i] / F["a_zoa"][i]
        flip = (F["mu"][i].real < 0) & (np.abs(F["mu"][i].imag) < 1e-6 * np.abs(F["mu"][i].real))
        out[f"{ae:g}"] = dict(speeds=int(len(q)), floquet_over_zoa=stats(q, 4), below_zoa=int(np.sum(q < 1 - 1e-5)),
                              flip=int(flip.sum()), node_convergence=r(np.nanmax(np.abs(F["rho_n8"][i] - 1)), 7))
    return out


def basin(B, phases=None):
    """Statistics of a basin-type campaign (run_basin or run_lowimm)."""
    c, c2s, ph = B["cert"], B["c2_cert"], B["phase_cert"]
    mk, ak, td = B["model_kick"], B["avg_kick"], B["td_kick"]
    cols = np.arange(ph.shape[1]) if phases is None else np.asarray(phases)
    ok = c > 0
    tdmin, mkmin, akmin = _fmin(td), _fmin(mk), _fmin(ak)
    both = np.isfinite(mk) & np.isfinite(ak)
    fin_td = np.isfinite(td) & ok[:, None]
    phc = ph[:, cols]
    out = dict(
        points=int(len(c)), certified=int(ok.sum()),
        c3_over_c2=stats(c[ok] / c2s[ok]), c3_above_c2=int(np.sum(c[ok] > c2s[ok])),
        c3_over_c2_by_rpm={f"{n:.0f}": stats(c[ok & (B["rpm"] == n)] / c2s[ok & (B["rpm"] == n)])
                           for n in np.unique(B["rpm"])},
        cert_mm_s=stats(c[ok] * 1e3), c2_mm_s=stats(c2s * 1e3),
        phase_spread=stats(np.max(ph[ok], axis=1) / np.min(ph[ok], axis=1)),
        phase_cert_below_model=[int(np.sum((ph <= mk)[ok])), int(ok.sum() * ph.shape[1])],
        phase_cert_below_td=[int(np.sum((phc <= td)[fin_td])), int(fin_td.sum())],
        cert_below_td_min=[int(np.sum(c[ok] <= tdmin[ok])), int(np.sum(ok & np.isfinite(tdmin)))],
        c2_below_periodic_model=[int(np.sum(c2s <= mkmin)), int(len(c2s))],
        periodic_over_averaged_model=stats(mk[both] / ak[both], 4),
        model_over_td=stats(mkmin / tdmin),
        td_over_cert=stats(tdmin[ok] / c[ok]), td_over_c2=stats(tdmin[ok] / c2s[ok]),
        proof_vmax=r(np.nanmax(B["proof_vmax"]), 5), proof_vend=r(np.nanmax(B["proof_vend"]), 5),
        offaxis=[int(B["offaxis_bad"].sum()), int(B["offaxis_runs"].sum())],
        u0_min=stats(B["u0_min"]), u0_max=stats(B["u0_max"]),
        regularised=int(sum("regularised" in str(m) for m in B["msg"])),
        failed=[f"{n:.0f} rpm {f:.2f}: {m}" for n, f, m, cc in zip(B["rpm"], B["frac"], B["msg"], c) if cc <= 0])
    return out


def decomp(D, B):
    full = np.array([B["cert"][(B["rpm"] == n) & np.isclose(B["frac"], f)][0] for n, f in zip(D["rpm"], D["frac"])])
    out = {str(v): stats(D["size"][:, j] / D["c2_size"]) for j, v in enumerate(D["variants"])}
    out["periodic, resolved, m_c 16"] = stats(full / D["c2_size"])
    out["points"] = int(len(D["rpm"]))
    return out


def _c4_gap(G, kick, kdvf):
    """c4_summary's certified depth without a gap at >= 1, 2, 4 mm/s (best DVF, best delayed design)."""
    kick, kdvf = kick * 1e3, kdvf * 1e3
    out = {}
    for i, n in enumerate(G["rpm"]):
        d_best = np.maximum(kick[i, :, :, 0].max(axis=1), kdvf[i].max(axis=1))
        c_best = kick[i, :, :, 1:].reshape(len(G["depth"]), -1).max(axis=1)
        row = {}
        for D in (1.0, 2.0, 4.0):
            for lab, y in (("dvf", d_best), ("comb", c_best)):
                bad = np.where(y < D)[0]
                last = (bad[0] - 1) if bad.size else len(y) - 1
                row[f"{lab}_{D:g}"] = float(G["depth"][last]) if last >= 0 else None
        out[f"{n:.0f}"] = row
    return out


def margin_errata():
    """C2's and C4's certificates rescaled to the closed-form command bound."""
    import c2_certificate as c2
    import c3_periodic as c3
    from zoa import a_lim
    import params as P
    out = {}
    V = load("c2_validate.npz")
    if V is not None:
        d, f = [], []
        for n, g, fr, cert in zip(V["rpm"], V["g"], V["frac"], V["cert"]):
            a = fr * a_lim(n, g)
            bm = c2.orbit(n, g, a)[2]
            bk = c3.orbit(n, g, a)[2].max()
            d.append(bk - bm)
            f.append((VMAX - bk) / (VMAX - bm))
        out["c2_validate"] = dict(points=len(d), extra_mV=stats(np.array(d) * 1e3), size_factor=stats(f, 5))
    S = load("c2_surface.npz")
    if S is not None:
        d, f = [], []
        fac = np.ones_like(S["cert"])
        for i, (n, g, al) in enumerate(zip(S["rpm"], S["g"], S["a_lim"])):
            for j, fr in enumerate(S["frac"]):
                if S["cert"][i, j] <= 0:
                    continue
                bk = c3.orbit(n, g, fr * al * 1e-3)[2].max()
                u0 = S["u0"][i, j]
                d.append(bk - (VMAX - u0))
                fac[i, j] = max(0.0, (VMAX - bk) / u0)
        k9 = list(np.round(S["frac"], 6)).index(0.9)
        out["c2_surface"] = dict(points=len(d), extra_mV=stats(np.array(d) * 1e3), size_factor=stats(fac[S["cert"] > 0], 5),
                                 kick_090_mmps=[stats(S["cert"][:, k9] * 1e3, 2), stats(S["cert"][:, k9] * fac[:, k9] * 1e3, 2)])
    G = load("c4_gainmap.npz")
    if G is not None:
        fd = np.ones_like(G["kick_dvf"])
        fc = np.ones_like(G["kick"])
        for i, n in enumerate(G["rpm"]):
            for j, dep in enumerate(G["depth"]):
                for k, g1 in enumerate(G["g1_dvf"]):
                    if G["kick_dvf"][i, j, k] > 0:
                        bk = c3.orbit(n, g1, dep * 1e-3)[2].max()
                        fd[i, j, k] = max(0.0, (VMAX - bk) / (VMAX - G["bmax_dvf"][i, j, k]))
                for k, g1 in enumerate(G["g1"]):
                    if np.any(G["kick"][i, j, k] > 0):
                        bk = c3.orbit(n, g1, dep * 1e-3)[2].max()
                        fc[i, j, k] = np.maximum(0.0, (VMAX - bk) / (VMAX - G["bmax"][i, j, k]))
        f = np.concatenate([fd[G["kick_dvf"] > 0], fc[G["kick"] > 0]])
        out["c4_gainmap"] = dict(certificates=int(f.size), size_factor=stats(f, 5),
                                 certified_without_gap_mm=[_c4_gap(G, G["kick"], G["kick_dvf"]),
                                                           _c4_gap(G, G["kick"] * fc, G["kick_dvf"] * fd)])
    return out


def build(errata=True):
    out = {}
    F, B, L, D = load("c3_floquet.npz"), load("c3_basin.npz"), load("c3_lowimm.npz"), load("c3_decomp.npz")
    if F is not None:
        out["floquet"] = floquet(F)
    if B is not None:
        out["basin"] = basin(B)
    if L is not None:
        out["lowimm"] = basin(L, L["td_phases"])
        out["lowimm"]["ae"] = float(L["ae"])
    if D is not None and B is not None:
        out["decomp"] = decomp(D, B)
    if errata:
        out["errata"] = margin_errata()
    return out


def main():
    s = build()
    with open(os.path.join(RES, "c3_summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
