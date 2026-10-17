"""Gain implied by each Fig. 6(a) closed-loop FRF peak (same simple model as check_fig6b_poles.py)."""
import json, os
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
K, FN, A1, A0 = 1.34e7, 129.3, 15.834, 2785.6
WN = 2 * np.pi * FN
M = K / WN ** 2
f = np.linspace(100, 160, 60001)
s = 2j * np.pi * f


def peak_db(g, zeta=0.0134):
    c = 2 * zeta * np.sqrt(K * M)
    G = 1 / (M * s ** 2 + c * s + K)
    Ga = 3 * s ** 2 / (s ** 2 + A1 * s + A0)
    H = G / (1 + g * s * Ga * G)
    m = 20 * np.log10(np.abs(H))
    j = int(np.argmax(m))
    return float(m[j]), float(f[j])


dig = json.load(open(os.path.join(HERE, "frf_features.json")))["fig6a"]
out = []
for case, g in (("uncontrolled", 0), ("500rpm_g1487", 1487), ("1300rpm_g1162", 1162), ("2500rpm_g585", 585), ("3000rpm_g797", 797)):
    pk, fp = peak_db(g)
    d = dig[case]["peak_value"]
    gfit = brentq(lambda gg: peak_db(gg)[0] - d, 0, 5000) if g > 0 else 0.0
    out.append(dict(case=case, legend_gain=g, model_peak_dB=round(pk, 2), model_peak_Hz=round(fp, 2),
                    digitised_peak_dB=d, digitised_peak_Hz=dig[case]["peak_freq_Hz"], gain_matching_digitised_peak=round(gfit, 1)))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "fig6a_peak_check.json"), "w"), indent=1)
