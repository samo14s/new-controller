"""Which DVF gain does each Fig. 6(b) pole correspond to?

Closed loop of the 1-DOF structure (k = 1.34e7 N/m, fn = 129.3 Hz, zeta) with the
proof-mass actuator Ga = 3 s^2/(s^2 + 15.834 s + 2785.6) and V = -g xdot:
    (m s^2 + c s + k)(s^2 + a1 s + a0) + 3 g s^3 = 0.
For each legend gain the dominant pole is computed and, conversely, the gain that
reproduces the digitised pole real part is solved for.  Used only to flag
inconsistencies between the Fig. 6 legend and the plotted poles.
"""
import csv, json, os
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
K, FN, A1, A0 = 1.34e7, 129.3, 15.834, 2785.6
WN = 2 * np.pi * FN
M = K / WN ** 2


def poles(g, zeta):
    c = 2 * zeta * np.sqrt(K * M)
    p = np.polymul([M, c, K], [1, A1, A0])
    p = np.polyadd(p, [3 * g, 0, 0, 0])
    r = np.roots(p)
    return sorted(r, key=lambda z: -abs(z.imag))


rows = list(csv.DictReader(open(os.path.join(HERE, "fig6b_poles.csv"))))
out = []
for zeta in (0.0134, 0.0145):
    for case, g in (("500rpm_g1487", 1487), ("1300rpm_g1162", 1162), ("2500rpm_g585", 585), ("3000rpm_g797", 797)):
        dig = [r for r in rows if r["case"] == case and float(r["imag_rad_s"]) > 500][0]
        sig_d = float(dig["real_rad_s"])
        p = poles(g, zeta)[0]
        gfit = brentq(lambda gg: poles(gg, zeta)[0].real - sig_d, 0, 5000)
        out.append(dict(zeta=zeta, case=case, legend_gain=g, model_pole=[round(p.real, 2), round(p.imag, 1)],
                        digitised_pole=[sig_d, float(dig["imag_rad_s"])], gain_matching_digitised_real_part=round(gfit, 1)))
    ol = poles(0, zeta)[0]
    out.append(dict(zeta=zeta, case="open_loop", model_pole=[round(ol.real, 2), round(ol.imag, 1)]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "fig6b_pole_check.json"), "w"), indent=1)
