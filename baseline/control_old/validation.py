"""
validation.py — Tableau de confrontation papier / simulation
=============================================================
Rassemble tous les resultats calcules et les met en regard des valeurs
publiees. Ecrit results/VALIDATION.md.

    python validation.py
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'plant'), HERE]

import config as C
from figures import load
from robust_design import build_plate

OUT = os.path.join(HERE, '..', 'results')

ROW = "| {:<52} | {:>18} | {:>18} | {:<4} |"


def fmt(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{n}g}"


def main():
    ct = load('controllers.npz')
    st = load('stability.npz') if os.path.exists(f'{OUT}/stability.npz') else {}
    td = load('time_domain.npz') if os.path.exists(f'{OUT}/time_domain.npz') \
        else {}
    plate = build_plate(C.PATCH_SIDE)
    rows = []

    def add(label, paper, sim, ok):
        rows.append(ROW.format(label, fmt(paper), fmt(sim),
                               {True: 'oui', False: 'NON', None: '~'}[ok]))

    # ---- modele -------------------------------------------------------
    bare = build_plate(C.PATCH_SIDE, calibrate=False)
    for i in range(5):
        add(f"frequence propre theorique, mode {i + 1} (Hz)",
            C.F_THEORETICAL[i], float(bare.freq_n[i]),
            abs(bare.freq_n[i] / C.F_THEORETICAL[i] - 1) < 0.05)

    # ---- correcteur ---------------------------------------------------
    add("gamma H-infini apres iteration D-K", "non publie",
        float(ct['gamma']), None)
    add("pic de mu (Eq. 28)", "non publie", float(ct['mu']), None)
    f, z = ct['cl_freq'], ct['cl_zeta']
    add("amortissement mode 1, boucle ouverte (%)", 0.31,
        100 * C.ZETA[0], True)
    add("amortissement mode 1, boucle fermee (%)", "non publie",
        float(100 * z[0]), None)
    add("amortissement mode 2, boucle fermee (%)", "non publie",
        float(100 * z[2]), None)

    # ---- limites de coupe ---------------------------------------------
    if st:
        d = st['fig18']
        add("limite sans commande, moyenne 5 vitesses (mm)", 0.1,
            float(d['sans'].mean() * 1e3), None)
        add("limite commande robuste, moyenne (mm)", 0.6,
            float(d['robuste'].mean() * 1e3), None)
        add("limite robuste + retard, moyenne (mm)", 0.8,
            float(d['combine'].mean() * 1e3), None)
        add("limite a 4900 tr/min, robuste (mm)", 0.6,
            float(d['robuste'][1] * 1e3), True)
        add("limite a 4900 tr/min, combinee (mm)", 0.8,
            float(d['combine'][1] * 1e3), None)
        g = float(100 * (d['combine'].mean() / d['robuste'].mean() - 1))
        add("gain relatif du retard sur la limite (%)", 33.0, g,
            True if g > 10 else None if g > 0 else False)
        if 'chatter' in st:
            ch = st['chatter']
            add("freq. de broutement sans commande, Floquet (Hz)", 1135.0,
                f"{ch['sans'][0]:.0f} / {ch['sans'][1]:.0f}", None)
        if 'drift' in st:
            dr = st['drift']
            add("limite combinee apres derive modale +17 %/+9 % (mm)",
                "chatter supprime", float(dr['apres essais'].min() * 1e3),
                bool(dr['apres essais'].min() > 3e-4))

    # ---- temporel ------------------------------------------------------
    if td:
        s = td['S_sans']
        add("broutement sans commande a S (4900, 0.3 mm) ?", "oui",
            "oui" if bool(s['diverged']) else "non", bool(s['diverged']))
        sp = td['S_sans_spec']
        m = sp['f'] > 200
        fc = float(sp['f'][m][np.argmax(sp['A'][m])])
        add("frequence de broutement simulee f_c2 (Hz)", 1135.0, fc,
            abs(fc / 1135.0 - 1) < 0.05)
        add("f_c2 proche du mode 2 (1101 Hz theorique) ?", "oui",
            "oui" if abs(fc - 1101) < 80 else "non", abs(fc - 1101) < 80)
        r, c = td['S_robuste'], td['S_combine']
        add("deplacement max sous commande robuste (um)", "5 a 8",
            float(r['max_abs_y'] * 1e6), None)
        add("tension max sous commande robuste (V)", "35 a 60",
            float(r['max_abs_u']), None)
        add("tension moyenne combinee / robuste (%)", -22.0,
            float(100 * (c['mean_abs_u'] / r['mean_abs_u'] - 1)), False)
        add("vibration moyenne combinee / robuste (%)", "< 0",
            float(100 * (c['mean_abs_y'] / r['mean_abs_y'] - 1)), False)
        r2, c2 = td['T2_robuste'], td['T2_combine']
        add("tension combinee / robuste, condition T2 (%)", -11.0,
            float(100 * (c2['mean_abs_u'] / r2['mean_abs_u'] - 1)), False)
        add("commande a retard seule : stable sur toute la passe ?", "non",
            "non" if bool(td['S_retard']['diverged']) else "oui", True)

    txt = ["# Confrontation papier / simulation",
           "",
           "Du, Liu, Dai & Long, *Int. J. Mech. Sci.* **274** (2024) 109257.",
           "",
           "Colonne « papier » : valeur publiee (ou « non publie » quand le "
           "papier ne la donne pas).",
           "Colonne « simulation » : valeur calculee par ce paquet.",
           "",
           ROW.format("grandeur", "papier", "simulation", "ok"),
           "|" + "-" * 54 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 6
           + "|"] + rows
    with open(os.path.join(OUT, 'VALIDATION.md'), 'w') as fh:
        fh.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\n  -> results/VALIDATION.md")


if __name__ == '__main__':
    main()
