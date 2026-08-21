"""
report2.py — Stage 7-9 tables for docs/07_STAGE_EXECUTION.md.
=============================================================
    python report2.py >> ../docs/07_STAGE_EXECUTION.md
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from stage_common import LABEL, ORDER, n_params

AR = dict(open='بلا تحكّم', fopid='FOPID', lqg='LQG',
          mu_tdc='μ-TDC (المرجع)', ps_ac='**PS-AC (مقترح)**',
          ps_ac_eta='PS-AC + η',
          ps_ac_obs='PS-AC — مُراقِب فقط ⚠️',
          ps_ac_full='PS-AC — $K$+مُراقِب ⚠️')


def load(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def keys(store, prefix):
    return [k for k in ORDER if f'{prefix}{k}' in store]


def main():
    s = load('stage78.pkl')
    c = load('stage56.pkl')
    d = load('stage3_controllers.pkl')

    if d:
        print('\n---\n\n## نتائج المرحلتين 3 و4 — التصميم\n')
        print('| المتحكّم | بارامترات | الرتبة | $J$ | $M_s$ | الجهد V/N | مجدوَل |')
        print('|---|---|---|---|---|---|---|')
        for k in ORDER:
            if k not in d:
                continue
            r = d[k]
            print(f'| {AR.get(k,k)} | {r["n_params"]} | {r["order"]} | '
                  f'{r["J"]:+.5f} | {r["Ms"]:.3f} | {r["V"]:.1f} | '
                  f'{"نعم" if k.startswith("ps_ac") else "لا"} |')

    if c:
        print('\n---\n\n## نتائج المرحلتين 5 و6 — الشهادة والهوامش\n')
        print('| المتحكّم | $\\sup_\\omega\\rho$ | $\\tau_{\\max}/\\tau_0$ | '
              '$a_p^{\\infty}$ (mm) | $\\Delta_{\\max}$ | LK |')
        print('|---|---|---|---|---|---|')
        for k in ORDER:
            if k not in c:
                continue
            r = c[k]
            tr = '∞' if not np.isfinite(r['tau_ratio']) else f'{r["tau_ratio"]:.2f}'
            print(f'| {AR.get(k,k)} | {r["peak"]:.3f} | {tr} | '
                  f'{r["ap_inf"]*1e3:.4f} | {r["delta_max"]:.2f} | '
                  f'{"نعم" if r["lk"] else "لا"} |')
        print('\n> $\\sup_\\omega\\rho<1$ يعني الاستقرار **لكل تأخير**، فيصير '
              '$\\tau_{\\max}=\\infty$. و$a_p^{\\infty}$ أكبر عمق يتحقّق عنده '
              'ذلك على مجموعة عدم اليقين كلّها ومجال المواضع كلّه.')

    if s:
        ks = keys(s, 'S1_')
        if ks:
            print('\n---\n\n## المرحلة 7 — المقارنة الاسمية\n')
            print('| المتحكّم | par | $a_{p,\\lim}$ أدنى | متوسط | '
                  '$A_{\\max}$ μm | $A_{RMS}$ μm | $T_s$ ms | $E_u$ V²s | '
                  '$u_{\\max}$ V |')
            print('|---|---|---|---|---|---|---|---|---|')
            for k in ks:
                r = s[f'S1_{k}']
                L = np.asarray(r['limits'], float) * 1e3
                div = ' ⚠️' if r['diverged'] else ''
                print(f'| {AR.get(k,k)} | {n_params(k)} | **{L.min():.4f}** | '
                      f'{L.mean():.4f} | {r["A_max"]:.3f}{div} | '
                      f'{r["A_rms"]:.3f} | {r["T_s"]*1e3:.1f} | '
                      f'{r["E_u"]:.4g} | {r["u_max"]:.2f} |')

        ks = keys(s, 'S2_')
        if ks and 'S2_positions' in s:
            fr = np.asarray(s['S2_positions'], float)
            print('\n---\n\n## المرحلة 8(أ) — مسح الموضع\n')
            print('| المتحكّم | ' + ' | '.join(f'{v*100:.0f} mm' for v in fr)
                  + ' | أدنى | المدى |')
            print('|---' * (len(fr) + 3) + '|')
            for k in ks:
                L = np.asarray(s[f'S2_{k}'], float) * 1e3
                print(f'| {AR.get(k,k)} | '
                      + ' | '.join(f'{v:.3f}' for v in L)
                      + f' | **{L.min():.4f}** | {L.max()-L.min():.4f} |')
            print('\n> عمود «المدى» هو تشتّت الحدّ على طول الحافّة: كلّما صغر، '
                  'كان أداء المتحكّم أكثر تجانسًا مكانيًّا — وهو ما تستهدفه '
                  'الجدولة على الموضع مباشرةً.')

        ks = keys(s, 'S3_')
        if ks and 'S3_cases' in s:
            cs = s['S3_cases']
            print('\n---\n\n## المرحلة 8(ب) — تغيّر المعاملات النمطية\n')
            print('| المتحكّم | ' + ' | '.join(cs) + ' | الأسوأ |')
            print('|---' * (len(cs) + 2) + '|')
            for k in ks:
                v = np.asarray(s[f'S3_{k}'], float)
                print(f'| {AR.get(k,k)} | ' + ' | '.join(f'{x:.4f}' for x in v)
                      + f' | **{v.min():.4f}** |')

        ks = keys(s, 'S4_')
        if ks and 'S4_ratios' in s:
            rr = np.asarray(s['S4_ratios'], float)
            print('\n---\n\n## المرحلة 8(ج) — زيادة التأخير\n')
            print('| المتحكّم | ' + ' | '.join(f'{r:.2f}τ₀' for r in rr)
                  + ' | الأسوأ |')
            print('|---' * (len(rr) + 2) + '|')
            for k in ks:
                v = np.asarray(s[f'S4_{k}'], float)
                print(f'| {AR.get(k,k)} | ' + ' | '.join(f'{x:.4f}' for x in v)
                      + f' | **{v.min():.4f}** |')


if __name__ == '__main__':
    main()
