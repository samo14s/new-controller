"""
report.py — turn the stored results into the tables of docs/05_COMPARISON.md.
=============================================================================
    python report.py >> ../docs/05_COMPARISON.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

NAMES = {'open': 'بلا تحكّم', 'pid': 'PID', 'lqr': 'LQR', 'smc': 'SMC',
         'mu_paper': 'μ (مجموعة المرجع)', 'mu_exact': 'μ (مجموعة مصحَّحة)',
         'proposed': 'PB-RAC', 'proposed*': '**PB-RAC مجدوَل**'}
NP = {'open': 0, 'pid': 4, 'lqr': 5, 'smc': 5, 'mu_paper': 4, 'mu_exact': 4,
      'proposed': 7, 'proposed*': 7}
ORDER = ('open', 'pid', 'lqr', 'smc', 'mu_paper', 'mu_exact', 'proposed',
         'proposed*')


def load(name):
    p = os.path.join(C.RESULTS, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def row(vals, fmt='{:.4f}'):
    return ' | '.join(fmt.format(v) if np.isfinite(v) else '—' for v in vals)


def table(d, prefix, header, cols, fmt='{:.4f}', note=''):
    ks = [k for k in ORDER if f'{prefix}{k}' in d.files]
    if not ks:
        return ''
    out = [f'\n### {header}\n', '| المتحكّم | ' + ' | '.join(cols) + ' | الأسوأ |',
           '|---' * (len(cols) + 2) + '|']
    for k in ks:
        v = np.asarray(d[f'{prefix}{k}'], float)
        out.append(f'| {NAMES.get(k, k)} | ' + row(v, fmt)
                   + f' | **{np.min(v):.4f}** |')
    if note:
        out.append('\n' + note)
    return '\n'.join(out)


def main():
    d = load('compare_phase2.npz')
    if d is None:
        print('no results yet')
        return
    print('\n---\n\n## السيناريو 1 — النظام الاسمي\n')
    ks = [k for k in ORDER if f'S1_{k}_limits' in d.files]
    print('| المتحكّم | بارامترات | $a_{p,\\lim}$ أدنى | متوسط | '
          '$A_{\\max}$ μm | $A_{RMS}$ μm | $E_u$ V²s | $u_{\\max}$ V |')
    print('|---|---|---|---|---|---|---|---|')
    for k in ks:
        L = np.asarray(d[f'S1_{k}_limits'], float) * 1e3
        g = lambda f: float(d[f'S1_{k}_{f}']) if f'S1_{k}_{f}' in d.files \
            else np.nan
        div = ' ⚠️تباعد' if g('diverged') else ''
        print(f'| {NAMES.get(k,k)} | {NP.get(k,0)} | **{L.min():.4f}** | '
              f'{L.mean():.4f} | {g("A_max"):.3f}{div} | {g("A_rms"):.3f} | '
              f'{g("E_u"):.4g} | {g("u_max"):.2f} |')

    etas = np.linspace(0.0, C.ETA_MAX, 5)
    print(table(d, 'S2a_', 'السيناريو 2(أ) — إزالة المادة (فيزيائي)',
                [f'η={e:.3f}' for e in etas],
                note='العمود الأخير هو **الأسوأ على مجال الإزالة كلّه** — وهو '
                     'المقياس الذي يهمّ في ممرّ حقيقي.'))
    print(table(d, 'S2b_', "السيناريو 2(ب) — صندوق المرجع المتناظر",
                ['−10 %', '−5 %', '0', '+5 %', '+10 %']))
    print(table(d, 'S3_', 'السيناريو 3 — الصلابة والتخميد',
                ['ζ×0.8', 'ζ×1.2', 'K −10 %', 'K +10 %', 'K−10 % ζ0.8']))
    print(table(d, 'S4_', 'السيناريو 4 — التأخير',
                ['τ₀', '1.25τ₀', '1.5τ₀', '1.75τ₀', '2τ₀']))

    ks = [k for k in ORDER if f'S4_{k}_ratio' in d.files]
    if ks:
        print('\n### مجال التأخير المستقرّ عند العمق الاسمي\n')
        print('| المتحكّم | $\\tau/\\tau_0$ الأدنى | الأعلى | العرض |')
        print('|---|---|---|---|')
        for k in ks:
            lo, hi = np.asarray(d[f'S4_{k}_ratio'], float)
            w = hi - lo if np.isfinite(hi) and np.isfinite(lo) else np.nan
            print(f'| {NAMES.get(k,k)} | {lo:.2f} | {hi:.2f} | '
                  f'**{w:.2f}** |')

    c = load('certify_phase2.npz')
    if c is not None:
        print('\n---\n\n## المراحل 9–11 — الشهادة والهوامش\n')
        print('| المتحكّم | مُصادَق عند $a_p=0.3$ mm | $a_p^{LK}$ (mm) | '
              '$\\delta_{\\max}$ | $\\tau/\\tau_0$ |')
        print('|---|---|---|---|---|')
        for k in ORDER:
            if f'{k}_ap_lk' not in c.files:
                continue
            print(f'| {NAMES.get(k,k)} | '
                  f'{"نعم" if bool(c[f"{k}_feasible"]) else "لا"} | '
                  f'{float(c[f"{k}_ap_lk"])*1e3:.4f} | '
                  f'{float(c[f"{k}_delta_max"]):.2f} | '
                  f'[{float(c[f"{k}_ratio_lo"]):.2f}, '
                  f'{float(c[f"{k}_ratio_hi"]):.2f}] |')


if __name__ == '__main__':
    main()
