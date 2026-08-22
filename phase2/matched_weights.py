"""
matched_weights.py — the same weights on both uncertainty descriptions.
========================================================================
Comparing the BEST design found on each set confounds two things: the set, and
which corner of the weight grid the search happened to land on.  This module
removes that confound by pairing the sixty weight triples one to one and asking,
at each triple, the same two questions:

    does the design meet the constraints (Ms <= 2, effort <= 450 V/N,
    nominal poles <= -1 1/s) ?
    and what objective does it reach ?

The answer that matters is not "which design is better" -- on the triples where
both are feasible it is a wash -- but how much of the weight space is USABLE at
all.  A design method that produces a feasible controller almost everywhere is
worth more than one that needs the weights to be guessed exactly right, and the
published description never publishes those weights.

    python matched_weights.py
"""
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

TRIAL = re.compile(r'k_f=([\d.e+-]+) k_u=([\d.e+-]+) fc=(\d+) -> mu=([\d.]+) '
                   r'order=\s*(\d+) J=([+-][\d.]+).*?(OK|constraint violated)')


def parse(path, marks):
    cur, out = None, {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        for key, pat in marks.items():
            if pat in line:
                cur = key
                out.setdefault(cur, {})
        m = TRIAL.search(line)
        if m and cur:
            key = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            out[cur][key] = dict(mu=float(m.group(4)), order=int(m.group(5)),
                                 J=float(m.group(6)), ok=m.group(7) == 'OK')
    return out


def load():
    a = parse(os.path.join(C.RESULTS, 'log_musyn_phase2.txt'),
              {'mu_paper': '[A] uncertainty set exactly as published',
               'mu_exact': '[B] same method, corrected envelope'})
    b = parse(os.path.join(C.RESULTS, 'log_musyn_phys.txt'),
              {'mu_phys': '[mu_phys] ', 'mu_phys_sym': '[mu_phys_sym]',
               'mu_phys_ind': '[mu_phys_ind]'})
    a.update(b)
    return a


def main():
    R = load()
    print('=' * 78)
    print('THE SAME SIXTY WEIGHT TRIPLES ON EVERY UNCERTAINTY DESCRIPTION')
    print('=' * 78)
    print(f'  {"set":14s}{"trials":>8}{"feasible":>10}{"share":>8}'
          f'{"best J":>10}{"median order":>14}')
    for tag, d in R.items():
        if not d:
            continue
        ok = [v for v in d.values() if v['ok']]
        best = max((v['J'] for v in ok), default=float('-inf'))
        print(f'  {tag:14s}{len(d):8d}{len(ok):10d}{100*len(ok)/len(d):7.0f} %'
              f'{best:+10.5f}{st.median([v["order"] for v in d.values()]):14.0f}')
    P, F = R.get('mu_paper', {}), R.get('mu_phys', {})
    common = sorted(set(P) & set(F))
    both = [k for k in common if P[k]['ok'] and F[k]['ok']]
    if both:
        win = sum(F[k]['J'] > P[k]['J'] + 1e-4 for k in both)
        print(f'\n  on the {len(both)} triples where BOTH are feasible: '
              f'phys better {win}, worse {len(both)-win}')
    only_f = [k for k in common if F[k]['ok'] and not P[k]['ok']]
    only_p = [k for k in common if P[k]['ok'] and not F[k]['ok']]
    print(f'  feasible on the physics set ONLY: {len(only_f)} triples')
    print(f'  feasible on the paper set ONLY:   {len(only_p)} triples')
    print('\n  A larger feasible share means the synthesis weights, which the'
          '\n  reference does not publish, no longer have to be guessed right.')


if __name__ == '__main__':
    main()
