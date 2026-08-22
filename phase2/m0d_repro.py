"""m0d_repro.py — stage M0d: pin down the D-K reproducibility gap.

The (c2) arm showed the stored mu_paper winning recipe -- triple (3e5, 6e-2,
3000), paper set, n_iter=2 -- re-synthesises to a controller whose
position-free per-node mu_RS is 6.271, while the STORED artifact scores 1.156
under the same judge.  Same class of anomaly as the earlier unexplained
regeneration (mu 50.2 against the stored 9.924).  Three parts settle where the
gap lives:

  A  determinism -- the winner triple synthesised twice from clean state in
     one process, then compared: identical or not.  scipy's simplex D-fit and
     the Riccati chain are deterministic on paper; this checks it in fact.

  B  history replay -- the ORIGINAL search loop of run_musyn.search, verbatim
     (same plate object reused, same per-trial weight patching, the same
     objective evaluate after each design), over the first 24 grid entries --
     the stored winner was trial 24.  Each replayed trial's (mu, J, order) is
     diffed against the historical log line by line, so if the replay tracks
     history for k trials and diverges at k+1, the state change is located in
     trial k without any guessing.  The final trial's controller is judged
     position-free and compared against the stored artifact's matrices.

  C  state diff -- fingerprints of the plate object and the patched modules
     taken before the first trial and after the last, so whatever mutated is
     named, not suspected.

Appends to results/log_musyn_sched.txt and writes results/m0d_repro.npz.

    python phase2/m0d_repro.py
"""
import hashlib
import itertools
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import musyn
from musyn_sched import judge_point, log
from objective import evaluate
from plate_model import build_plate
from robust_design import freq_grid

OUT = C.RESULTS
WINNER = (3e5, 6e-2, 3000.0)
GRID = list(itertools.product([1e5, 3e5, 1e6, 3e6],
                              [1.7e-3, 5e-3, 1.7e-2, 6e-2, 2e-1],
                              [800.0, 1500.0, 3000.0]))
N_REPLAY = GRID.index(WINNER) + 1          # the stored winner was this trial
PERTS = dict(mass_pert=0.10, stiff_pert=0.10, damp_pert=0.20)


def synth(plate, kf, ku, fc):
    import weights as W
    W.W_PF_DEF = dict(k=kf, fc_hz=fc, M=250.0)
    W.W_PU_DEF = dict(k=ku, fc_hz=2500.0, M=60.0)
    return musyn.design(plate, alpha_coupling='paper', n_iter=2,
                        reduce_to=None, verbose=False, **PERTS)


def ss_hash(ss):
    h = hashlib.sha256()
    for m in ss:
        h.update(np.ascontiguousarray(np.asarray(m, float)).tobytes())
    return h.hexdigest()[:16]


def fingerprint(plate):
    """Hashes of everything plausibly mutable that the chain touches."""
    import weights as W
    import uncertain_plant as up
    fp = {}
    for k, v in sorted(vars(plate).items()):
        a = np.asarray(v, float) if isinstance(v, (np.ndarray, list, tuple,
                                                   float, int)) else None
        fp[f'plate.{k}'] = (hashlib.sha256(
            np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
            if a is not None and a.dtype.kind == 'f' else repr(v)[:60])
    fp['weights.W_PF_DEF'] = repr(W.W_PF_DEF)
    fp['weights.W_PU_DEF'] = repr(W.W_PU_DEF)
    fp['weights.W_PN_DEF'] = repr(W.W_PN_DEF)
    fp['up.nominal_and_perturbations'] = up.nominal_and_perturbations.__name__
    return fp


def historical_trials():
    """(mu, J, order) per trial of the [A] block, parsed from the stored log."""
    txt = open(os.path.join(OUT, 'log_musyn_phase2.txt')).read()
    block = txt.split('[A] uncertainty set exactly as published')[1]
    block = block.split('[B]')[0]
    rows = []
    for m in re.finditer(r'k_f=(\S+) k_u=(\S+) fc=(\d+) -> mu=([\d.]+) '
                         r'order=\s*(\d+) J=([+-][\d.]+)', block):
        rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                     float(m.group(4)), int(m.group(5)), float(m.group(6))))
    return rows


def main():
    t0 = time.time()
    plate = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    f = freq_grid()
    nodes = tuple(fr * plate.lp for fr in (0.0, 0.5, 1.0))
    hist = historical_trials()

    log('')
    log('M0d - WHERE THE D-K REPRODUCIBILITY GAP LIVES')
    log('=' * 78)

    # ---- A: determinism from clean state --------------------------------
    log('[A] determinism: the winner triple twice from clean state')
    r1 = synth(plate, *WINNER)
    r2 = synth(plate, *WINNER)
    same = ss_hash(r1['ss']) == ss_hash(r2['ss'])
    log(f'  run 1: mu={r1["mu"]:.3f} order={r1["order"]}   '
        f'run 2: mu={r2["mu"]:.3f} order={r2["order"]}   '
        f'K bytes {"IDENTICAL" if same else "DIFFER"}')

    # ---- B: verbatim history replay --------------------------------------
    log(f'[B] history replay: the original loop, trials 1..{N_REPLAY} '
        '(the stored winner was the last)')
    fp0 = fingerprint(plate)
    plate_r = build_plate(patch=C.PATCH_SIDE, freqs=C.F_MEASURED)
    rows = []
    diverged_at = None
    last = None
    for i, (kf, ku, fc) in enumerate(GRID[:N_REPLAY], 1):
        t = time.time()
        try:
            r = synth(plate_r, kf, ku, fc)
            J, info = evaluate(plate_r, r['ss'], detail=True)
        except Exception as e:                                # noqa: BLE001
            log(f'  {i:2d}: kf={kf:.0e} ku={ku:.0e} fc={fc:.0f} -> failed '
                f'({type(e).__name__})')
            rows.append((i, np.nan, np.nan, np.nan))
            continue
        h = hist[i - 1] if i <= len(hist) else None
        match = (h is not None and abs(r['mu'] - h[3]) < 5e-3
                 and abs(J - h[5]) < 5e-3)
        if h is not None and not match and diverged_at is None:
            diverged_at = i
        rows.append((i, r['mu'], J, r['order']))
        last = r
        log(f'  {i:2d}: mu={r["mu"]:8.3f} (hist {h[3]:8.3f})  '
            f'J={J:+9.5f} (hist {h[5]:+9.5f})  '
            f'{"match" if match else "DIVERGES"}  [{time.time()-t:.0f}s]')

    if diverged_at is None:
        log('  every replayed trial matches the historical log')
    else:
        log(f'  first divergence at trial {diverged_at}: the state change '
            f'sits in trial {diverged_at - 1} or the process start')

    d = np.load(os.path.join(OUT, 'musyn_phase2.npz'), allow_pickle=True)
    stored = tuple(d[f'mu_paper_ss{i}'] for i in range(4))
    if last is not None:
        same_mat = (all(np.asarray(a).shape == np.asarray(b).shape
                        for a, b in zip(last['ss'], stored))
                    and all(np.allclose(a, b, atol=1e-8)
                            for a, b in zip(last['ss'], stored)))
        sup = max(judge_point(plate, x, last['ss'], f) for x in nodes)
        sup_st = max(judge_point(plate, x, stored, f) for x in nodes)
        log(f'  replayed winner vs stored artifact: matrices '
            f'{"IDENTICAL" if same_mat else "DIFFER"};  judged sup RS: '
            f'replay {sup:.3f}  stored {sup_st:.3f}')

    # ---- C: what mutated --------------------------------------------------
    fp1 = fingerprint(plate_r)
    changed = {k for k in fp0 if fp0[k] != fp1.get(k)} | \
              {k for k in fp1 if k not in fp0}
    changed -= {'weights.W_PF_DEF', 'weights.W_PU_DEF'}   # the intended input
    log('[C] state diff over the replay (weights excluded, they are the '
        'input):')
    if changed:
        for k in sorted(changed):
            log(f'  CHANGED {k}: {fp0.get(k, "<absent>")[:40]} -> '
                f'{fp1.get(k, "<absent>")[:40]}')
    else:
        log('  nothing else changed: the chain is state-clean, and any gap '
            'must be elsewhere')

    np.savez(os.path.join(OUT, 'm0d_repro.npz'),
             rows=np.array(rows, float), n_replay=N_REPLAY,
             winner=np.array(WINNER),
             deterministic=bool(same),
             diverged_at=-1 if diverged_at is None else diverged_at)
    log(f'M0d total {time.time()-t0:.0f}s -> results/m0d_repro.npz')


if __name__ == '__main__':
    main()
