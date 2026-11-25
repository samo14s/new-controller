#!/usr/bin/env bash
# Round-2 design Monte Carlo: the 0.9-margin basin-aware chart (own and common depths), then the
# original chart (own depths), sharing one cache so the reference runs are computed once.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
C=verification/paper/mc_designs_cache.jsonl
python3 verification/paper/mc_designs.py --n 30 --ba-source kappa --kinds own,common --cache $C \
    --out verification/paper/mc_designs_kappa.json --workers 4 > verification/paper/mc_designs_kappa.log 2>&1
python3 verification/paper/mc_designs.py --n 30 --ba-source chart --kinds own --cache $C \
    --out verification/paper/mc_designs_chart.json --workers 4 > verification/paper/mc_designs_chart.log 2>&1
