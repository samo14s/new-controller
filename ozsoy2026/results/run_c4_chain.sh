#!/bin/sh
# C4 campaigns: linear limits with the comb term, certified gain map, simulated entries, validation.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for cmd in limits gainmap entry validate; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    python3 c4_comb.py $cmd --workers 4 > ../results/log_c4_$cmd.txt 2>&1 || echo "FAILED $cmd"
    tail -5 ../results/log_c4_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
