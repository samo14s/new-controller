#!/bin/sh
# C4 after the second review: limits with the uncapped check, the DVF fine grid of
# the gain map, robustness with g2 = 1000 / 4000 and a 600 Hz mode, and the
# supplementary validation (fine-grid DVF, same-g1 delayed designs, deep points).
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for cmd in limits gainmap robust validate2; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    if [ "$cmd" = robust ]; then
        python3 c4_robust.py --workers 4 > ../results/log_c4_$cmd.txt 2>&1 || echo "FAILED $cmd"
    else
        python3 c4_comb.py $cmd --workers 4 > ../results/log_c4_$cmd.txt 2>&1 || echo "FAILED $cmd"
    fi
    tail -5 ../results/log_c4_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
