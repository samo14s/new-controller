#!/bin/sh
# C4 campaigns: linear limits and feasible designs, certified gain map, simulated
# entries, robustness (closed form + simulator checks), validation.
# The limits step was rerun afterwards (about 11:4x for the P-term designs, and at
# 15:17 by run_c4_chain2.sh with the uncapped check): see log_c4_limits.txt and log_c4_chain2.txt.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for cmd in limits gainmap entry robust validate; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    if [ "$cmd" = robust ]; then
        python3 c4_robust.py --workers 4 > ../results/log_c4_$cmd.txt 2>&1 || echo "FAILED $cmd"
    else
        python3 c4_comb.py $cmd --workers 4 > ../results/log_c4_$cmd.txt 2>&1 || echo "FAILED $cmd"
    fi
    tail -5 ../results/log_c4_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
