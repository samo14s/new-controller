#!/bin/sh
# C2 campaigns: certified kick surface, then the step-entry certificate.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for cmd in surface step; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    python3 c2_certificate.py $cmd --workers 3 > ../results/log_c2_$cmd.txt 2>&1 || echo "FAILED $cmd"
    tail -3 ../results/log_c2_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
