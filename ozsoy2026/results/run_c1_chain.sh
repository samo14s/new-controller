#!/bin/sh
# C1 campaigns with multiple regeneration (nreg = 4), one after the other.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for cmd in df basin band gainplane map; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    python3 c1_unsafe_zone.py $cmd --workers 4 > ../results/log_c1_$cmd.txt 2>&1 || echo "FAILED $cmd"
    tail -3 ../results/log_c1_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
