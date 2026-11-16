#!/bin/sh
# C1 campaigns redone after the audit: the chip cuts against the full surface
# (tdsim nreg = 0) instead of the last 4 passes, and the critical kick is
# searched upwards on a kick grid (the outcome is not monotone in the kick).
# The describing-function campaign (df) uses no simulation and is kept.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-3}
for cmd in basin band gainplane map; do
    echo "== $cmd $(date -u +%H:%M:%S)"
    python3 c1_unsafe_zone.py $cmd --workers $W > ../results/log_c1_$cmd.txt 2>&1 || echo "FAILED $cmd"
    tail -3 ../results/log_c1_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
