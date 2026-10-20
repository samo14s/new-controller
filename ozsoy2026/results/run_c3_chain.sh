#!/bin/sh
# C3 campaigns: Floquet limits (half and low immersion), the periodic
# certificate at the C1 basin points, at low immersion, and the decomposition
# of what each ingredient adds.  Workers are limited by memory (an SDP with
# 16 subintervals takes 2-4 GB, with 32 up to about 8 GB).
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for job in "floquet 4" "basin 3" "lowimm 3" "decomp 2"; do
    set -- $job
    echo "== $1 $(date -u +%H:%M:%S)"
    python3 c3_periodic.py $1 --workers $2 > ../results/log_c3_$1.txt 2>&1 || echo "FAILED $1"
    tail -6 ../results/log_c3_$1.txt
done
echo "== done $(date -u +%H:%M:%S)"
