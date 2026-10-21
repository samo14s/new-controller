#!/bin/sh
# C3 campaigns: Floquet limits (half and low immersion), the periodic
# certificate at the C1 basin points, at low immersion, and the decomposition
# of what each ingredient adds.  Workers are limited by memory (an SDP with
# 16 subintervals takes 2-4 GB, with 32 up to about 8 GB).
#     sh run_c3_chain.sh                 # all
#     sh run_c3_chain.sh basin lowimm    # some
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ $# -eq 0 ] && set -- floquet basin lowimm decomp
for cmd in "$@"; do
    case $cmd in floquet) w=4 ;; decomp) w=2 ;; *) w=3 ;; esac
    echo "== $cmd $(date -u +%H:%M:%S)"
    python3 c3_periodic.py $cmd --workers $w > ../results/log_c3_$cmd.txt 2>&1 || echo "FAILED $cmd"
    tail -6 ../results/log_c3_$cmd.txt
done
echo "== done $(date -u +%H:%M:%S)"
