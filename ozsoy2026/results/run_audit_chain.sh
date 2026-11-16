#!/bin/sh
# Campaigns redone after the audit (README 14.11) that use the C1 kicks, once
# the C1 chain (run_c1_chain2.sh) has finished: C2's validation against the new
# C1 kicks, the simulated kicks of C3, and C4's entries and validation.
# Section 8 runs separately (run_audit_chain_s8.sh).
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-3}
while pgrep -f run_c1_chain2.sh > /dev/null; do sleep 60; done
run() { echo "== $* $(date -u +%H:%M:%S)"; log=$1; shift; "$@" > ../results/$log 2>&1 || echo "FAILED $*"; tail -2 ../results/$log; }
run log_c2_validate.txt python3 c2_certificate.py validate --workers $W
run log_c3_refresh_td.txt python3 c3_periodic.py refresh_td --workers $W
run log_c3_lowimm_td.txt python3 c3_periodic.py lowimm_td --workers $W
run log_c4_entry.txt python3 c4_comb.py entry --workers $W
run log_c4_validate.txt python3 c4_comb.py validate --workers $W
run log_c4_validate2.txt python3 c4_comb.py validate2 --workers $W
echo "== done $(date -u +%H:%M:%S)"
