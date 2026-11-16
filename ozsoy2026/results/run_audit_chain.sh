#!/bin/sh
# Campaigns redone after the audit (README 14.11), once the C1 chain
# (run_c1_chain2.sh) has finished: section 8's time-domain checks with the full
# surface memory, C2's validation against the new C1 kicks, the simulated
# kicks of C3, and C4's entries and validation.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-3}
while pgrep -f run_c1_chain2.sh > /dev/null; do sleep 60; done
run() { echo "== $* $(date -u +%H:%M:%S)"; log=$1; shift; "$@" > ../results/$log 2>&1 || echo "FAILED $*"; tail -2 ../results/$log; }
run log_feas_g1500_cert.txt python3 run_feasibility_map.py --tag g1500 --certify-only --workers $W
run log_feas_g3000_cert.txt python3 run_feasibility_map.py --tag g3000 --certify-only --workers $W
run log_feas_cert_g1500.txt python3 run_feas_certify.py --tag g1500 --workers $W
run log_feas_cert_g3000.txt python3 run_feas_certify.py --tag g3000 --workers $W
run log_feas_compare_g1500.txt python3 run_feas_compare.py --tag g1500 --workers $W
run log_c2_validate.txt python3 c2_certificate.py validate --workers $W
run log_c3_refresh_td.txt python3 c3_periodic.py refresh_td --workers $W
run log_c3_lowimm_td.txt python3 c3_periodic.py lowimm_td --workers $W
run log_c4_entry.txt python3 c4_comb.py entry --workers $W
run log_c4_validate.txt python3 c4_comb.py validate --workers $W
run log_c4_validate2.txt python3 c4_comb.py validate2 --workers $W
echo "== done $(date -u +%H:%M:%S)"
