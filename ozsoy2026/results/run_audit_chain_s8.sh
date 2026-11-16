#!/bin/sh
# Section 8's time-domain checks redone after the audit with the full surface
# memory (feasibility.NREG = 0).  Independent of the C1 campaigns, so it runs
# alongside them on the remaining core.
cd "$(dirname "$0")/../src" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-1}
run() { echo "== $* $(date -u +%H:%M:%S)"; log=$1; shift; "$@" > ../results/$log 2>&1 || echo "FAILED $*"; tail -2 ../results/$log; }
run log_feas_g1500_cert.txt python3 run_feasibility_map.py --tag g1500 --certify-only --workers $W
run log_feas_g3000_cert.txt python3 run_feasibility_map.py --tag g3000 --certify-only --workers $W
run log_feas_cert_g1500.txt python3 run_feas_certify.py --tag g1500 --workers $W
run log_feas_cert_g3000.txt python3 run_feas_certify.py --tag g3000 --workers $W
run log_feas_compare_g1500.txt python3 run_feas_compare.py --tag g1500 --workers $W
echo "== done $(date -u +%H:%M:%S)"
