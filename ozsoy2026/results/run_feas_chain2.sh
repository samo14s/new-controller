#!/bin/bash
# remaining reformulated-criterion maps, single-threaded BLAS per worker;
# starts once the main map (feas_g1500.npz) exists; writes feas_chain.done
cd "$(dirname "$0")/../src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ ! -f ../results/feas_g1500.npz ]; do sleep 20; done
set -x
python3 run_feasibility_map.py --tag g3000 --gmax 3000 --certify > ../results/log_feas_g3000.txt 2>&1
python3 run_feasibility_map.py --tag k095 --gmax 1500 --kappa 0.95 > ../results/log_feas_k095.txt 2>&1
python3 run_feasibility_map.py --tag guard085 --gmax 1500 --guard 0.85 > ../results/log_feas_guard085.txt 2>&1
touch ../results/feas_chain.done
