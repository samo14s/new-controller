#!/bin/bash
# reformulated-criterion maps; writes results/feas_chain.done at the end
cd "$(dirname "$0")/../src"
set -x
python3 run_feasibility_map.py --tag g1500 --gmax 1500 --sade --certify > ../results/log_feas_g1500.txt 2>&1
python3 run_feasibility_map.py --tag g3000 --gmax 3000 --certify > ../results/log_feas_g3000.txt 2>&1
python3 run_feasibility_map.py --tag k095 --gmax 1500 --kappa 0.95 > ../results/log_feas_k095.txt 2>&1
python3 run_feasibility_map.py --tag guard085 --gmax 1500 --guard 0.85 > ../results/log_feas_guard085.txt 2>&1
touch ../results/feas_chain.done
