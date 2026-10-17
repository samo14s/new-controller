#!/bin/bash
# remaining analyses after the main gain map; writes results/chain.done at the end
cd "$(dirname "$0")/../src"
set -x
python3 run_single_gain.py --tag main --sade > ../results/log_single_gain_main.txt 2>&1
python3 run_table3.py --tag main > ../results/log_table3_main.txt 2>&1
python3 run_cuts.py --tag main > ../results/log_cuts_main.txt 2>&1
python3 run_gain_map.py --tag zeta145 --zeta 0.0145 > ../results/log_gainmap_zeta145.txt 2>&1
python3 run_gain_map.py --tag volt --kind volt > ../results/log_gainmap_volt.txt 2>&1
python3 run_gain_map.py --tag hold512 --fmode 0 > ../results/log_gainmap_hold512.txt 2>&1
touch ../results/chain.done
