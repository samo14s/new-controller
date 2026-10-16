#!/usr/bin/env bash
# Re-creates every number in docs/ from scratch (about 40 min on a laptop).
set -euo pipefail
cd "$(dirname "$0")/scripts"
python calibrate.py              # results/calibration.{txt,json}  (~15 min)
python run_identification.py     # results/identification.txt, rom.npz
python run_control.py            # results/control.txt
python probe_theory.py           # results/theory_probe.txt   (G1-G3)
python probe_budget.py           # results/budget.txt         (G4)
python run_sweeps.py             # results/sweeps.txt
python probe_map.py              # results/map_probe.txt
python analysis_edges.py         # results/edges.txt
python probe_tmin.py             # results/tmin_probe.txt
python probe_hybrid.py           # results/hybrid_probe.txt
python fig_design_chart.py       # figures/fig_design_chart.png
python analysis_collapse.py
COLLAPSE_SOURCE=pixel python analysis_collapse.py
