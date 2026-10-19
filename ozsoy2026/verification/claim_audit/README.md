# Claim audit of the Arabic report

Five agents each took one part of `../../README.md` and recomputed every numeric
or factual claim from the stored results (188 claims). 149 were
correct; 39 were flagged as wrong, unsupported or misleading. Each flagged
claim went to a second, independent agent told to reject the flag when the report's
number was defensible: 29 were confirmed and 10 rejected.

All confirmed corrections were re-derived by the orchestrator before being applied
(e.g. the single gain 237 is infeasible at 18 speeds, not 19; the high-speed gain
match holds over 2320-3000 rpm, while 2060-2310 rpm runs 1.2-1.9x above the paper;
the exact-ZOH check covered only the held-force simulator scheme, so
`../tdsim/check7_final_scheme.py` was added for the final scheme).
Details: `claim_audit.json` (claim, recomputation, second opinion, fix).
