#!/bin/bash
# Numerical-protocol checks asked by the reviewer, run one after another (one
# process at a time) in priority order:
#   (4) protocol_eq14_sampling  sampling of Eq. (14): 80 depths against a dense search
#   (1) protocol_kick_timing    kick at revolution 300 instead of 100; entry residual
#   (3) protocol_clip_level     critical kick with an 8.1 V clip against 9 V
#   (5) protocol_window         classifier window of 900 revolutions instead of 300
#   (2) protocol_phases         64 kick phases instead of 16
#   (6) protocol_band_holes     bistable band scanned on a depth grid
# Each step waits for an idle core through the queue wrapper WAITCORE (a
# helper of the shared machine; if it is absent the step starts at once), and
# each script is resumable (units already in its JSON are skipped).
# Usage:
#   cd ozsoy2026 && nohup bash verification/paper/protocol_chain.sh > verification/paper/protocol_chain.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
WAIT=${WAITCORE:-/tmp/claude-0/-home-user-new-controller/d42dfb76-3818-53d8-8bcc-e62ff0a55916/scratchpad/waitcore.py}
STEPS=${STEPS:-"protocol_eq14_sampling protocol_kick_timing protocol_clip_level protocol_window protocol_phases protocol_band_holes"}
for s in $STEPS; do
  echo "$(date '+%F %T') start $s"
  if [ -f "$WAIT" ]; then
    python3 "$WAIT" --cores 1 -- env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      python3 verification/paper/$s.py >> verification/paper/$s.log 2>&1
  else
    env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verification/paper/$s.py >> verification/paper/$s.log 2>&1
  fi
  echo "$(date '+%F %T') end $s (exit $?)"
done
