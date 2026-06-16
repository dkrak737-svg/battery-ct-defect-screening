#!/bin/bash
# Train YOLO and auto-Stop the RunPod pod when done (saves GPU $$).
#
# - 'set -o pipefail' so pipe failures propagate exit codes
# - tee logs to persistent volume so we can debug after pod stops
# - Stop on success OR failure (set behavior with the SUCCESS_ONLY env var)
# - Use 'nohup bash vision/train_and_stop.sh &' to survive SSH disconnect.
set -uo pipefail

PROJ=/workspace/battery-ct-security
mkdir -p "$PROJ/models"

LOG="$PROJ/models/train_$(date +%Y%m%d_%H%M).log"
echo "Logging to $LOG"

# Forward all extra args to train.py (e.g., --epochs 30 --batch 32)
python "$PROJ/vision/train.py" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

echo "train.py exit code: $RC"

if [ "${SUCCESS_ONLY:-0}" = "1" ] && [ "$RC" -ne 0 ]; then
    echo "SUCCESS_ONLY=1 set and training failed (rc=$RC) — keeping pod alive for debug."
    exit "$RC"
fi

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "RUNPOD_POD_ID not set — not on RunPod, skipping stop."
    exit "$RC"
fi

echo "Stopping pod $RUNPOD_POD_ID ..."
# Command form may vary by runpodctl version; try both.
runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null \
  || runpodctl pod stop "$RUNPOD_POD_ID" 2>/dev/null \
  || echo "WARN: runpodctl stop failed — stop the pod manually."

exit "$RC"
