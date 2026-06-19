#!/bin/bash
# Waits for landmarks test_manifest.csv, then stops extract_features (before xception)
# and launches landmarks-only experiments. Safe to run while extraction is in progress.
set -euo pipefail

MANIFEST=/workspace/outputs/features/visual/landmarks/test_manifest.csv
DONE_FLAG=/workspace/logs/.landmarks_experiments_started
LOG=/workspace/logs/pipeline.log

if [ -f "$DONE_FLAG" ]; then
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] waiting for landmarks test_manifest..." >> "$LOG"

while [ ! -f "$MANIFEST" ]; do
    sleep 60
done

# Allow save_stats + manifest flush to finish
sleep 20

LOCK=/workspace/logs/.landmarks_experiments_lock
mkdir "$LOCK" 2>/dev/null || exit 0

touch "$DONE_FLAG"
echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] landmarks test done — skipping xception, starting experiments" >> "$LOG"

pkill -f "python scripts/extract_features.py" 2>/dev/null || true
sleep 5

cd /workspace
PYTHONUNBUFFERED=1 python scripts/run_experiments.py \
    --landmarks-only \
    --config configs/experiment.yaml \
    2>&1 | tee logs/experiments.log

echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] experiments finished" >> "$LOG"
