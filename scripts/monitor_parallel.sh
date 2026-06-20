#!/usr/bin/env bash
# Writes human-readable status to logs/monitor_status.txt
set -uo pipefail
cd /workspace

OUT=logs/monitor_status.txt
TS=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

{
  echo "=== PARALLEL MONITOR @ $TS ==="
  echo ""

  echo "## Processes"
  ps -ef | grep -E 'run_remaining|run_experiments|explain.py' | grep -v grep || echo "  (none running)"
  echo ""

  echo "## Completed (rerun metrics)"
  for exp in 01_audio_only_handcrafted 02_audio_only_wav2vec2 03_audio_only_hubert \
             04_video_only_landmarks 06_early_handcrafted_landmarks \
             08_early_wav2vec2_landmarks 10_early_hubert_landmarks \
             12_late_handcrafted_landmarks 14_late_wav2vec2_landmarks 16_late_hubert_landmarks; do
    latest=$(ls -t outputs/experiments/$exp/metrics/test_*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
      auc=$(python -c "import json; print(f\"{json.load(open('$latest'))['auc']:.4f}\")" 2>/dev/null || echo "?")
      mtime=$(stat -c '%y' "$latest" 2>/dev/null | cut -d. -f1)
      echo "  OK  $exp  AUC=$auc  ($mtime)"
    else
      echo "  --  $exp  (pending)"
    fi
  done
  echo ""

  echo "## Active experiment logs (last 3 lines each)"
  for f in 08 10 12 14 16; do
    log=logs/exp_${f}.log
    if [ -f "$log" ]; then
      echo "  --- exp_$f ---"
      tail -3 "$log" | sed 's/^/    /'
    fi
  done
  echo ""

  echo "## Pipeline"
  tail -3 logs/parallel_rerun.log 2>/dev/null || echo "  (no parallel_rerun.log)"
  echo ""

  echo "## GPU"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  n/a"
  echo ""

  done_count=$(for exp in 01_audio_only_handcrafted 02_audio_only_wav2vec2 03_audio_only_hubert \
             04_video_only_landmarks 06_early_handcrafted_landmarks \
             08_early_wav2vec2_landmarks 10_early_hubert_landmarks \
             12_late_handcrafted_landmarks 14_late_wav2vec2_landmarks 16_late_hubert_landmarks; do
    ls outputs/experiments/$exp/metrics/test_*.json 2>/dev/null | head -1
  done | wc -l)
  echo "## Summary: $done_count/10 landmarks experiments have test metrics"
} > "$OUT"

cat "$OUT"
