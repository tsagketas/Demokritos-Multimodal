#!/usr/bin/env bash
set -euo pipefail
cd /workspace

echo "=== [1/2] Re-running landmarks-only experiments (fixed loss) ==="
PYTHONUNBUFFERED=1 python scripts/run_experiments.py --landmarks-only 2>&1 | tee logs/experiments_rerun.log

echo ""
echo "=== [2/2] Explainability on sample videos ==="
FAKE="/workspace/datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/Caucasian (American)/men/id00088/00005_id01124_wavtolip.mp4"
REAL="/workspace/datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio/Asian (East)/women/id03940/00025.mp4"

run_explain() {
  local tag="$1" video="$2" mode="$3"
  shift 3
  echo ""
  echo "--- explain: ${tag} / ${mode} ---"
  PYTHONUNBUFFERED=1 python scripts/explain.py \
    --video "$video" \
    --mode "$mode" \
    --output_dir "/workspace/outputs/explanations/${tag}/${mode}" \
    "$@"
}

explain_video() {
  local tag="$1" video="$2"
  run_explain "$tag" "$video" audio_only \
    --audio_method wav2vec2 \
    --audio_ckpt outputs/experiments/02_audio_only_wav2vec2/checkpoints/best_audio.pt

  run_explain "$tag" "$video" video_only \
    --visual_method landmarks \
    --video_ckpt outputs/experiments/04_video_only_landmarks/checkpoints/best_video.pt

  run_explain "$tag" "$video" early_fusion \
    --audio_method handcrafted \
    --visual_method landmarks \
    --fusion_ckpt outputs/experiments/06_early_handcrafted_landmarks/checkpoints/best_model.pt \
    --pca_ckpt outputs/experiments/06_early_handcrafted_landmarks/pca.pkl

  run_explain "$tag" "$video" late_fusion \
    --audio_method handcrafted \
    --visual_method landmarks \
    --audio_ckpt outputs/experiments/01_audio_only_handcrafted/checkpoints/best_audio.pt \
    --video_ckpt outputs/experiments/04_video_only_landmarks/checkpoints/best_video.pt
}

explain_video fake "$FAKE"
explain_video real "$REAL"

echo ""
echo "=== PIPELINE COMPLETE ==="
