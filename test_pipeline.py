"""
Smoke tests — τρέχει κάθε βήμα με ένα πραγματικό MP4 και ελέγχει shapes/dims.
Χρήση: python test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

# ── sample video ──────────────────────────────────────────────────────────────
SAMPLE = Path(
    "datasets/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076"
    "/00109_10_id00476_wavtolip.mp4"
)

def ok(label, value=None):
    print(f"  ✅ {label}" + (f"  →  {value}" if value is not None else ""))

def fail(label, err):
    print(f"  ❌ {label}  →  {err}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── PREPROCESSING ──────────────────────────────────────────────────────")

from data.preprocessing import extract_audio, extract_frames

try:
    waveform = extract_audio(str(SAMPLE))
    assert waveform.ndim == 2 and waveform.shape[0] == 1
    ok("extract_audio", f"shape={tuple(waveform.shape)}, dtype={waveform.dtype}")
except Exception as e:
    fail("extract_audio", e)
    waveform = None

try:
    frames = extract_frames(str(SAMPLE))
    assert isinstance(frames, list) and len(frames) > 0
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
    ok("extract_frames", f"n_frames={len(frames)}, frame_shape={frames[0].shape}")
except Exception as e:
    fail("extract_frames", e)
    frames = None


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUDIO FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── AUDIO FEATURE EXTRACTION ───────────────────────────────────────────")

if waveform is not None:

    # Handcrafted
    try:
        from features.audio import handcrafted
        vec = handcrafted.extract(waveform, 16000, {})
        assert vec.shape == (handcrafted.FEATURE_DIM,), f"expected {handcrafted.FEATURE_DIM}, got {vec.shape}"
        ok("handcrafted", f"shape={vec.shape}")
    except Exception as e:
        fail("handcrafted", e)

    # wav2vec2
    try:
        from features.audio import wav2vec2
        vec = wav2vec2.extract(waveform, 16000, {})
        assert vec.shape == (wav2vec2.FEATURE_DIM,), f"expected {wav2vec2.FEATURE_DIM}, got {vec.shape}"
        ok("wav2vec2", f"shape={vec.shape}")
    except Exception as e:
        fail("wav2vec2", e)

    # HuBERT
    try:
        from features.audio import hubert
        vec = hubert.extract(waveform, 16000, {})
        assert vec.shape == (hubert.FEATURE_DIM,), f"expected {hubert.FEATURE_DIM}, got {vec.shape}"
        ok("hubert", f"shape={vec.shape}")
    except Exception as e:
        fail("hubert", e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUAL FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VISUAL FEATURE EXTRACTION ──────────────────────────────────────────")

if frames is not None:

    # Landmarks
    try:
        from features.visual import landmark_features
        vec = landmark_features.extract(frames, {})
        assert vec.shape == (landmark_features.FEATURE_DIM,), f"expected {landmark_features.FEATURE_DIM}, got {vec.shape}"
        ok("landmark_features", f"shape={vec.shape}")
    except Exception as e:
        fail("landmark_features", e)

    # ResNet18
    try:
        from features.visual import resnet_features
        vec = resnet_features.extract(frames, {})
        assert vec.shape == (resnet_features.FEATURE_DIM,), f"expected {resnet_features.FEATURE_DIM}, got {vec.shape}"
        ok("resnet_features", f"shape={vec.shape}")
    except Exception as e:
        fail("resnet_features", e)


print("\n── DONE ────────────────────────────────────────────────────────────────\n")
