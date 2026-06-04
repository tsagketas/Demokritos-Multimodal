"""
Smoke tests για src/train.py — όλα τα modes.
Synthetic data, χωρίς GPU ή πραγματικά features.
Usage: python test_train.py
"""
import csv
import json
import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

def ok(label, value=None):
    print(f"  ✅ {label}" + (f"  →  {value}" if value is not None else ""))

def fail(label, err):
    print(f"  ❌ {label}  →  [{type(err).__name__}] {err}")
    import traceback; traceback.print_exc()


# ── Synthetic data setup ──────────────────────────────────────────────────────
AUDIO_DIM  = 768   # wav2vec2
VISUAL_DIM = 731   # landmarks (no PCA needed: 768 ≈ 731)
SPLITS     = {"train": 50, "val": 20, "test": 20}
CATEGORIES = ["FakeVideo-FakeAudio", "RealVideo-RealAudio",
               "FakeVideo-RealAudio", "RealVideo-FakeAudio"]

def build_synthetic_env(tmp: Path):
    """Create fake .npy files and manifest CSVs. Returns (audio_dir, visual_dir)."""
    audio_dir  = tmp / "features" / "audio" / "wav2vec2"
    visual_dir = tmp / "features" / "visual"
    audio_dir.mkdir(parents=True)
    visual_dir.mkdir(parents=True)

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["feature_path", "label", "mp4_path"])
            w.writeheader(); w.writerows(rows)

    for split, n in SPLITS.items():
        audio_rows, visual_rows = [], []
        for i in range(n):
            cat   = CATEGORIES[i % len(CATEGORIES)]
            mp4   = f"/data/{cat}/video_{split}_{i:03d}.mp4"
            label = 1 if "Fake" in cat else 0
            a_path = audio_dir  / f"video_{split}_{i:03d}.npy"
            v_path = visual_dir / f"video_{split}_{i:03d}_landmarks.npy"
            np.save(a_path, np.random.randn(AUDIO_DIM).astype(np.float32))
            np.save(v_path, np.random.randn(VISUAL_DIM).astype(np.float32))
            audio_rows.append({"feature_path": str(a_path),  "label": label, "mp4_path": mp4})
            visual_rows.append({"feature_path": str(v_path), "label": label, "mp4_path": mp4})
        write_csv(audio_dir  / f"{split}_manifest.csv", audio_rows)
        write_csv(visual_dir / f"{split}_manifest.csv", visual_rows)

    return audio_dir, visual_dir


def build_configs(tmp: Path, audio_dir: Path, visual_dir: Path) -> Path:
    """Write minimal yaml config files. Returns path to experiment.yaml."""
    cfg_dir = tmp / "configs"
    cfg_dir.mkdir()

    (cfg_dir / "data.yaml").write_text(yaml.dump({
        "dataset": {"root": "", "meta_csv": "", "categories": {"real": [], "fake": []}},
        "split":   {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42, "stratify": True},
        "audio":   {"sample_rate": 16000,
                    "extract_dir": str(audio_dir.parent)},  # .../features/audio
        "video":   {"fps": 25, "max_frames": 100},
    }))

    (cfg_dir / "features.yaml").write_text(yaml.dump({
        "audio": {
            "handcrafted": {"enabled": False},
            "wav2vec2":    {"enabled": True, "model_name": "facebook/wav2vec2-base",
                            "layer": -1, "pooling": "mean"},
            "hubert":      {"enabled": False},
        },
        "visual": {
            "method":    "landmarks",
            "enabled":   True,
            "cache_dir": str(visual_dir),
            "pretrained": True,
        },
    }))

    (cfg_dir / "train.yaml").write_text(yaml.dump({
        "training": {
            "epochs": 3, "batch_size": 16,
            "learning_rate": 1e-4, "weight_decay": 1e-5,
            "optimizer": "adam", "scheduler": "cosine",
            "early_stopping_patience": 2, "seed": 42, "device": "cpu",
        },
        "classifier": {"hidden_dims": [256, 128], "dropout": 0.3, "activation": "relu"},
        "checkpointing": {"save_best_only": True, "monitor": "val_auc"},
    }))

    (cfg_dir / "fusion.yaml").write_text(yaml.dump({
        "fusion": {
            "early": {"enabled": True},
            "late":  {"enabled": True, "strategy": "average", "weight": 0.5},
        },
        "multimodal": {"audio_feature": "wav2vec2", "visual_feature": "landmarks"},
    }))

    (cfg_dir / "eval.yaml").write_text(yaml.dump({
        "evaluation": {"metrics": ["accuracy", "auc", "eer", "f1"], "per_category": True},
    }))

    (cfg_dir / "experiment.yaml").write_text(yaml.dump({
        "experiment": {"name": "smoke_test"},
        "configs": {
            "data": "data.yaml", "features": "features.yaml",
            "train": "train.yaml", "fusion": "fusion.yaml", "eval": "eval.yaml",
        },
    }))

    return cfg_dir / "experiment.yaml"


def patch_setup_run_dir(tmp: Path):
    """Redirect run dirs to tmp instead of /workspace."""
    import utils.config as _cfg_mod
    orig    = _cfg_mod.setup_run_dir
    counter = [0]

    def _patched(cfg):
        from datetime import datetime
        counter[0] += 1
        name    = cfg["experiment"]["name"]
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = tmp / "runs" / f"{ts}_{counter[0]}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("metrics", "plots", "checkpoints"):
            (run_dir / sub).mkdir(exist_ok=True)
        import yaml as _y
        (run_dir / "config_snapshot.yaml").write_text(_y.dump(cfg))
        return run_dir

    _cfg_mod.setup_run_dir = _patched
    return orig, _cfg_mod


def restore_setup_run_dir(orig, mod):
    mod.setup_run_dir = orig


def last_run_dir(tmp: Path) -> Path:
    runs = sorted((tmp / "runs").iterdir())
    return runs[-1]


def assert_metrics_saved(run_dir: Path, tag_prefix: str) -> dict:
    files = list((run_dir / "metrics").glob(f"{tag_prefix}*.json"))
    assert files, f"No metrics JSON matching '{tag_prefix}*' in {run_dir / 'metrics'}"
    return json.loads(files[0].read_text())


# ═════════════════════════════════════════════════════════════════════════════
print("\n── SETUP ────────────────────────────────────────────────────────────────")
tmp = Path(tempfile.mkdtemp(prefix="smoke_all_"))
audio_dir, visual_dir = build_synthetic_env(tmp)
exp_yaml = build_configs(tmp, audio_dir, visual_dir)
ok("Synthetic env created", str(tmp))

orig_setup, cfg_mod = patch_setup_run_dir(tmp)

from train import main as train_main


# ═════════════════════════════════════════════════════════════════════════════
print("\n── MODE: audio_only ─────────────────────────────────────────────────────")
try:
    sys.argv = ["train.py", "--mode", "audio_only", "--config", str(exp_yaml)]
    train_main()
    run_dir     = last_run_dir(tmp)
    audio_ckpt  = run_dir / "checkpoints" / "best_audio.pt"
    assert audio_ckpt.exists(), "best_audio.pt not found"
    ckpt        = torch.load(audio_ckpt, map_location="cpu")
    assert ckpt["input_dim"] == AUDIO_DIM
    assert ckpt["modality"]  == "audio"
    metrics     = assert_metrics_saved(run_dir, "test_audio_only")
    assert "auc" in metrics and "per_category" in metrics
    ok("audio_only", f"input_dim={ckpt['input_dim']}, auc={metrics['auc']:.4f}")
    AUDIO_CKPT_PATH = str(audio_ckpt)
except Exception as e:
    fail("audio_only", e)
    AUDIO_CKPT_PATH = None


# ═════════════════════════════════════════════════════════════════════════════
print("\n── MODE: video_only ─────────────────────────────────────────────────────")
try:
    sys.argv = ["train.py", "--mode", "video_only", "--config", str(exp_yaml)]
    train_main()
    run_dir     = last_run_dir(tmp)
    video_ckpt  = run_dir / "checkpoints" / "best_video.pt"
    assert video_ckpt.exists(), "best_video.pt not found"
    ckpt        = torch.load(video_ckpt, map_location="cpu")
    assert ckpt["input_dim"] == VISUAL_DIM
    assert ckpt["modality"]  == "video"
    metrics     = assert_metrics_saved(run_dir, "test_video_only")
    assert "auc" in metrics and "per_category" in metrics
    ok("video_only", f"input_dim={ckpt['input_dim']}, auc={metrics['auc']:.4f}")
    VIDEO_CKPT_PATH = str(video_ckpt)
except Exception as e:
    fail("video_only", e)
    VIDEO_CKPT_PATH = None


# ═════════════════════════════════════════════════════════════════════════════
print("\n── MODE: early_fusion ───────────────────────────────────────────────────")
try:
    sys.argv = ["train.py", "--mode", "early_fusion", "--config", str(exp_yaml)]
    train_main()
    run_dir = last_run_dir(tmp)
    ckpt    = run_dir / "checkpoints" / "best_model.pt"
    assert ckpt.exists(), "best_model.pt not found"
    data    = torch.load(ckpt, map_location="cpu")
    assert data["audio_dim"]  == AUDIO_DIM
    assert data["visual_dim"] == VISUAL_DIM  # no PCA: wav2vec2+landmarks
    metrics = assert_metrics_saved(run_dir, "test_early_fusion")
    assert "auc" in metrics and "per_category" in metrics
    ok("early_fusion", f"input_dim={AUDIO_DIM + VISUAL_DIM}, auc={metrics['auc']:.4f}")
except Exception as e:
    fail("early_fusion", e)


# ═════════════════════════════════════════════════════════════════════════════
print("\n── MODE: late_fusion (average) ──────────────────────────────────────────")
if AUDIO_CKPT_PATH and VIDEO_CKPT_PATH:
    try:
        sys.argv = [
            "train.py", "--mode", "late_fusion", "--config", str(exp_yaml),
            "--audio_ckpt", AUDIO_CKPT_PATH,
            "--video_ckpt", VIDEO_CKPT_PATH,
        ]
        train_main()
        run_dir = last_run_dir(tmp)
        metrics = assert_metrics_saved(run_dir, "test_late_fusion_average")
        assert "auc" in metrics and "per_category" in metrics
        ok("late_fusion (average)", f"auc={metrics['auc']:.4f}")
    except Exception as e:
        fail("late_fusion (average)", e)
else:
    print("  ⚠️  skipped — audio_only or video_only failed")


# ═════════════════════════════════════════════════════════════════════════════
print("\n── MODE: late_fusion (learned) ──────────────────────────────────────────")
if AUDIO_CKPT_PATH and VIDEO_CKPT_PATH:
    try:
        # Switch fusion config to learned strategy
        fusion_cfg_path = exp_yaml.parent / "fusion.yaml"
        fusion_cfg_path.write_text(yaml.dump({
            "fusion": {
                "early": {"enabled": True},
                "late":  {"enabled": True, "strategy": "learned", "weight": 0.5},
            },
            "multimodal": {"audio_feature": "wav2vec2", "visual_feature": "landmarks"},
        }))

        sys.argv = [
            "train.py", "--mode", "late_fusion", "--config", str(exp_yaml),
            "--audio_ckpt", AUDIO_CKPT_PATH,
            "--video_ckpt", VIDEO_CKPT_PATH,
        ]
        train_main()
        run_dir = last_run_dir(tmp)
        ckpt    = run_dir / "checkpoints" / "best_combiner.pt"
        assert ckpt.exists(), "best_combiner.pt not found"
        metrics = assert_metrics_saved(run_dir, "test_late_fusion_learned")
        assert "auc" in metrics and "per_category" in metrics
        ok("late_fusion (learned)", f"auc={metrics['auc']:.4f}")
    except Exception as e:
        fail("late_fusion (learned)", e)
else:
    print("  ⚠️  skipped — audio_only or video_only failed")


# ─────────────────────────────────────────────────────────────────────────────
restore_setup_run_dir(orig_setup, cfg_mod)
print("\n── DONE ────────────────────────────────────────────────────────────────\n")
