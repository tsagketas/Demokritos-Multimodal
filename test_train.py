"""
Smoke test for src/train.py — synthetic data, no GPU or real features needed.
Uses wav2vec2 (768d) + landmarks (731d) → no PCA required.
Usage: python test_train.py
"""
import csv
import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

def ok(label, value=None):
    print(f"  ✅ {label}" + (f"  →  {value}" if value is not None else ""))

def fail(label, err):
    print(f"  ❌ {label}  →  [{type(err).__name__}] {err}")
    import traceback; traceback.print_exc()


print("\n── TRAIN SMOKE TEST ────────────────────────────────────────────────────")

AUDIO_DIM  = 768   # wav2vec2
VISUAL_DIM = 731   # landmarks (768 ≈ 731 → no PCA)
SPLITS     = {"train": 50, "val": 20, "test": 20}
CATEGORIES = ["FakeVideo-FakeAudio", "RealVideo-RealAudio",
               "FakeVideo-RealAudio", "RealVideo-FakeAudio"]

try:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_train_"))

    audio_dir  = tmp / "features" / "audio" / "wav2vec2"
    visual_dir = tmp / "features" / "visual"
    audio_dir.mkdir(parents=True)
    visual_dir.mkdir(parents=True)

    # ── Synthetic features + manifests ───────────────────────────────────────
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

    ok("Synthetic data created",
       f"train={SPLITS['train']}, val={SPLITS['val']}, test={SPLITS['test']}")

    # ── Minimal config files ──────────────────────────────────────────────────
    cfg_dir = tmp / "configs"
    cfg_dir.mkdir()

    (cfg_dir / "data.yaml").write_text(yaml.dump({
        "dataset": {"root": "", "meta_csv": "", "categories": {"real": [], "fake": []}},
        "split":   {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 42, "stratify": True},
        "audio":   {"sample_rate": 16000, "extract_dir": str(tmp / "features" / "audio")},
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
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "weight_decay":  1e-5,
            "optimizer":     "adam",
            "scheduler":     "cosine",
            "early_stopping_patience": 2,
            "seed":   42,
            "device": "cpu",
        },
        "classifier": {"hidden_dims": [256, 128], "dropout": 0.3, "activation": "relu"},
        "checkpointing": {"save_dir": str(tmp / "checkpoints"), "save_best_only": True,
                          "monitor": "val_auc"},
    }))

    (cfg_dir / "fusion.yaml").write_text(yaml.dump({
        "fusion": {"early": {"enabled": True}, "late": {"enabled": False}},
        "multimodal": {"audio_feature": "wav2vec2", "visual_feature": "landmarks"},
    }))

    (cfg_dir / "eval.yaml").write_text(yaml.dump({
        "evaluation": {"metrics": ["accuracy", "auc", "eer", "f1"],
                       "output_dir": str(tmp / "metrics"),
                       "per_category": True},
    }))

    (cfg_dir / "experiment.yaml").write_text(yaml.dump({
        "experiment": {"name": "smoke_test", "description": "smoke"},
        "configs": {
            "data":     "data.yaml",
            "features": "features.yaml",
            "train":    "train.yaml",
            "fusion":   "fusion.yaml",
            "eval":     "eval.yaml",
        },
    }))

    ok("Config files created", str(cfg_dir))

    # ── Run train.py ──────────────────────────────────────────────────────────
    from utils.config import load_config

    # patch setup_run_dir to use temp dir (avoids hardcoded /workspace path)
    import utils.config as _config_mod
    _orig_setup = _config_mod.setup_run_dir

    def _patched_setup(cfg):
        from datetime import datetime
        name      = cfg["experiment"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = tmp / "runs" / f"{timestamp}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir()
        (run_dir / "plots").mkdir()
        (run_dir / "checkpoints").mkdir()
        import yaml as _yaml
        (run_dir / "config_snapshot.yaml").write_text(_yaml.dump(cfg))
        return run_dir

    _config_mod.setup_run_dir = _patched_setup

    # run
    sys.argv = ["train.py", "--config", str(cfg_dir / "experiment.yaml")]
    from train import main
    main()

    _config_mod.setup_run_dir = _orig_setup  # restore

    # ── Verify outputs ────────────────────────────────────────────────────────
    run_dirs = list((tmp / "runs").iterdir())
    assert run_dirs, "No run directory created"
    run_dir = run_dirs[0]

    ckpt = run_dir / "checkpoints" / "best_model.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    ok("Checkpoint saved", ckpt.name)

    metrics_files = list((run_dir / "metrics").glob("test_*.json"))
    assert metrics_files, "No metrics JSON found"
    import json
    metrics = json.loads(metrics_files[0].read_text())
    assert "auc" in metrics and "f1" in metrics and "per_category" in metrics
    ok("Metrics saved",
       f"auc={metrics['auc']:.4f}, f1={metrics['f1']:.4f}, "
       f"categories={list(metrics['per_category'].keys())}")

    ok("train.py smoke test PASSED")

except Exception as e:
    fail("train.py smoke test", e)

print("\n── DONE ────────────────────────────────────────────────────────────────\n")
