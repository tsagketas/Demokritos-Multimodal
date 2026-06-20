"""
End-to-end smoke test — verifies the full pipeline without full training.

Sections:
  1. Extraction pipeline   (1 synthetic mp4, handcrafted + landmarks)
  2. Experiment runner     (synthetic .npy, 3 experiments × 3 epochs)
  3. Comparison report     (plots, CSV, txt)
  4. Explainability        (explain.py on the synthetic video)

Usage:
    python tests/test_e2e.py
"""
import csv
import json
import sys
import os
import copy
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PASS = 0
FAIL = 0


def ok(label, value=None):
    global PASS
    PASS += 1
    print(f"  ✅ {label}" + (f"  →  {value}" if value is not None else ""))


def fail(label, err):
    global FAIL
    FAIL += 1
    print(f"  ❌ {label}")
    print(f"     {type(err).__name__}: {err}")
    traceback.print_exc()


def check(label, condition, detail=""):
    if condition:
        ok(label, detail if detail else None)
    else:
        fail(label, AssertionError(f"condition False — {detail}"))


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 1 — EXTRACTION PIPELINE ─────────────────────────────────────")

try:
    tmp_ex = Path(tempfile.mkdtemp(prefix="e2e_extract_"))
    mp4    = tmp_ex / "smoke_test.mp4"

    # 1a. Create synthetic mp4
    try:
        result = subprocess.run([
            "ffmpeg", "-f", "lavfi",
            "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-y", str(mp4)
        ], capture_output=True, timeout=30)
        check("ffmpeg creates synthetic mp4", mp4.exists(), str(mp4))
    except Exception as e:
        fail("ffmpeg creates synthetic mp4", e)
        raise

    # 1b. Audio extraction — handcrafted (no model download)
    try:
        from data.preprocessing import extract_audio
        from features.audio.handcrafted import extract as handcrafted_extract, FEATURE_DIM as A_DIM

        waveform = extract_audio(str(mp4), sample_rate=16000)
        check("extract_audio from mp4", waveform.shape[0] == 1, f"shape={tuple(waveform.shape)}")

        audio_cfg = {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512}
        vec = handcrafted_extract(waveform, 16000, audio_cfg)
        check("handcrafted audio → 536d vector", vec.shape == (A_DIM,), f"shape={vec.shape}")

        npy_path = tmp_ex / "audio_feat.npy"
        np.save(npy_path, vec)
        check("audio .npy saved", npy_path.exists(), str(npy_path))
    except Exception as e:
        fail("audio extraction", e)

    # 1c. Visual extraction — landmarks (no face → zeros fallback)
    try:
        from data.preprocessing import extract_frames
        from features.visual.landmark_features import extract as lm_extract, FEATURE_DIM as V_DIM

        frames = extract_frames(str(mp4), max_frames=10)
        check("extract_frames from mp4", len(frames) > 0, f"{len(frames)} frames")

        lm_vec = lm_extract(frames, {})
        check("landmark extract → 731d (zeros ok)", lm_vec.shape == (V_DIM,), f"shape={lm_vec.shape}")

        npy_v = tmp_ex / "visual_feat.npy"
        np.save(npy_v, lm_vec)
        check("visual .npy saved", npy_v.exists())
    except Exception as e:
        fail("visual extraction", e)

    # 1d. Manifest creation
    try:
        manifest = tmp_ex / "train_manifest.csv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["feature_path", "label", "mp4_path"])
            w.writeheader()
            w.writerow({"feature_path": str(npy_path), "label": 1,
                         "mp4_path": str(mp4)})
        check("manifest CSV created", manifest.exists())
    except Exception as e:
        fail("manifest creation", e)

except Exception as e:
    fail("Section 1 setup", e)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 2 — EXPERIMENT RUNNER (3 experiments × 3 epochs) ────────────")

AUDIO_DIM  = 536   # handcrafted
VISUAL_DIM = 731   # landmarks
SPLITS     = {"train": 40, "val": 16, "test": 16}
CATEGORIES = ["FakeVideo-FakeAudio", "RealVideo-RealAudio",
               "FakeVideo-RealAudio", "RealVideo-FakeAudio"]

try:
    tmp_exp = Path(tempfile.mkdtemp(prefix="e2e_exp_"))
    audio_dir  = tmp_exp / "features" / "audio" / "handcrafted"
    visual_dir = tmp_exp / "features" / "visual" / "landmarks"
    audio_dir.mkdir(parents=True)
    visual_dir.mkdir(parents=True)

    # build synthetic .npy + manifests
    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["feature_path", "label", "mp4_path"])
            w.writeheader(); w.writerows(rows)

    for split, n in SPLITS.items():
        a_rows, v_rows = [], []
        for i in range(n):
            cat      = CATEGORIES[i % len(CATEGORIES)]
            mp4_path = f"/FakeAVCeleb/{cat}/video_{split}_{i:03d}.mp4"
            label    = 0 if cat == "RealVideo-RealAudio" else 1
            ap = audio_dir  / f"video_{split}_{i:03d}.npy"
            vp = visual_dir / f"video_{split}_{i:03d}_landmarks.npy"
            np.save(ap, np.random.randn(AUDIO_DIM).astype(np.float32))
            np.save(vp, np.random.randn(VISUAL_DIM).astype(np.float32))
            a_rows.append({"feature_path": str(ap), "label": label, "mp4_path": mp4_path})
            v_rows.append({"feature_path": str(vp), "label": label, "mp4_path": mp4_path})
        write_csv(audio_dir  / f"{split}_manifest.csv", a_rows)
        write_csv(visual_dir / f"{split}_manifest.csv", v_rows)

    check("synthetic .npy + manifests created",
          all((audio_dir / f"{s}_manifest.csv").exists() for s in SPLITS))

    # minimal config
    def make_cfg():
        return {
            "experiment": {"name": "smoke_e2e"},
            "data": {
                "audio":   {"sample_rate": 16000,
                             "extract_dir": str(audio_dir.parent)},
                "video":   {"max_frames": 10},
            },
            "features": {
                "audio":  {"handcrafted": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512}},
                "visual": {"method": "landmarks", "cache_dir": str(visual_dir.parent)},
            },
            "train": {
                "training": {
                    "epochs": 3, "batch_size": 16,
                    "learning_rate": 1e-4, "weight_decay": 1e-5,
                    "optimizer": "adam", "scheduler": "cosine",
                    "early_stopping_patience": 2, "seed": 42, "device": "cpu",
                },
                "classifier": {"hidden_dims": [256, 128], "dropout": 0.3},
            },
            "fusion": {
                "fusion": {"late": {"strategy": "average", "weight": 0.5}},
                "multimodal": {"audio_feature": "handcrafted",
                               "visual_feature": "landmarks"},
            },
        }

    from data.feature_dataset import FeatureDataset
    from models.pca_reducer import VisualPCAReducer, get_n_components
    from models.early_fusion import EarlyFusionMLP
    from models.unimodal_classifier import UnimodalClassifier
    from train import (run_early_fusion, run_unimodal, run_late_fusion,
                       collate_fn, make_weighted_train_loader, collect_visual)

    device = torch.device("cpu")

    def make_loaders(cfg):
        am = cfg["fusion"]["multimodal"]["audio_feature"]
        vm = cfg["fusion"]["multimodal"]["visual_feature"]
        a_dir = Path(cfg["data"]["audio"]["extract_dir"]) / am
        v_dir = Path(cfg["features"]["visual"]["cache_dir"]) / vm

        def ds(split):
            return FeatureDataset(str(a_dir / f"{split}_manifest.csv"),
                                  str(v_dir / f"{split}_manifest.csv"))
        train_ds = ds("train"); val_ds = ds("val"); test_ds = ds("test")
        bs = cfg["train"]["training"]["batch_size"]
        return (train_ds, val_ds, test_ds,
                make_weighted_train_loader(train_ds, bs),
                DataLoader(val_ds,  bs, shuffle=False, collate_fn=collate_fn, num_workers=0),
                DataLoader(test_ds, bs, shuffle=False, collate_fn=collate_fn, num_workers=0))

    unimodal_ckpts = {}

    # ── Experiment A: audio_only_handcrafted ──────────────────────────────────
    try:
        cfg = make_cfg()
        cfg["features"]["visual"]["cache_dir"] = str(visual_dir.parent)
        run_dir = tmp_exp / "01_audio_only_handcrafted"
        for d in ("metrics", "plots", "checkpoints"): (run_dir / d).mkdir(parents=True, exist_ok=True)

        train_ds, val_ds, test_ds, tl, vl, tel = make_loaders(cfg)
        reducer = VisualPCAReducer(n_components=None)
        common  = dict(cfg=cfg, run_dir=run_dir, train_ds=train_ds, val_ds=val_ds,
                       test_ds=test_ds, train_loader=tl, val_loader=vl,
                       test_loader=tel, reducer=reducer, device=device)
        run_unimodal(**common, modality="audio")

        ckpt = run_dir / "checkpoints" / "best_audio.pt"
        check("audio_only — checkpoint saved", ckpt.exists())
        check("audio_only — metrics JSON saved",
              any((run_dir / "metrics").glob("test_*.json")))
        check("audio_only — ROC curve png",
              (run_dir / "plots").glob("*roc_curve.png") and
              any((run_dir / "plots").glob("*roc_curve.png")))
        check("audio_only — ROC curve txt",
              any((run_dir / "plots").glob("*roc_curve.txt")))
        unimodal_ckpts[("audio", "handcrafted")] = ckpt
    except Exception as e:
        fail("audio_only experiment", e)

    # ── Experiment B: video_only_landmarks ────────────────────────────────────
    try:
        cfg = make_cfg()
        cfg["features"]["visual"]["cache_dir"] = str(visual_dir.parent)
        run_dir = tmp_exp / "04_video_only_landmarks"
        for d in ("metrics", "plots", "checkpoints"): (run_dir / d).mkdir(parents=True, exist_ok=True)

        train_ds, val_ds, test_ds, tl, vl, tel = make_loaders(cfg)
        reducer = VisualPCAReducer(n_components=None)
        common  = dict(cfg=cfg, run_dir=run_dir, train_ds=train_ds, val_ds=val_ds,
                       test_ds=test_ds, train_loader=tl, val_loader=vl,
                       test_loader=tel, reducer=reducer, device=device)
        run_unimodal(**common, modality="video")

        ckpt = run_dir / "checkpoints" / "best_video.pt"
        check("video_only — checkpoint saved", ckpt.exists())
        check("video_only — metrics JSON saved",
              any((run_dir / "metrics").glob("test_*.json")))
        check("video_only — confusion matrix png",
              any((run_dir / "plots").glob("*confusion_matrix.png")))
        check("video_only — per_category bar txt",
              any((run_dir / "plots").glob("*per_category_bar.txt")))
        unimodal_ckpts[("video", "landmarks")] = ckpt
    except Exception as e:
        fail("video_only experiment", e)

    # ── Experiment C: early_handcrafted_landmarks ─────────────────────────────
    try:
        cfg = make_cfg()
        cfg["features"]["visual"]["cache_dir"] = str(visual_dir.parent)
        run_dir = tmp_exp / "06_early_handcrafted_landmarks"
        for d in ("metrics", "plots", "checkpoints"): (run_dir / d).mkdir(parents=True, exist_ok=True)

        train_ds, val_ds, test_ds, tl, vl, tel = make_loaders(cfg)
        n_components = get_n_components("handcrafted", "landmarks")
        reducer = VisualPCAReducer(n_components=n_components)
        if reducer.needed:
            reducer.fit(collect_visual(train_ds))
            reducer.save(run_dir / "pca.pkl")
        common = dict(cfg=cfg, run_dir=run_dir, train_ds=train_ds, val_ds=val_ds,
                      test_ds=test_ds, train_loader=tl, val_loader=vl,
                      test_loader=tel, reducer=reducer, device=device)
        run_early_fusion(**common, audio_method="handcrafted", visual_method="landmarks")

        ckpt = run_dir / "checkpoints" / "best_model.pt"
        check("early_fusion — checkpoint saved", ckpt.exists())
        check("early_fusion — metrics JSON saved",
              any((run_dir / "metrics").glob("test_*.json")))
        check("early_fusion — ROC png + txt", (
              any((run_dir / "plots").glob("*roc_curve.png")) and
              any((run_dir / "plots").glob("*roc_curve.txt"))))
    except Exception as e:
        fail("early_fusion experiment", e)

except Exception as e:
    fail("Section 2 setup", e)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 3 — COMPARISON REPORT ───────────────────────────────────────")

try:
    from utils.plots import plot_comparison, plot_per_category_heatmap

    all_results = []
    for exp_name in ["01_audio_only_handcrafted", "04_video_only_landmarks",
                     "06_early_handcrafted_landmarks"]:
        metrics_files = sorted((tmp_exp / exp_name / "metrics").glob("test_*.json"))
        if metrics_files:
            m = json.loads(metrics_files[-1].read_text())
            all_results.append({
                "name": exp_name,
                **{k: v for k, v in m.items() if k != "per_category"},
                "per_category": m.get("per_category", {}),
            })

    check("collected results for comparison", len(all_results) == 3,
          f"{len(all_results)}/3")

    cmp_dir = tmp_exp / "comparison"
    cmp_dir.mkdir(exist_ok=True)

    plot_comparison(all_results, cmp_dir)
    check("comparison_auc.png",        (cmp_dir / "comparison_auc.png").exists())
    check("comparison_all.txt",        (cmp_dir / "comparison_all.txt").exists())

    plot_per_category_heatmap(all_results, cmp_dir, metric="auc")
    # AUC heatmap is skipped when all categories are single-class (expected for this dataset)
    check("per_category_heatmap_auc — ran without error", True)

    # CSV
    cols = ["name", "accuracy", "auc", "eer", "f1", "precision", "recall"]
    with open(cmp_dir / "all_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_results)
    check("all_results.csv",           (cmp_dir / "all_results.csv").exists())

    # show ranked table preview
    txt = (cmp_dir / "comparison_all.txt").read_text().split("\n")
    for line in txt[:8]:
        print(f"    {line}")

except Exception as e:
    fail("Section 3 comparison", e)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 4 — EXPLAINABILITY ───────────────────────────────────────────")

try:
    from data.preprocessing import extract_audio
    from models.unimodal_classifier import UnimodalClassifier

    audio_ckpt = tmp_exp / "01_audio_only_handcrafted" / "checkpoints" / "best_audio.pt"
    out_dir    = tmp_exp / "explanations" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data  = torch.load(str(audio_ckpt), map_location="cpu")
    model      = UnimodalClassifier(input_dim=ckpt_data["input_dim"],
                                    hidden_dims=ckpt_data.get("hidden_dims", [256, 128]))
    model.load_state_dict(ckpt_data["model_state"])
    model.eval()

    waveform  = extract_audio(str(mp4), sample_rate=16000)
    y         = waveform.squeeze().numpy()
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    WINDOW = 8000; STEP = 1600
    from features.audio.handcrafted import extract as hc_extract
    audio_cfg = {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512}
    times, scores = [], []
    start = 0
    while start + WINDOW <= len(y):
        window = torch.from_numpy(y[start:start + WINDOW]).unsqueeze(0)
        vec    = hc_extract(window, 16000, audio_cfg)
        with torch.no_grad():
            score = torch.sigmoid(model(torch.from_numpy(vec).float().unsqueeze(0))).item()
        times.append((start + WINDOW / 2) / 16000)
        scores.append(score)
        start += STEP

    check("sliding window — windows generated", len(times) > 0, f"{len(times)} windows")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, scores, color="#3498db", lw=1.5)
    ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("P(fake)"); ax.set_ylim(0, 1)
    ax.set_title("Smoke test — Audio Temporal Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "audio_heatmap.png", dpi=100)
    plt.close(fig)
    check("audio_heatmap.png saved", (out_dir / "audio_heatmap.png").exists())

    with open(out_dir / "audio_heatmap.txt", "w") as f:
        f.write("AUDIO TEMPORAL HEATMAP — smoke_test\n\n")
        f.write(f"{'Time (s)':<12} {'P(fake)':<10} {'Label'}\n")
        f.write("-" * 34 + "\n")
        for t, s in zip(times, scores):
            f.write(f"{t:<12.2f} {s:<10.4f} {'FAKE' if s > 0.5 else 'real'}\n")
    check("audio_heatmap.txt saved", (out_dir / "audio_heatmap.txt").exists())

    summary = {"video": str(mp4), "mode": "audio_only", "windows": len(times),
               "mean_score": round(float(np.mean(scores)), 4)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    check("summary.json saved", (out_dir / "summary.json").exists())

except Exception as e:
    fail("explainability smoke test", e)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── FINAL REPORT ─────────────────────────────────────────────────────────")
print(f"  Passed: {PASS}   Failed: {FAIL}")
if FAIL == 0:
    print("  ALL CHECKS PASSED ✅")
else:
    print(f"  {FAIL} CHECK(S) FAILED ❌")
print(f"  Temp dirs: {tmp_ex}  |  {tmp_exp}\n")
