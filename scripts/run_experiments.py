"""
Run all 17 experiments and generate comparison report.
Requires extract_features.py to have been run first.

Usage:
    python run_experiments.py
    python run_experiments.py --config configs/experiment.yaml
    python run_experiments.py --skip_done   # skip experiments with existing results
"""
import sys
import json
import copy
import argparse
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.feature_dataset import FeatureDataset
from models.pca_reducer import VisualPCAReducer, get_n_components
from train import (
    run_early_fusion, run_unimodal, run_late_fusion,
    collate_fn, make_weighted_train_loader, collect_visual,
)
from utils.config import load_config
from utils.plots import plot_comparison, plot_per_category_heatmap

# ── Experiment matrix ─────────────────────────────────────────────────────────
# (exp_id, mode, audio_method, visual_method, exp_name)
# For unimodal: the unused modality still needs a valid manifest for the inner join.
#   audio_only → pairs with "landmarks" (fastest visual FE)
#   video_only → pairs with "wav2vec2"  (any audio FE)
EXPERIMENTS = [
    ( 1, "audio_only",   "handcrafted", "landmarks", "01_audio_only_handcrafted"),
    ( 2, "audio_only",   "wav2vec2",    "landmarks", "02_audio_only_wav2vec2"),
    ( 3, "audio_only",   "hubert",      "landmarks", "03_audio_only_hubert"),
    ( 4, "video_only",   "wav2vec2",    "landmarks", "04_video_only_landmarks"),
    ( 5, "video_only",   "wav2vec2",    "xception",  "05_video_only_xception"),
    ( 6, "early_fusion", "handcrafted", "landmarks", "06_early_handcrafted_landmarks"),
    ( 7, "early_fusion", "handcrafted", "xception",  "07_early_handcrafted_xception"),
    ( 8, "early_fusion", "wav2vec2",    "landmarks", "08_early_wav2vec2_landmarks"),
    ( 9, "early_fusion", "wav2vec2",    "xception",  "09_early_wav2vec2_xception"),
    (10, "early_fusion", "hubert",      "landmarks", "10_early_hubert_landmarks"),
    (11, "early_fusion", "hubert",      "xception",  "11_early_hubert_xception"),
    (12, "late_fusion",  "handcrafted", "landmarks", "12_late_handcrafted_landmarks"),
    (13, "late_fusion",  "handcrafted", "xception",  "13_late_handcrafted_xception"),
    (14, "late_fusion",  "wav2vec2",    "landmarks", "14_late_wav2vec2_landmarks"),
    (15, "late_fusion",  "wav2vec2",    "xception",  "15_late_wav2vec2_xception"),
    (16, "late_fusion",  "hubert",      "landmarks", "16_late_hubert_landmarks"),
    (17, "late_fusion",  "hubert",      "xception",  "17_late_hubert_xception"),
]


# ── Single experiment runner ──────────────────────────────────────────────────

def run_single(cfg_base: dict, mode: str, audio_method: str, visual_method: str,
               run_dir: Path, unimodal_ckpts: dict, device: torch.device) -> dict:
    """
    Sets up datasets/dataloaders for one experiment and dispatches to train.py runners.
    Returns the metrics dict loaded from the saved JSON.
    """
    cfg = copy.deepcopy(cfg_base)
    cfg["fusion"]["multimodal"]["audio_feature"]  = audio_method
    cfg["fusion"]["multimodal"]["visual_feature"] = visual_method

    # deeper MLP for xception (2048d visual) to avoid 8× compression in one step
    if visual_method == "xception":
        cfg["train"]["classifier"]["hidden_dims"] = [512, 256, 128]

    train_cfg = cfg["train"]["training"]
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    # ── Paths ─────────────────────────────────────────────────────────────────
    audio_dir  = Path(cfg["data"]["audio"]["extract_dir"]) / audio_method
    visual_dir = Path(cfg["features"]["visual"]["cache_dir"]) / visual_method

    # ── Datasets ──────────────────────────────────────────────────────────────
    def make_ds(split):
        return FeatureDataset(
            str(audio_dir  / f"{split}_manifest.csv"),
            str(visual_dir / f"{split}_manifest.csv"),
        )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"  samples — train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── PCA reducer ───────────────────────────────────────────────────────────
    if mode == "early_fusion":
        n_components = get_n_components(audio_method, visual_method)
        reducer = VisualPCAReducer(n_components=n_components)
        if reducer.needed:
            print(f"  fitting PCA {visual_method} → {n_components}d ...")
            reducer.fit(collect_visual(train_ds))
            reducer.save(run_dir / "pca.pkl")
        else:
            print(f"  no PCA needed ({audio_method} + {visual_method})")
    else:
        reducer = VisualPCAReducer(n_components=None)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs = train_cfg["batch_size"]
    train_loader = make_weighted_train_loader(train_ds, batch_size=bs, target_ratio=5)
    val_loader   = DataLoader(val_ds,  batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    common = dict(
        cfg=cfg, run_dir=run_dir,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        reducer=reducer, device=device,
    )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if mode == "early_fusion":
        run_early_fusion(**common, audio_method=audio_method, visual_method=visual_method)

    elif mode == "audio_only":
        run_unimodal(**common, modality="audio")

    elif mode == "video_only":
        run_unimodal(**common, modality="video")

    elif mode == "late_fusion":
        audio_ckpt = unimodal_ckpts.get(("audio", audio_method))
        video_ckpt = unimodal_ckpts.get(("video", visual_method))
        if not audio_ckpt or not video_ckpt:
            raise RuntimeError(
                f"Missing unimodal checkpoints for late_fusion "
                f"(audio={audio_method}, visual={visual_method}). "
                f"Run unimodal experiments first."
            )
        run_late_fusion(**common,
                        audio_ckpt_path=str(audio_ckpt),
                        video_ckpt_path=str(video_ckpt))

    # ── Load saved metrics ────────────────────────────────────────────────────
    metrics_files = sorted((run_dir / "metrics").glob("test_*.json"))
    if not metrics_files:
        raise RuntimeError("No metrics JSON found after training.")
    return json.loads(metrics_files[-1].read_text())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/experiment.yaml")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip experiments whose run_dir already exists")
    args = parser.parse_args()

    cfg_base    = load_config(args.config)
    experiments_dir = Path("/workspace/outputs/experiments")
    comparison_dir  = Path("/workspace/outputs/comparison")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[run_experiments] device={device}  total={len(EXPERIMENTS)} experiments\n")

    unimodal_ckpts = {}   # {("audio"|"video", method): Path}
    all_results    = []
    failed         = []

    for exp_id, mode, audio_method, visual_method, exp_name in EXPERIMENTS:
        run_dir = experiments_dir / exp_name
        banner  = f"[{exp_id:02d}/17] {exp_name}"
        print("\n" + "=" * 65)
        print(f"  {banner}")
        print(f"  mode={mode}  audio={audio_method}  visual={visual_method}")
        print("=" * 65)

        # ── Skip if done ──────────────────────────────────────────────────────
        if args.skip_done and run_dir.exists() and any((run_dir / "metrics").glob("test_*.json")):
            print(f"  [skip] results found in {run_dir}")
            metrics_files = sorted((run_dir / "metrics").glob("test_*.json"))
            metrics = json.loads(metrics_files[-1].read_text())
            all_results.append({"name": exp_name, **{k: v for k, v in metrics.items() if k != "per_category"},
                                 "per_category": metrics.get("per_category", {})})
            _update_unimodal_ckpts(unimodal_ckpts, mode, audio_method, visual_method, run_dir)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("metrics", "plots", "checkpoints"):
            (run_dir / sub).mkdir(exist_ok=True)

        try:
            metrics = run_single(cfg_base, mode, audio_method, visual_method,
                                 run_dir, unimodal_ckpts, device)
            _update_unimodal_ckpts(unimodal_ckpts, mode, audio_method, visual_method, run_dir)

            result = {
                "name":         exp_name,
                "mode":         mode,
                "audio_method": audio_method,
                "visual_method":visual_method,
                **{k: v for k, v in metrics.items() if k != "per_category"},
                "per_category": metrics.get("per_category", {}),
            }
            all_results.append(result)
            print(f"\n  ✅ {exp_name}  AUC={metrics.get('auc', 0):.4f}  "
                  f"EER={metrics.get('eer', 0):.4f}  F1={metrics.get('f1', 0):.4f}")

        except Exception as e:
            print(f"\n  ❌ {exp_name} FAILED: {e}")
            traceback.print_exc()
            failed.append(exp_name)

    # ── Comparison ────────────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 65)
        print("  GENERATING COMPARISON REPORT")
        print("=" * 65)

        plot_comparison(all_results, comparison_dir)
        plot_per_category_heatmap(all_results, comparison_dir, metric="auc")
        plot_per_category_heatmap(all_results, comparison_dir, metric="accuracy")

        # all_results.json
        with open(comparison_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # all_results.csv
        import csv
        cols = ["name", "mode", "audio_method", "visual_method",
                "accuracy", "auc", "eer", "f1", "precision", "recall"]
        with open(comparison_dir / "all_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n  Comparison saved → {comparison_dir}")
        print(f"    all_results.csv / all_results.json")
        print(f"    comparison_auc.png / comparison_eer.png / comparison_f1.png")
        print(f"    per_category_heatmap_auc.png / per_category_heatmap_accuracy.png")
        print(f"    comparison_all.txt  (LLM-readable ranked table)")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  DONE  {len(all_results)}/{len(EXPERIMENTS)} experiments completed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 65 + "\n")


def _update_unimodal_ckpts(ckpts: dict, mode: str, audio_method: str,
                            visual_method: str, run_dir: Path):
    if mode == "audio_only":
        ckpt = run_dir / "checkpoints" / "best_audio.pt"
        if ckpt.exists():
            ckpts[("audio", audio_method)] = ckpt
    elif mode == "video_only":
        ckpt = run_dir / "checkpoints" / "best_video.pt"
        if ckpt.exists():
            ckpts[("video", visual_method)] = ckpt


if __name__ == "__main__":
    main()
