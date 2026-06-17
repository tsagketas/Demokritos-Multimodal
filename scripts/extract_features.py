"""
Extract all 5 feature sets from the FakeAVCeleb dataset.
Run ONCE before experiments — results are cached, re-running is instant.

Usage:
    python extract_features.py
    python extract_features.py --config configs/experiment.yaml
"""
import sys
import copy
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import load_splits
from features.audio.extractor import run as run_audio
from features.visual.extractor import run as run_visual
from utils.config import load_config, setup_run_dir


def _banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract all 5 feature sets")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    run_dir = setup_run_dir(cfg)

    # ── Dataset splits ────────────────────────────────────────────────────────
    _banner("LOADING DATASET SPLITS")
    train_ds, val_ds, test_ds = load_splits(cfg, run_dir)
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    print(f"\n  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ── Audio (3 methods) ─────────────────────────────────────────────────────
    _banner("AUDIO FEATURE EXTRACTION  [1/3 handcrafted | 2/3 wav2vec2 | 3/3 hubert]")
    audio_cfg = cfg["features"]["audio"]
    for method in ["handcrafted", "wav2vec2", "hubert"]:
        audio_cfg[method]["enabled"] = True
    run_audio(cfg, splits, run_dir)

    # ── Visual: landmarks ─────────────────────────────────────────────────────
    _banner("VISUAL FEATURE EXTRACTION  [4/5 landmarks]")
    cfg_lm = copy.deepcopy(cfg)
    cfg_lm["features"]["visual"]["method"] = "landmarks"
    run_visual(cfg_lm, splits, run_dir)

    # ── Visual: xception ──────────────────────────────────────────────────────
    _banner("VISUAL FEATURE EXTRACTION  [5/5 xception]")
    cfg_xc = copy.deepcopy(cfg)
    cfg_xc["features"]["visual"]["method"] = "xception"
    run_visual(cfg_xc, splits, run_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("ALL EXTRACTIONS COMPLETE")
    audio_base  = Path(cfg["data"]["audio"]["extract_dir"])
    visual_base = Path(cfg["features"]["visual"]["cache_dir"])
    print(f"\n  Audio features  → {audio_base}")
    for method in ["handcrafted", "wav2vec2", "hubert"]:
        method_dir = audio_base / method
        n = sum(1 for f in method_dir.glob("*.npy")) if method_dir.exists() else 0
        print(f"    [{method}]  {n} .npy files")
    print(f"\n  Visual features → {visual_base}")
    for method in ["landmarks", "xception"]:
        method_dir = visual_base / method
        n = sum(1 for f in method_dir.glob("*.npy")) if method_dir.exists() else 0
        print(f"    [{method}]  {n} .npy files")
    print()


if __name__ == "__main__":
    main()
