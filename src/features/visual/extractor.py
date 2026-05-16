import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from data.preprocessing import extract_frames
from features.visual import lip_features
from utils.results import save_stats


def run(cfg: dict, dataset_splits: dict, run_dir: Path):
    visual_cfg = cfg["features"]["visual"]["lip"]
    if not visual_cfg["enabled"]:
        return

    max_frames = cfg["data"]["video"]["max_frames"]
    cache_dir  = Path(visual_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n[visual extractor] method=lip_landmarks")

    for split_name, dataset in dataset_splits.items():
        manifest_path = cache_dir / f"{split_name}_manifest.csv"
        if manifest_path.exists():
            print(f"  [{split_name}] already cached, skipping.")
            continue

        rows   = []
        errors = 0

        for i, (mp4_path, label) in enumerate(dataset.samples):
            feat_path = cache_dir / f"{Path(mp4_path).stem}_lip.npy"

            if not feat_path.exists():
                try:
                    frames = extract_frames(mp4_path, max_frames=max_frames)
                    vec    = lip_features.extract(frames, visual_cfg)
                    np.save(feat_path, vec)
                except Exception as e:
                    errors += 1
                    print(f"  WARN: {Path(mp4_path).name} — {e}")
                    continue

            rows.append({"feature_path": str(feat_path), "label": label, "mp4_path": mp4_path})

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] {i+1}/{len(dataset.samples)}")

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "mp4_path"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"  [{split_name}] done — {len(rows)} ok, {errors} errors → {manifest_path.name}")

    save_stats({
        "method": "lip_landmarks",
        "vector_dims": 244,
        "splits": {
            m.stem.replace("_manifest", ""): sum(1 for _ in open(m)) - 1
            for m in cache_dir.glob("*_manifest.csv")
        }
    }, run_dir, tag="feature_extraction_visual")
