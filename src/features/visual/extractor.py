import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from data.preprocessing import extract_frames
from features.visual import landmark_features, xception_features
from utils.results import save_stats

_EXTRACTORS = {
    "landmarks": landmark_features,
    "xception":  xception_features,
}


def run(cfg: dict, dataset_splits: dict, run_dir: Path):
    visual_cfg = cfg["features"]["visual"]
    method     = visual_cfg.get("method", "landmarks")

    if not visual_cfg.get("enabled", True):
        return

    if method not in _EXTRACTORS:
        raise ValueError(f"Unknown visual method '{method}'. Choose from: {list(_EXTRACTORS)}")

    extractor  = _EXTRACTORS[method]
    max_frames = cfg["data"]["video"]["max_frames"]
    device     = cfg["train"]["training"]["device"]
    cache_dir  = Path(visual_cfg["cache_dir"]) / method
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[visual extractor] method={method}  dims={extractor.FEATURE_DIM}")

    for split_name, dataset in dataset_splits.items():
        manifest_path = cache_dir / f"{split_name}_manifest.csv"
        if manifest_path.exists():
            print(f"  [{split_name}] already cached, skipping.")
            continue

        rows, errors = [], 0

        for i, (mp4_path, label) in enumerate(dataset.samples):
            feat_path = cache_dir / f"{Path(mp4_path).stem}_{method}.npy"

            if not feat_path.exists():
                try:
                    frames = extract_frames(mp4_path, max_frames=max_frames)
                    vec    = extractor.extract(frames, visual_cfg, device)
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
        "method":      method,
        "vector_dims": extractor.FEATURE_DIM,
        "splits": {
            m.stem.replace("_manifest", ""): sum(1 for _ in open(m)) - 1
            for m in cache_dir.glob("*_manifest.csv")
        }
    }, run_dir, tag="feature_extraction_visual")
