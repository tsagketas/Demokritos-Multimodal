import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from data.preprocessing import extract_audio
from features.audio import handcrafted, wav2vec2, hubert
from utils.results import save_stats


_EXTRACTORS = {
    "handcrafted": handcrafted,
    "wav2vec2":    wav2vec2,
    "hubert":      hubert,
}


def run(cfg: dict, dataset_splits: dict, run_dir: Path):
    """
    dataset_splits = {"train": FakeAVCelebDataset, "val": ..., "test": ...}
    Saves .npy per file, manifest CSV per split per method.
    """
    audio_cfg  = cfg["features"]["audio"]
    sr         = cfg["data"]["audio"]["sample_rate"]
    device     = cfg["train"]["training"]["device"]
    base_cache = Path(cfg["data"]["audio"]["extract_dir"])

    for method, extractor in _EXTRACTORS.items():
        if not audio_cfg[method]["enabled"]:
            continue

        method_dir = base_cache / method
        method_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[audio extractor] method={method}")

        for split_name, dataset in dataset_splits.items():
            manifest_path = method_dir / f"{split_name}_manifest.csv"
            if manifest_path.exists():
                print(f"  [{split_name}] already cached, skipping.")
                continue

            rows = []
            errors = 0

            for i, (mp4_path, label) in enumerate(dataset.samples):
                feat_path = method_dir / f"{Path(mp4_path).stem}.npy"

                if not feat_path.exists():
                    try:
                        waveform = extract_audio(mp4_path, sample_rate=sr)
                        vec = extractor.extract(waveform, sr, audio_cfg[method], device)
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

        _save_extraction_stats(method, base_cache, run_dir)


def _save_extraction_stats(method: str, base_cache: Path, run_dir: Path):
    stats = {"method": method, "splits": {}}
    for manifest in (base_cache / method).glob("*_manifest.csv"):
        split = manifest.stem.replace("_manifest", "")
        with open(manifest) as f:
            count = sum(1 for _ in f) - 1
        stats["splits"][split] = count
    save_stats(stats, run_dir, tag=f"feature_extraction_{method}")
