import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CATEGORIES = [
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]


def _parse_category(mp4_path: str) -> str:
    for cat in CATEGORIES:
        if cat in mp4_path:
            return cat
    return "unknown"


class FeatureDataset(Dataset):
    """
    Loads pre-extracted audio + visual .npy features for early fusion.

    Expects two manifest CSVs (one audio, one visual) with columns:
        feature_path, label, mp4_path

    Only samples present in BOTH manifests are kept (inner join on mp4_path).
    """

    def __init__(self, audio_manifest: str, visual_manifest: str):
        audio_df = pd.read_csv(audio_manifest)
        visual_df = pd.read_csv(visual_manifest)

        merged = audio_df.merge(visual_df, on="mp4_path", suffixes=("_audio", "_visual"))

        if merged.empty:
            raise ValueError(
                f"No common samples found between:\n  {audio_manifest}\n  {visual_manifest}"
            )

        self.samples = [
            {
                "audio_path": row.feature_path_audio,
                "visual_path": row.feature_path_visual,
                "label": int(row.label_audio),
                "category": _parse_category(row.mp4_path),
            }
            for _, row in merged.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        audio_vec = torch.from_numpy(np.load(s["audio_path"])).float()
        visual_vec = torch.from_numpy(np.load(s["visual_path"])).float()
        label = torch.tensor(s["label"], dtype=torch.float32)
        return audio_vec, visual_vec, label, s["category"]
