from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.results import save_stats


REAL_TYPES = {"RealVideo-RealAudio"}


class FakeAVCelebDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]]):
        self.samples = samples  # [(mp4_path, label), ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_splits(cfg: dict, run_dir: Path = None) -> tuple[FakeAVCelebDataset, FakeAVCelebDataset, FakeAVCelebDataset]:
    root     = Path(cfg["data"]["dataset"]["root"])
    meta_csv = Path(cfg["data"]["dataset"]["meta_csv"])
    split    = cfg["data"]["split"]

    df = pd.read_csv(meta_csv)
    df.columns = df.columns.str.strip()

    # last column has the directory path (e.g. FakeAVCeleb/RealVideo-RealAudio/African/men/id00076)
    dir_col  = df.columns[-1]
    file_col = "path"

    df["mp4_path"] = df.apply(
        lambda r: _build_path(root, str(r[dir_col]).strip(), str(r[file_col]).strip()),
        axis=1,
    )
    df["label"] = df["type"].apply(lambda t: 0 if t.strip() in REAL_TYPES else 1)

    category_counts = df["type"].value_counts().to_dict()

    df = df[["mp4_path", "label"]].dropna()

    train_df, tmp_df = train_test_split(
        df,
        test_size=split["val"] + split["test"],
        stratify=df["label"],
        random_state=split["seed"],
    )
    relative_test = split["test"] / (split["val"] + split["test"])
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=relative_test,
        stratify=tmp_df["label"],
        random_state=split["seed"],
    )

    stats = {
        "splits": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
            "total": len(df),
        },
        "label_distribution": {
            "train": {f"label_{k}": int(v) for k, v in train_df["label"].value_counts().items()},
            "val":   {f"label_{k}": int(v) for k, v in val_df["label"].value_counts().items()},
            "test":  {f"label_{k}": int(v) for k, v in test_df["label"].value_counts().items()},
        },
        "class_imbalance_ratio": round(
            train_df["label"].value_counts().get(1, 0) / max(train_df["label"].value_counts().get(0, 1), 1), 2
        ),
        "category_counts": category_counts,
    }

    print(f"[dataset] train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    print(f"[dataset] class imbalance ratio (fake/real): {stats['class_imbalance_ratio']}:1")

    if run_dir is not None:
        save_stats(stats, run_dir, tag="dataset")

    return (
        FakeAVCelebDataset(list(train_df.itertuples(index=False, name=None))),
        FakeAVCelebDataset(list(val_df.itertuples(index=False, name=None))),
        FakeAVCelebDataset(list(test_df.itertuples(index=False, name=None))),
    )


def _build_path(root: Path, dir_path: str, filename: str) -> str:
    # strip leading "FakeAVCeleb/" prefix if present
    if dir_path.startswith("FakeAVCeleb/"):
        dir_path = dir_path[len("FakeAVCeleb/"):]
    return str(root / dir_path / filename)
