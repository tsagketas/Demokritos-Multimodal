"""
src/data/inspect_dataset.py

Αναλύει την κατανομή του dataset. Δύο modes:

  --mode raw        Διαβάζει απευθείας το meta_data.csv — δεν χρειάζεται
                    feature extraction. Εφαρμόζει το ίδιο stratified split
                    που κάνει και το training pipeline.

  --mode manifests  Διαβάζει τα extracted manifest CSVs (μετά την feature
                    extraction) — δείχνει το effective dataset μετά το
                    inner join audio+visual (μπορεί να λείπουν samples
                    λόγω extraction failures).

Παράγει:
  - terminal summary table
  - imbalance_<tag>.png  : real vs fake count per split
  - categories_<tag>.png : 4 κατηγορίες per split

Usage:
  # πριν την feature extraction:
  python src/data/inspect_dataset.py --config configs/experiment.yaml --mode raw

  # μετά την feature extraction:
  python src/data/inspect_dataset.py --config configs/experiment.yaml --mode manifests
  python src/data/inspect_dataset.py --config configs/experiment.yaml --mode manifests \\
      --audio_method hubert --visual_method landmarks
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config

CATEGORIES = [
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]
SPLITS = ["train", "val", "test"]
CAT_COLORS = {
    "RealVideo-RealAudio": "#10b981",
    "RealVideo-FakeAudio": "#f59e0b",
    "FakeVideo-RealAudio": "#f97316",
    "FakeVideo-FakeAudio": "#ef4444",
}
REAL_TYPES = {"RealVideo-RealAudio"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _parse_category(mp4_path: str) -> str:
    for cat in CATEGORIES:
        if cat in mp4_path:
            return cat
    return "unknown"


def load_from_meta(cfg: dict) -> dict[str, pd.DataFrame]:
    """Reads meta_data.csv and applies the same stratified split as dataset.py."""
    meta_csv = Path(cfg["data"]["dataset"]["meta_csv"])
    root     = Path(cfg["data"]["dataset"]["root"])
    split    = cfg["data"]["split"]

    df = pd.read_csv(meta_csv)
    df.columns = df.columns.str.strip()

    dir_col = df.columns[-1]
    df["dir"] = df[dir_col].astype(str).str.strip()
    df["dir"] = df["dir"].apply(
        lambda d: d[len("FakeAVCeleb/"):] if d.startswith("FakeAVCeleb/") else d
    )
    df["mp4_path"] = df.apply(
        lambda r: str(root / r["dir"] / str(r["path"]).strip()), axis=1
    )
    df["label"]    = df["type"].apply(lambda t: 0 if t.strip() in REAL_TYPES else 1)
    df["category"] = df["type"].str.strip()
    df = df[["mp4_path", "label", "category"]].dropna()

    train_df, tmp_df = train_test_split(
        df, test_size=split["val"] + split["test"],
        stratify=df["label"], random_state=split["seed"],
    )
    rel_test = split["test"] / (split["val"] + split["test"])
    val_df, test_df = train_test_split(
        tmp_df, test_size=rel_test,
        stratify=tmp_df["label"], random_state=split["seed"],
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def load_from_manifests(cfg: dict, audio_method: str, visual_method: str) -> dict[str, pd.DataFrame]:
    """Inner-joins audio+visual manifests exactly like FeatureDataset does."""
    audio_dir  = Path(cfg["data"]["audio"]["extract_dir"]) / audio_method
    visual_dir = Path(cfg["features"]["visual"]["cache_dir"])

    splits = {}
    for split in SPLITS:
        a_man = audio_dir  / f"{split}_manifest.csv"
        v_man = visual_dir / f"{split}_manifest.csv"
        if not a_man.exists():
            raise FileNotFoundError(f"Audio manifest not found: {a_man}")
        if not v_man.exists():
            raise FileNotFoundError(f"Visual manifest not found: {v_man}")

        a_df = pd.read_csv(a_man)
        v_df = pd.read_csv(v_man)
        merged = a_df.merge(v_df, on="mp4_path", suffixes=("_audio", "_visual"))
        merged["label"]    = merged["label_audio"].astype(int)
        merged["category"] = merged["mp4_path"].apply(_parse_category)
        splits[split] = merged
    return splits


# ── Summarize & print ─────────────────────────────────────────────────────────

def _summarize(df: pd.DataFrame, split: str) -> dict:
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    ratio  = round(n_fake / n_real, 3) if n_real > 0 else float("inf")
    row = {
        "split": split, "total": len(df),
        "real": n_real, "fake": n_fake,
        "fake_to_real_ratio": ratio,
    }
    for cat in CATEGORIES:
        row[cat] = int((df["category"] == cat).sum())
    return row


def _print_table(rows: list, mode: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  MODE: {mode.upper()}")
    print(f"  {'SPLIT':<8} {'TOTAL':>7} {'REAL':>7} {'FAKE':>7} {'FAKE/REAL':>10}")
    print(f"{'─' * 70}")
    for r in rows:
        flag = "  ⚠️  imbalance" if abs(r["fake_to_real_ratio"] - 1.0) > 0.2 else ""
        print(f"  {r['split']:<8} {r['total']:>7} {r['real']:>7} {r['fake']:>7}"
              f" {r['fake_to_real_ratio']:>10.3f}{flag}")
    print(f"{'─' * 70}")
    print("\n  CATEGORY BREAKDOWN")
    print(f"{'─' * 70}")
    for r in rows:
        print(f"  [{r['split']}]  total={r['total']}")
        for cat in CATEGORIES:
            pct = 100 * r[cat] / r["total"] if r["total"] > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {cat:<28} {r[cat]:>5}  ({pct:5.1f}%)  {bar}")
    print(f"{'=' * 70}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_real_vs_fake(rows: list, out_path: Path, title_suffix: str = "") -> None:
    splits = [r["split"] for r in rows]
    reals  = [r["real"]  for r in rows]
    fakes  = [r["fake"]  for r in rows]
    x = np.arange(len(splits))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b_real = ax.bar(x - w / 2, reals, w, label="Real (0)", color="#10b981", alpha=0.85)
    b_fake = ax.bar(x + w / 2, fakes, w, label="Fake (1)", color="#ef4444", alpha=0.85)

    peak = max(reals + fakes)
    for bar in list(b_real) + list(b_fake):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + peak * 0.012,
                str(int(h)), ha="center", va="bottom", fontsize=9)

    for i, r in enumerate(rows):
        ax.annotate(f"ratio {r['fake_to_real_ratio']:.2f}",
                    xy=(i, peak * 1.10), ha="center", fontsize=8, color="#64748b")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=11)
    ax.set_ylabel("Samples")
    ax.set_title(f"Class Imbalance — Real vs Fake{title_suffix}", fontweight="bold")
    ax.set_ylim(0, peak * 1.22)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_categories(rows: list, out_path: Path, title_suffix: str = "") -> None:
    splits  = [r["split"] for r in rows]
    x       = np.arange(len(splits))
    w       = 0.55
    bottoms = np.zeros(len(splits))

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in CATEGORIES:
        vals = np.array([r[cat] for r in rows], dtype=float)
        ax.bar(x, vals, w, bottom=bottoms, label=cat, color=CAT_COLORS[cat], alpha=0.88)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 15:
                ax.text(i, b + v / 2, str(int(v)),
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=11)
    ax.set_ylabel("Samples")
    ax.set_title(f"Category Distribution per Split{title_suffix}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/experiment.yaml")
    parser.add_argument("--mode",          default="raw",
                        choices=["raw", "manifests"],
                        help="raw=meta_data.csv (pre-extraction), manifests=post-extraction")
    parser.add_argument("--audio_method",  default=None,
                        help="Only for --mode manifests: handcrafted|wav2vec2|hubert")
    parser.add_argument("--visual_method", default=None,
                        help="Only for --mode manifests: xception|landmarks")
    parser.add_argument("--out_dir",       default="graphs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "raw":
        print("\n[inspect] mode=raw  → reading meta_data.csv")
        split_dfs = load_from_meta(cfg)
        tag = "raw"
        title_suffix = " (pre-extraction)"
    else:
        audio_method  = args.audio_method  or cfg["fusion"]["multimodal"]["audio_feature"]
        visual_method = args.visual_method or cfg["fusion"]["multimodal"]["visual_feature"]
        print(f"\n[inspect] mode=manifests  audio={audio_method}  visual={visual_method}")
        split_dfs = load_from_manifests(cfg, audio_method, visual_method)
        tag = f"{audio_method}_{visual_method}"
        title_suffix = f" ({audio_method} + {visual_method})"

    rows = [_summarize(split_dfs[s], s) for s in SPLITS]
    _print_table(rows, args.mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[plots]")
    plot_real_vs_fake(rows, out_dir / f"imbalance_{tag}.png",   title_suffix)
    plot_categories( rows, out_dir / f"categories_{tag}.png",   title_suffix)


if __name__ == "__main__":
    main()
