"""
Training script — supports four modes:
  early_fusion  concat(audio, pca_visual) → MLP
  audio_only    audio → MLP
  video_only    visual → MLP  (raw features, no PCA)
  late_fusion   load trained unimodal models, combine posteriors

Usage:
  python src/train.py --mode early_fusion --config configs/experiment.yaml
  python src/train.py --mode audio_only   --config configs/experiment.yaml
  python src/train.py --mode video_only   --config configs/experiment.yaml
  python src/train.py --mode late_fusion  --config configs/experiment.yaml \
      --audio_ckpt /path/to/best_audio.pt \
      --video_ckpt /path/to/best_video.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.feature_dataset import FeatureDataset, CATEGORIES
from models.pca_reducer import VisualPCAReducer, get_n_components
from models.early_fusion import EarlyFusionMLP
from models.unimodal_classifier import UnimodalClassifier
from models.late_fusion import LateFusionCombiner
from utils.config import load_config, setup_run_dir
from utils.results import save_metrics


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collate_fn(batch):
    """Keep category strings as list — PyTorch can't auto-collate strings."""
    audio_vecs, visual_vecs, labels, categories = zip(*batch)
    return (
        torch.stack(audio_vecs),
        torch.stack(visual_vecs),
        torch.stack(labels),
        list(categories),
    )


def collect_visual(dataset: FeatureDataset, batch_size: int = 256) -> np.ndarray:
    """Load all visual vectors in batches — used to fit PCA on train split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    parts = [visual.numpy() for _, visual, _, _ in loader]
    return np.concatenate(parts, axis=0)


# ── Metric utilities ──────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "auc":       float(roc_auc_score(labels, scores)),
        "eer":       float(compute_eer(labels, scores)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
    }


def compute_per_category(labels: np.ndarray, scores: np.ndarray, categories: list) -> dict:
    per_cat = {}
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        cat_labels = labels[idx]
        cat_scores = scores[idx]
        if len(np.unique(cat_labels)) < 2:
            preds = (cat_scores >= 0.5).astype(int)
            per_cat[cat] = {"accuracy": float(accuracy_score(cat_labels, preds))}
        else:
            per_cat[cat] = compute_metrics(cat_labels, cat_scores)
    return per_cat


# ── Generic evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(score_fn, loader: DataLoader,
              reducer: VisualPCAReducer, device: torch.device):
    """
    Generic evaluation pass.

    score_fn(audio: Tensor, pca_visual: Tensor) -> Tensor of scores in [0, 1]
    PCA is applied inside (passthrough if reducer.needed is False).
    """
    all_labels, all_scores, all_cats = [], [], []

    for audio, visual, labels, cats in loader:
        pca_visual = torch.from_numpy(reducer.transform(visual.numpy())).float().to(device)
        audio      = audio.to(device)
        scores     = score_fn(audio, pca_visual).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_scores.extend(scores)
        all_cats.extend(cats)

    return np.array(all_labels), np.array(all_scores), all_cats


# ── Generic training loop ─────────────────────────────────────────────────────

def _training_loop(
    model: nn.Module,
    logit_fn,                  # (audio_t, pca_visual_t) -> logit Tensor  [for training]
    score_fn,                  # (audio_t, pca_visual_t) -> score Tensor  [for eval]
    train_loader: DataLoader,
    val_loader: DataLoader,
    reducer: VisualPCAReducer,
    device: torch.device,
    train_cfg: dict,
    clf_cfg: dict,
    run_dir: Path,
    ckpt_name: str,
    extra_ckpt: dict = None,
) -> tuple:
    """
    Trains model.parameters() only.
    Returns (ckpt_path, best_val_auc).
    """
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    best_val_auc   = -1.0
    patience_count = 0
    patience       = train_cfg["early_stopping_patience"]
    ckpt_path      = run_dir / "checkpoints" / ckpt_name
    n_train        = len(train_loader.dataset)

    print(f"\n[train] starting — epochs={train_cfg['epochs']}, patience={patience}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        # ── train step ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for audio, visual, labels, _ in train_loader:
            pca_visual = torch.from_numpy(reducer.transform(visual.numpy())).float().to(device)
            audio      = audio.to(device)
            labels     = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(logit_fn(audio, pca_visual), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)

        scheduler.step()
        epoch_loss /= n_train

        # ── val step ──────────────────────────────────────────────────────────
        model.eval()
        val_labels, val_scores, _ = _evaluate(score_fn, val_loader, reducer, device)
        val_auc = float(roc_auc_score(val_labels, val_scores))

        improved = "↑" if val_auc > best_val_auc else " "
        print(f"  epoch {epoch:03d}/{train_cfg['epochs']}  "
              f"loss={epoch_loss:.4f}  val_auc={val_auc:.4f} {improved}")

        # ── checkpoint + early stopping ───────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            patience_count = 0
            ckpt_data = {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
            }
            if extra_ckpt:
                ckpt_data.update(extra_ckpt)
            torch.save(ckpt_data, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[train] early stopping at epoch {epoch}")
                break

    print(f"\n[train] best val AUC={best_val_auc:.4f}  checkpoint → {ckpt_path}")
    return ckpt_path, best_val_auc


# ── Output helper ─────────────────────────────────────────────────────────────

def _print_and_save(metrics: dict, run_dir: Path, tag: str):
    print("── TEST RESULTS ────────────────────────────────────────────────────")
    for k, v in metrics.items():
        if k != "per_category":
            print(f"  {k:<12} {v:.4f}")
    print("\n── PER CATEGORY ────────────────────────────────────────────────────")
    for cat, cat_m in metrics.get("per_category", {}).items():
        print(f"  [{cat}]")
        for k, v in cat_m.items():
            print(f"    {k:<12} {v:.4f}")
    save_metrics(metrics, run_dir, tag=tag)


# ── Mode runners ──────────────────────────────────────────────────────────────

def run_early_fusion(cfg, run_dir, train_ds, val_ds, test_ds,
                     train_loader, val_loader, test_loader,
                     reducer, device, audio_method, visual_method):
    train_cfg = cfg["train"]["training"]
    clf_cfg   = cfg["train"]["classifier"]

    audio_dim  = train_ds[0][0].shape[0]
    visual_dim = reducer.n_components if reducer.needed else train_ds[0][1].shape[0]

    model = EarlyFusionMLP(
        audio_dim=audio_dim, visual_dim=visual_dim,
        dropout=clf_cfg["dropout"],
    ).to(device)
    print(f"[train] EarlyFusionMLP  input_dim={audio_dim + visual_dim} "
          f"({audio_dim} audio + {visual_dim} visual)")

    logit_fn = lambda a, v: model(a, v)
    score_fn = lambda a, v: torch.sigmoid(model(a, v))

    ckpt_path, _ = _training_loop(
        model, logit_fn, score_fn,
        train_loader, val_loader, reducer, device,
        train_cfg, clf_cfg, run_dir,
        ckpt_name="best_model.pt",
        extra_ckpt={"audio_dim": audio_dim, "visual_dim": visual_dim, "mode": "early_fusion"},
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded best model from epoch {ckpt['epoch']}\n")

    test_labels, test_scores, test_cats = _evaluate(score_fn, test_loader, reducer, device)
    metrics = compute_metrics(test_labels, test_scores)
    metrics["per_category"] = compute_per_category(test_labels, test_scores, test_cats)
    _print_and_save(metrics, run_dir, tag="test_early_fusion")


def run_unimodal(cfg, run_dir, train_ds, val_ds, test_ds,
                 train_loader, val_loader, test_loader,
                 reducer, device, modality: str):
    """
    modality: "audio" or "video"
    Reducer is always passthrough for unimodal modes — PCA only
    makes sense when balancing two modalities for fusion.
    """
    train_cfg = cfg["train"]["training"]
    clf_cfg   = cfg["train"]["classifier"]

    if modality == "audio":
        input_dim = train_ds[0][0].shape[0]    # e.g. 768 for wav2vec2
    else:
        input_dim = train_ds[0][1].shape[0]    # e.g. 2048 for xception, 731 for landmarks

    model = UnimodalClassifier(input_dim=input_dim, dropout=clf_cfg["dropout"]).to(device)
    print(f"[train] UnimodalClassifier({modality})  input_dim={input_dim}")

    if modality == "audio":
        logit_fn = lambda a, v: model(a)
        score_fn = lambda a, v: torch.sigmoid(model(a))
    else:
        logit_fn = lambda a, v: model(v)
        score_fn = lambda a, v: torch.sigmoid(model(v))

    ckpt_path, _ = _training_loop(
        model, logit_fn, score_fn,
        train_loader, val_loader, reducer, device,
        train_cfg, clf_cfg, run_dir,
        ckpt_name=f"best_{modality}.pt",
        extra_ckpt={"input_dim": input_dim, "modality": modality},
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded best {modality} model from epoch {ckpt['epoch']}\n")

    test_labels, test_scores, test_cats = _evaluate(score_fn, test_loader, reducer, device)
    metrics = compute_metrics(test_labels, test_scores)
    metrics["per_category"] = compute_per_category(test_labels, test_scores, test_cats)
    _print_and_save(metrics, run_dir, tag=f"test_{modality}_only")


def run_late_fusion(cfg, run_dir, train_ds, val_ds, test_ds,
                    train_loader, val_loader, test_loader,
                    reducer, device, audio_ckpt_path: str, video_ckpt_path: str):
    train_cfg = cfg["train"]["training"]
    clf_cfg   = cfg["train"]["classifier"]
    strategy  = cfg["fusion"]["fusion"]["late"].get("strategy", "average")
    weight    = cfg["fusion"]["fusion"]["late"].get("weight", 0.5)

    # ── Load and freeze trained unimodal models ───────────────────────────────
    audio_ckpt  = torch.load(audio_ckpt_path, map_location=device)
    video_ckpt  = torch.load(video_ckpt_path, map_location=device)

    audio_model = UnimodalClassifier(input_dim=audio_ckpt["input_dim"]).to(device)
    audio_model.load_state_dict(audio_ckpt["model_state"])
    audio_model.requires_grad_(False).eval()

    video_model = UnimodalClassifier(input_dim=video_ckpt["input_dim"]).to(device)
    video_model.load_state_dict(video_ckpt["model_state"])
    video_model.requires_grad_(False).eval()

    print(f"[late_fusion] strategy={strategy}  "
          f"audio_dim={audio_ckpt['input_dim']}  video_dim={video_ckpt['input_dim']}")

    if strategy == "average":
        # ── No training: fixed weighted average ───────────────────────────────
        print(f"[late_fusion] weights: audio={weight:.2f}, video={1 - weight:.2f}")
        score_fn = lambda a, v: (
            weight * torch.sigmoid(audio_model(a))
            + (1.0 - weight) * torch.sigmoid(video_model(v))
        )
        test_labels, test_scores, test_cats = _evaluate(
            score_fn, test_loader, reducer, device
        )

    else:
        # ── Learned: train a Linear(2→1) combiner ────────────────────────────
        combiner = LateFusionCombiner(strategy="learned").to(device)

        logit_fn = lambda a, v: combiner(
            torch.sigmoid(audio_model(a)),
            torch.sigmoid(video_model(v)),
        )
        score_fn = lambda a, v: torch.sigmoid(logit_fn(a, v))

        print("[late_fusion] training Linear(2→1) combiner...")
        ckpt_path, _ = _training_loop(
            combiner, logit_fn, score_fn,
            train_loader, val_loader, reducer, device,
            train_cfg, clf_cfg, run_dir,
            ckpt_name="best_combiner.pt",
            extra_ckpt={"strategy": strategy},
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        combiner.load_state_dict(ckpt["model_state"])
        combiner.eval()

        test_labels, test_scores, test_cats = _evaluate(
            score_fn, test_loader, reducer, device
        )

    metrics = compute_metrics(test_labels, test_scores)
    metrics["per_category"] = compute_per_category(test_labels, test_scores, test_cats)
    _print_and_save(metrics, run_dir, tag=f"test_late_fusion_{strategy}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--mode",   default="early_fusion",
                        choices=["early_fusion", "audio_only", "video_only", "late_fusion"])
    parser.add_argument("--audio_ckpt", default=None,
                        help="Trained audio checkpoint (required for late_fusion)")
    parser.add_argument("--video_ckpt", default=None,
                        help="Trained video checkpoint (required for late_fusion)")
    args = parser.parse_args()

    if args.mode == "late_fusion" and not (args.audio_ckpt and args.video_ckpt):
        parser.error("--audio_ckpt and --video_ckpt are required for late_fusion mode")

    cfg     = load_config(args.config)
    run_dir = setup_run_dir(cfg)

    train_cfg     = cfg["train"]["training"]
    audio_method  = cfg["fusion"]["multimodal"]["audio_feature"]
    visual_method = cfg["fusion"]["multimodal"]["visual_feature"]

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device = torch.device(
        train_cfg["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[{args.mode}] audio={audio_method}  visual={visual_method}  device={device}")

    # ── Manifest paths ────────────────────────────────────────────────────────
    audio_dir  = Path(cfg["data"]["audio"]["extract_dir"]) / audio_method
    visual_dir = Path(cfg["features"]["visual"]["cache_dir"])

    # ── Datasets ──────────────────────────────────────────────────────────────
    def make_ds(split):
        return FeatureDataset(
            str(audio_dir  / f"{split}_manifest.csv"),
            str(visual_dir / f"{split}_manifest.csv"),
        )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")
    print(f"[{args.mode}] samples — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ── PCA reducer ───────────────────────────────────────────────────────────
    # PCA is only needed for early_fusion to balance modality dimensions.
    # Unimodal modes use raw features; late_fusion uses the same raw features
    # that the unimodal models were trained on.
    if args.mode == "early_fusion":
        n_components = get_n_components(audio_method, visual_method)
        reducer      = VisualPCAReducer(n_components=n_components)
        if reducer.needed:
            print(f"[{args.mode}] fitting PCA {visual_method} → {n_components}d (train only)...")
            reducer.fit(collect_visual(train_ds))
            reducer.save(run_dir / "pca.pkl")
            print(f"[{args.mode}] PCA saved → {run_dir / 'pca.pkl'}")
        else:
            print(f"[{args.mode}] no PCA needed for {audio_method} + {visual_method}")
    else:
        reducer = VisualPCAReducer(n_components=None)   # passthrough

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── Dispatch to mode runner ───────────────────────────────────────────────
    common = dict(
        cfg=cfg, run_dir=run_dir,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        reducer=reducer, device=device,
    )

    if args.mode == "early_fusion":
        run_early_fusion(**common, audio_method=audio_method, visual_method=visual_method)
    elif args.mode == "audio_only":
        run_unimodal(**common, modality="audio")
    elif args.mode == "video_only":
        run_unimodal(**common, modality="video")
    elif args.mode == "late_fusion":
        run_late_fusion(**common,
                        audio_ckpt_path=args.audio_ckpt,
                        video_ckpt_path=args.video_ckpt)

    print(f"\n[done] run_dir → {run_dir}")


if __name__ == "__main__":
    main()
