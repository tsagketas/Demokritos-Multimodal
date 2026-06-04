"""
Early Fusion training script.
Usage: python src/train.py --config configs/experiment.yaml
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
from utils.config import load_config, setup_run_dir
from utils.results import save_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Load all visual vectors in batches for PCA fitting."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    parts = []
    for _, visual, _, _ in loader:
        parts.append(visual.numpy())
    return np.concatenate(parts, axis=0)


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             reducer: VisualPCAReducer, device: torch.device):
    """Run inference, return (labels, scores, categories)."""
    model.eval()
    all_labels, all_scores, all_cats = [], [], []

    for audio, visual, labels, cats in loader:
        visual_np = reducer.transform(visual.numpy())
        visual    = torch.from_numpy(visual_np).float().to(device)
        audio     = audio.to(device)

        logits = model(audio, visual)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_scores.extend(scores)
        all_cats.extend(cats)

    return np.array(all_labels), np.array(all_scores), all_cats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    run_dir = setup_run_dir(cfg)

    train_cfg = cfg["train"]["training"]
    clf_cfg   = cfg["train"]["classifier"]

    # reproducibility
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    device = torch.device(
        train_cfg["device"] if torch.cuda.is_available() else "cpu"
    )

    audio_method  = cfg["fusion"]["multimodal"]["audio_feature"]
    visual_method = cfg["fusion"]["multimodal"]["visual_feature"]
    print(f"\n[train] audio={audio_method}  visual={visual_method}  device={device}")

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
    print(f"[train] samples — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ── PCA (fit on train only) ────────────────────────────────────────────────
    n_components = get_n_components(audio_method, visual_method)
    reducer      = VisualPCAReducer(n_components=n_components)

    if reducer.needed:
        print(f"[train] fitting PCA {visual_method} → {n_components}d (train only)...")
        train_visual = collect_visual(train_ds)
        reducer.fit(train_visual)
        reducer.save(run_dir / "pca.pkl")
        print(f"[train] PCA fitted → {run_dir / 'pca.pkl'}")
    else:
        print(f"[train] no PCA needed for {audio_method} + {visual_method}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    audio_dim  = train_ds[0][0].shape[0]
    visual_dim = n_components if reducer.needed else train_ds[0][1].shape[0]

    model = EarlyFusionMLP(
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        dropout=clf_cfg["dropout"],
    ).to(device)
    print(f"[train] MLP input_dim={audio_dim + visual_dim}  ({audio_dim} audio + {visual_dim} visual)")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc   = -1.0
    patience_count = 0
    patience       = train_cfg["early_stopping_patience"]
    ckpt_path      = run_dir / "checkpoints" / "best_model.pt"

    print(f"\n[train] starting — epochs={train_cfg['epochs']}, patience={patience}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0

        for audio, visual, labels, _ in train_loader:
            visual_np = reducer.transform(visual.numpy())
            visual    = torch.from_numpy(visual_np).float().to(device)
            audio     = audio.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(audio, visual), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)

        scheduler.step()
        epoch_loss /= len(train_ds)

        # --- val ---
        val_labels, val_scores, _ = evaluate(model, val_loader, reducer, device)
        val_auc = float(roc_auc_score(val_labels, val_scores))

        improved = "↑" if val_auc > best_val_auc else " "
        print(f"  epoch {epoch:03d}/{train_cfg['epochs']}  "
              f"loss={epoch_loss:.4f}  val_auc={val_auc:.4f} {improved}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "audio_dim":   audio_dim,
                "visual_dim":  visual_dim,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[train] early stopping at epoch {epoch}")
                break

    print(f"\n[train] best val AUC={best_val_auc:.4f}  checkpoint → {ckpt_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval]  loaded best model from epoch {ckpt['epoch']}\n")

    test_labels, test_scores, test_cats = evaluate(model, test_loader, reducer, device)
    metrics = compute_metrics(test_labels, test_scores)
    metrics["per_category"] = compute_per_category(test_labels, test_scores, test_cats)

    print("── TEST RESULTS ────────────────────────────────────────────────────")
    for k, v in metrics.items():
        if k != "per_category":
            print(f"  {k:<12} {v:.4f}")

    print("\n── PER CATEGORY ────────────────────────────────────────────────────")
    for cat, cat_m in metrics["per_category"].items():
        print(f"  [{cat}]")
        for k, v in cat_m.items():
            print(f"    {k:<12} {v:.4f}")

    save_metrics(metrics, run_dir, tag="test")
    print(f"\n[done] run_dir → {run_dir}")


if __name__ == "__main__":
    main()
