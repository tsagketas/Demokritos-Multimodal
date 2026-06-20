import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix as sk_confusion_matrix


def _save_txt(path: Path, lines: list):
    with open(path, "w") as f:
        f.write("\n".join(str(l) for l in lines) + "\n")


# ── Per-experiment plots ───────────────────────────────────────────────────────

def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, plots_dir: Path, tag: str):
    plots_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(sk_auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="#3498db", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_roc_curve.png", dpi=150)
    plt.close(fig)

    # companion txt
    indices = np.linspace(0, len(fpr) - 1, min(10, len(fpr)), dtype=int)
    lines = [
        f"ROC CURVE — {tag}",
        f"AUC: {auc:.4f}",
        "",
        f"{'Threshold':<14} {'FPR':<10} {'TPR':<10}",
        "-" * 34,
    ]
    for i in indices:
        thr = f"{thresholds[i]:.4f}" if i < len(thresholds) else "N/A"
        lines.append(f"{thr:<14} {fpr[i]:<10.4f} {tpr[i]:<10.4f}")
    _save_txt(plots_dir / f"{tag}_roc_curve.txt", lines)


def plot_confusion_matrix(labels: np.ndarray, scores: np.ndarray,
                          plots_dir: Path, tag: str, threshold: float = 0.5):
    plots_dir.mkdir(parents=True, exist_ok=True)

    preds = (scores >= threshold).astype(int)
    cm = sk_confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes = ["Real (0)", "Fake (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag}")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    lines = [
        f"CONFUSION MATRIX — {tag}",
        f"Threshold: {threshold}",
        "",
        f"{'':22} {'Pred Real':>12} {'Pred Fake':>12}",
        f"{'True Real':22} {tn:>12} {fp:>12}",
        f"{'True Fake':22} {fn:>12} {tp:>12}",
        "",
        f"TP: {tp}   FP: {fp}   TN: {tn}   FN: {fn}",
        f"Accuracy:  {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
    ]
    _save_txt(plots_dir / f"{tag}_confusion_matrix.txt", lines)


def plot_per_category_bar(per_category: dict, plots_dir: Path, tag: str):
    plots_dir.mkdir(parents=True, exist_ok=True)

    cats    = list(per_category.keys())
    metrics = ["accuracy", "auc", "f1"]
    x       = np.arange(len(cats))
    width   = 0.25
    colors  = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [per_category[c].get(metric, 0.0) for c in cats]
        ax.bar(x + i * width, values, width, label=metric.upper(), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("-", "\n") for c in cats], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Category Metrics — {tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_per_category_bar.png", dpi=150)
    plt.close(fig)

    col_w = 12
    header = f"{'Category':<30}" + "".join(f"{m.upper():<{col_w}}" for m in metrics)
    lines  = [f"PER-CATEGORY METRICS — {tag}", "", header, "-" * (30 + col_w * len(metrics))]
    for cat in cats:
        row = f"{cat:<30}"
        for m in metrics:
            v = per_category[cat].get(m)
            row += f"{v:<{col_w}.4f}" if v is not None else f"{'N/A':<{col_w}}"
        lines.append(row)
    _save_txt(plots_dir / f"{tag}_per_category_bar.txt", lines)


# ── Comparison plots (across all experiments) ─────────────────────────────────

def plot_comparison(all_results: list, comparison_dir: Path):
    """
    all_results: list of dicts — keys: name, auc, eer, f1, accuracy, precision, recall
    """
    comparison_dir.mkdir(parents=True, exist_ok=True)

    def _color(name):
        if "late"  in name: return "#e74c3c"
        if "early" in name: return "#3498db"
        return "#2ecc71"

    sorted_r = sorted(all_results, key=lambda r: r.get("auc", 0), reverse=True)
    names    = [r["name"] for r in sorted_r]

    for metric in ["auc", "eer", "f1", "accuracy"]:
        values = [r.get(metric, 0.0) for r in sorted_r]
        colors = [_color(n) for n in names]

        fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.45 + 2)))
        bars = ax.barh(names, values, color=colors)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"All Experiments — {metric.upper()}")
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        # legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#2ecc71", label="unimodal"),
            Patch(color="#3498db", label="early fusion"),
            Patch(color="#e74c3c", label="late fusion"),
        ], loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(comparison_dir / f"comparison_{metric}.png", dpi=150)
        plt.close(fig)

    # ranked txt table
    cols  = ["auc", "eer", "f1", "accuracy", "precision", "recall"]
    col_w = 12
    header = f"{'Experiment':<45}" + "".join(f"{c.upper():<{col_w}}" for c in cols)
    lines  = [
        "EXPERIMENT COMPARISON (sorted by AUC desc)",
        "=" * (45 + col_w * len(cols)),
        "",
        header,
        "-" * (45 + col_w * len(cols)),
    ]
    for r in sorted_r:
        row = f"{r['name']:<45}"
        for c in cols:
            v = r.get(c)
            row += f"{v:<{col_w}.4f}" if v is not None else f"{'N/A':<{col_w}}"
        lines.append(row)
    _save_txt(comparison_dir / "comparison_all.txt", lines)


def plot_per_category_heatmap(all_results: list, comparison_dir: Path, metric: str = "auc"):
    comparison_dir.mkdir(parents=True, exist_ok=True)

    cats  = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
             "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    names = [r["name"] for r in all_results]

    matrix = np.full((len(names), len(cats)), np.nan)
    for i, r in enumerate(all_results):
        for j, cat in enumerate(cats):
            v = r.get("per_category", {}).get(cat, {}).get(metric)
            if v is not None:
                matrix[i, j] = v

    if np.all(np.isnan(matrix)):
        print(f"[plots] per_category_heatmap_{metric}: all values are NaN — skipping (single-class categories)")
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.5 + 2)))
    try:
        import seaborn as sns
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[c.replace("-", "\n") for c in cats],
                    yticklabels=names, ax=ax, vmin=0, vmax=1,
                    linewidths=0.5)
    except ImportError:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(cats)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([c.replace("-", "\n") for c in cats], fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(cats)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.3f}",
                            ha="center", va="center", fontsize=7)
    ax.set_title(f"Per-Category {metric.upper()} — All Experiments")
    fig.tight_layout()
    fig.savefig(comparison_dir / f"per_category_heatmap_{metric}.png", dpi=150)
    plt.close(fig)

    col_w = 22
    header = f"{'Experiment':<45}" + "".join(f"{c[:col_w-1]:<{col_w}}" for c in cats)
    lines  = [
        f"PER-CATEGORY {metric.upper()} — ALL EXPERIMENTS",
        "=" * (45 + col_w * len(cats)),
        "",
        header,
        "-" * (45 + col_w * len(cats)),
    ]
    for i, r in enumerate(all_results):
        row = f"{r['name']:<45}"
        for j in range(len(cats)):
            v = matrix[i, j]
            row += f"{v:<{col_w}.4f}" if not np.isnan(v) else f"{'N/A':<{col_w}}"
        lines.append(row)
    _save_txt(comparison_dir / f"per_category_heatmap_{metric}.txt", lines)
