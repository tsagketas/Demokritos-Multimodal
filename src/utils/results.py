import csv
import json
from pathlib import Path
from datetime import datetime


def save_stats(stats: dict, run_dir: Path, tag: str):
    out = run_dir / "stats"
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = out / f"{tag}_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(f"=== {tag.upper()} — {ts} ===\n\n")
        _write_dict(f, stats, indent=0)

    with open(out / f"{tag}_{ts}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[stats] saved → {txt_path}")


def _write_dict(f, d: dict, indent: int):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            f.write(f"{pad}{k}:\n")
            _write_dict(f, v, indent + 1)
        else:
            f.write(f"{pad}  {k:<25} {v}\n")


def save_metrics(metrics: dict, run_dir: Path, tag: str = "eval"):
    """
    metrics = {
        "accuracy": 0.923,
        "auc": 0.961,
        "eer": 0.078,
        "f1": 0.918,
        "per_category": {
            "RealVideo-FakeAudio": {"accuracy": 0.94, "auc": 0.97},
            "FakeVideo-RealAudio": {"accuracy": 0.89, "auc": 0.93},
        }
    }
    """
    out = run_dir / "metrics"
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- TXT (human readable) ---
    txt_path = out / f"{tag}_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(f"=== {tag.upper()} RESULTS — {ts} ===\n\n")
        for k, v in metrics.items():
            if k == "per_category":
                continue
            f.write(f"  {k:<15} {v:.4f}\n")

        if "per_category" in metrics:
            f.write("\n--- Per Category ---\n")
            for cat, cat_metrics in metrics["per_category"].items():
                f.write(f"\n  [{cat}]\n")
                for k, v in cat_metrics.items():
                    f.write(f"    {k:<15} {v:.4f}\n")

    # --- CSV (για σύγκριση runs) ---
    csv_path = out / "all_runs.csv"
    flat = {"timestamp": ts, "tag": tag}
    for k, v in metrics.items():
        if k != "per_category":
            flat[k] = round(v, 4)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    # --- JSON (full, για programmatic access) ---
    with open(out / f"{tag}_{ts}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[results] saved → {txt_path}")
    return txt_path
