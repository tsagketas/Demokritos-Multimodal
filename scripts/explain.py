"""
Post-hoc explainability for a single video.
Run AFTER run_experiments.py (needs trained checkpoints).

Usage:
    # Audio-only
    python explain.py --video path/to/video.mp4 --mode audio_only \
        --audio_ckpt outputs/experiments/02_audio_only_wav2vec2/checkpoints/best_audio.pt \
        --audio_method wav2vec2

    # Video-only (landmarks only)
    python explain.py --video path/to/video.mp4 --mode video_only \
        --video_ckpt outputs/experiments/04_video_only_landmarks/checkpoints/best_video.pt \
        --visual_method landmarks

    # Early fusion
    python explain.py --video path/to/video.mp4 --mode early_fusion \
        --fusion_ckpt outputs/experiments/09_early_wav2vec2_landmarks/checkpoints/best_model.pt \
        --pca_ckpt   outputs/experiments/09_early_wav2vec2_landmarks/pca.pkl \
        --audio_method wav2vec2 --visual_method landmarks

    # Late fusion
    python explain.py --video path/to/video.mp4 --mode late_fusion \
        --audio_ckpt outputs/experiments/02_audio_only_wav2vec2/checkpoints/best_audio.pt \
        --video_ckpt outputs/experiments/04_video_only_landmarks/checkpoints/best_video.pt \
        --audio_method wav2vec2 --visual_method landmarks
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.preprocessing import extract_audio, extract_frames
from features.visual.landmark_features import extract as extract_landmarks, LANDMARK_REGIONS
from models.unimodal_classifier import UnimodalClassifier
from models.early_fusion import EarlyFusionMLP
from models.late_fusion import LateFusionCombiner
from models.pca_reducer import VisualPCAReducer
from utils.config import load_config

SAMPLE_RATE    = 16000
WINDOW_SAMPLES = int(0.5 * SAMPLE_RATE)   # 0.5s
STEP_SAMPLES   = int(0.1 * SAMPLE_RATE)   # 0.1s step


# ── Feature helpers ───────────────────────────────────────────────────────────

def _extract_audio_feature(waveform: torch.Tensor, method: str, cfg: dict,
                            device: str) -> np.ndarray:
    """Extract a single audio feature vector from a waveform."""
    audio_cfg = cfg["features"]["audio"]
    if method == "handcrafted":
        from features.audio import handcrafted
        return handcrafted.extract(waveform, SAMPLE_RATE, audio_cfg["handcrafted"], device)
    elif method == "wav2vec2":
        from features.audio import wav2vec2
        return wav2vec2.extract(waveform, SAMPLE_RATE, audio_cfg["wav2vec2"], device)
    elif method == "hubert":
        from features.audio import hubert
        return hubert.extract(waveform, SAMPLE_RATE, audio_cfg["hubert"], device)
    raise ValueError(f"Unknown audio method: {method}")


def _preprocess_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Convert to mono, normalize."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    y = waveform.squeeze().numpy()
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return torch.from_numpy(y).unsqueeze(0)


# ── Audio sliding window ──────────────────────────────────────────────────────

def audio_sliding_window(waveform: torch.Tensor, score_fn, method: str,
                          cfg: dict, device: str) -> tuple:
    """
    Slides a 0.5s window with 0.1s step over the waveform.
    score_fn(audio_vec: np.ndarray) -> float in [0, 1]
    Returns (times_sec, scores).
    """
    y      = waveform.squeeze().numpy()
    n      = len(y)
    times, scores = [], []

    start = 0
    while start + WINDOW_SAMPLES <= n:
        window   = torch.from_numpy(y[start:start + WINDOW_SAMPLES]).unsqueeze(0)
        feat_vec = _extract_audio_feature(window, method, cfg, device)
        score    = score_fn(feat_vec)
        times.append((start + WINDOW_SAMPLES / 2) / SAMPLE_RATE)
        scores.append(float(score))
        start += STEP_SAMPLES

    return np.array(times), np.array(scores)


# ── Visual region ablation ────────────────────────────────────────────────────

def visual_region_ablation(landmark_vec: np.ndarray, score_fn) -> dict:
    """
    Zeros each landmark region in turn.
    score_fn(visual_vec: np.ndarray) -> float in [0, 1]
    Returns dict: {region_name: {"baseline": float, "ablated": float, "drop": float}}
    """
    baseline = score_fn(landmark_vec)
    results  = {"baseline": float(baseline)}

    for region, (start, end) in LANDMARK_REGIONS.items():
        ablated_vec            = landmark_vec.copy()
        ablated_vec[start:end] = 0.0
        ablated_score          = score_fn(ablated_vec)
        results[region] = {
            "baseline": float(baseline),
            "ablated":  float(ablated_score),
            "drop":     float(baseline - ablated_score),
        }
    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_temporal_heatmap(times, scores, out_dir: Path, tag: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, scores, color="#3498db", lw=1.5)
    ax.fill_between(times, scores, 0.5, where=(np.array(scores) > 0.5),
                    alpha=0.3, color="#e74c3c", label="P(fake) > 0.5")
    ax.fill_between(times, scores, 0.5, where=(np.array(scores) <= 0.5),
                    alpha=0.3, color="#2ecc71", label="P(fake) ≤ 0.5")
    ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(fake)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_audio_heatmap.png", dpi=150)
    plt.close(fig)

    lines = [
        f"AUDIO TEMPORAL HEATMAP — {title}",
        f"Window: 0.5s  Step: 0.1s  Windows: {len(times)}",
        "",
        f"{'Time (s)':<12} {'P(fake)':<10} {'Label'}",
        "-" * 36,
    ]
    for t, s in zip(times, scores):
        label = "FAKE" if s > 0.5 else "real"
        lines.append(f"{t:<12.2f} {s:<10.4f} {label}")
    with open(out_dir / f"{tag}_audio_heatmap.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot_region_ablation(ablation: dict, out_dir: Path, tag: str, title: str):
    regions  = list(LANDMARK_REGIONS.keys())
    drops    = [ablation[r]["drop"]    for r in regions]
    ablated  = [ablation[r]["ablated"] for r in regions]
    baseline = ablation["baseline"]
    colors   = ["#e74c3c" if d > 0 else "#2ecc71" for d in drops]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(regions, drops, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Score drop  (baseline − ablated)")
    ax.set_title(f"{title}\nBaseline P(fake) = {baseline:.4f}")
    for bar, drop, abl in zip(bars, drops, ablated):
        ax.text(drop + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{abl:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_region_ablation.png", dpi=150)
    plt.close(fig)

    lines = [
        f"VISUAL REGION ABLATION — {title}",
        f"Baseline P(fake): {baseline:.4f}",
        "",
        f"{'Region':<10} {'Baseline':<12} {'Ablated':<12} {'Drop':<10} {'Impact'}",
        "-" * 56,
    ]
    for r in regions:
        d   = ablation[r]["drop"]
        abl = ablation[r]["ablated"]
        impact = "HIGH" if abs(d) > 0.05 else "low"
        lines.append(f"{r:<10} {baseline:<12.4f} {abl:<12.4f} {d:<10.4f} {impact}")
    with open(out_dir / f"{tag}_region_ablation.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Mode runners ──────────────────────────────────────────────────────────────

@torch.no_grad()
def explain_audio_only(waveform, frames, audio_method, audio_ckpt_path,
                       cfg, device, out_dir, tag):
    ckpt  = torch.load(audio_ckpt_path, map_location=device)
    model = UnimodalClassifier(input_dim=ckpt["input_dim"],
                               hidden_dims=ckpt.get("hidden_dims", [256, 128])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def score_fn(vec):
        t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
        return torch.sigmoid(model(t)).item()

    times, scores = audio_sliding_window(waveform, score_fn, audio_method, cfg, device)
    _plot_temporal_heatmap(times, scores, out_dir, tag, f"Audio-only ({audio_method})")
    print(f"  [audio] {len(times)} windows  mean_score={scores.mean():.4f}")


@torch.no_grad()
def explain_video_only(waveform, frames, visual_method, video_ckpt_path,
                       cfg, device, out_dir, tag):
    if visual_method != "landmarks":
        print("  [visual] Region ablation only supported for landmarks. Skipping.")
        return

    ckpt  = torch.load(video_ckpt_path, map_location=device)
    model = UnimodalClassifier(input_dim=ckpt["input_dim"],
                               hidden_dims=ckpt.get("hidden_dims", [256, 128])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    visual_cfg = cfg["features"]["visual"]
    lm_vec = extract_landmarks(frames, visual_cfg, device)

    def score_fn(vec):
        t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
        return torch.sigmoid(model(t)).item()

    ablation = visual_region_ablation(lm_vec, score_fn)
    _plot_region_ablation(ablation, out_dir, tag, f"Video-only (landmarks)")
    print(f"  [visual] baseline={ablation['baseline']:.4f}  "
          + "  ".join(f"{r}={ablation[r]['drop']:+.3f}" for r in LANDMARK_REGIONS))


@torch.no_grad()
def explain_early_fusion(waveform, frames, audio_method, visual_method,
                         fusion_ckpt_path, pca_ckpt_path,
                         cfg, device, out_dir, tag):
    ckpt  = torch.load(fusion_ckpt_path, map_location=device)
    model = EarlyFusionMLP(
        audio_dim=ckpt["audio_dim"], visual_dim=ckpt["visual_dim"],
        hidden_dims=ckpt.get("hidden_dims", [256, 128]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    reducer = VisualPCAReducer.load(pca_ckpt_path) if pca_ckpt_path else VisualPCAReducer(None)

    # ── Baseline features ─────────────────────────────────────────────────────
    baseline_audio = _extract_audio_feature(waveform, audio_method, cfg, device)

    if visual_method == "landmarks":
        baseline_visual = extract_landmarks(frames, cfg["features"]["visual"], device)
    else:
        print("  [visual] Early fusion ablation only supported for landmarks visual.")
        baseline_visual = None

    baseline_visual_pca = reducer.transform(baseline_visual[np.newaxis])[0] \
        if baseline_visual is not None else None

    # ── Audio sliding window (fix visual) ─────────────────────────────────────
    def audio_score_fn(audio_vec):
        a = torch.from_numpy(audio_vec).float().unsqueeze(0).to(device)
        v = torch.from_numpy(baseline_visual_pca).float().unsqueeze(0).to(device) \
            if baseline_visual_pca is not None \
            else torch.zeros(1, ckpt["visual_dim"]).to(device)
        return torch.sigmoid(model(a, v)).item()

    times, scores = audio_sliding_window(waveform, audio_score_fn, audio_method, cfg, device)
    _plot_temporal_heatmap(times, scores, out_dir, tag,
                           f"Early Fusion audio ({audio_method}) — visual fixed")

    # ── Visual region ablation (fix audio) ────────────────────────────────────
    if baseline_visual is not None:
        audio_t = torch.from_numpy(baseline_audio).float().unsqueeze(0).to(device)

        def visual_score_fn(vec):
            pca_vec = reducer.transform(vec[np.newaxis])[0]
            v = torch.from_numpy(pca_vec).float().unsqueeze(0).to(device)
            return torch.sigmoid(model(audio_t, v)).item()

        ablation = visual_region_ablation(baseline_visual, visual_score_fn)
        _plot_region_ablation(ablation, out_dir, tag,
                              f"Early Fusion visual (landmarks) — audio fixed")

    print(f"  [early_fusion] audio windows={len(times)}  mean={scores.mean():.4f}")


@torch.no_grad()
def explain_late_fusion(waveform, frames, audio_method, visual_method,
                        audio_ckpt_path, video_ckpt_path,
                        cfg, device, out_dir, tag):
    audio_ckpt  = torch.load(audio_ckpt_path, map_location=device)
    video_ckpt  = torch.load(video_ckpt_path, map_location=device)

    audio_model = UnimodalClassifier(input_dim=audio_ckpt["input_dim"],
                                     hidden_dims=audio_ckpt.get("hidden_dims", [256, 128])).to(device)
    audio_model.load_state_dict(audio_ckpt["model_state"])
    audio_model.eval()

    video_model = UnimodalClassifier(input_dim=video_ckpt["input_dim"],
                                     hidden_dims=video_ckpt.get("hidden_dims", [256, 128])).to(device)
    video_model.load_state_dict(video_ckpt["model_state"])
    video_model.eval()

    # ── Audio sliding window ──────────────────────────────────────────────────
    def audio_score_fn(vec):
        t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
        return torch.sigmoid(audio_model(t)).item()

    times, scores = audio_sliding_window(waveform, audio_score_fn, audio_method, cfg, device)
    _plot_temporal_heatmap(times, scores, out_dir, tag,
                           f"Late Fusion — audio sub-model ({audio_method})")

    # ── Visual region ablation ────────────────────────────────────────────────
    if visual_method == "landmarks":
        lm_vec = extract_landmarks(frames, cfg["features"]["visual"], device)

        def visual_score_fn(vec):
            t = torch.from_numpy(vec).float().unsqueeze(0).to(device)
            return torch.sigmoid(video_model(t)).item()

        ablation = visual_region_ablation(lm_vec, visual_score_fn)
        _plot_region_ablation(ablation, out_dir, tag,
                              f"Late Fusion — visual sub-model (landmarks)")
    else:
        print("  [visual] Region ablation skipped for xception.")

    print(f"  [late_fusion] audio windows={len(times)}  mean={scores.mean():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explainability for a single video")
    parser.add_argument("--video",        required=True)
    parser.add_argument("--mode",         required=True,
                        choices=["audio_only", "video_only", "early_fusion", "late_fusion"])
    parser.add_argument("--audio_method", default="wav2vec2",
                        choices=["handcrafted", "wav2vec2", "hubert"])
    parser.add_argument("--visual_method",default="landmarks",
                        choices=["landmarks", "xception"])
    parser.add_argument("--audio_ckpt",   default=None)
    parser.add_argument("--video_ckpt",   default=None)
    parser.add_argument("--fusion_ckpt",  default=None)
    parser.add_argument("--pca_ckpt",     default=None)
    parser.add_argument("--config",       default="configs/experiment.yaml")
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_name = Path(args.video).stem
    out_dir    = Path(args.output_dir) if args.output_dir \
                 else Path("/workspace/outputs/explanations") / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{video_name}_{args.mode}"

    print(f"\n[explain] video={args.video}")
    print(f"[explain] mode={args.mode}  device={device}")
    print(f"[explain] output → {out_dir}\n")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("[explain] extracting audio + frames...")
    waveform = extract_audio(args.video, sample_rate=SAMPLE_RATE)
    waveform = _preprocess_waveform(waveform)
    frames   = extract_frames(args.video, max_frames=cfg["data"]["video"]["max_frames"])
    duration = waveform.shape[-1] / SAMPLE_RATE
    print(f"  audio: {duration:.1f}s  frames: {len(frames)}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    common = dict(waveform=waveform, frames=frames, cfg=cfg, device=device,
                  out_dir=out_dir, tag=tag)

    if args.mode == "audio_only":
        explain_audio_only(audio_method=args.audio_method,
                           audio_ckpt_path=args.audio_ckpt, **common)

    elif args.mode == "video_only":
        explain_video_only(visual_method=args.visual_method,
                           video_ckpt_path=args.video_ckpt, **common)

    elif args.mode == "early_fusion":
        explain_early_fusion(audio_method=args.audio_method,
                             visual_method=args.visual_method,
                             fusion_ckpt_path=args.fusion_ckpt,
                             pca_ckpt_path=args.pca_ckpt, **common)

    elif args.mode == "late_fusion":
        explain_late_fusion(audio_method=args.audio_method,
                            visual_method=args.visual_method,
                            audio_ckpt_path=args.audio_ckpt,
                            video_ckpt_path=args.video_ckpt, **common)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "video":         args.video,
        "mode":          args.mode,
        "audio_method":  args.audio_method,
        "visual_method": args.visual_method,
        "duration_sec":  round(duration, 2),
        "n_frames":      len(frames),
        "output_dir":    str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[explain] done → {out_dir}")


if __name__ == "__main__":
    main()
