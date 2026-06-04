"""
Smoke tests για τα fusion components (Βήματα 1-3).
Χρησιμοποιεί synthetic δεδομένα — δεν χρειάζεται feature extraction.
Χρήση: python test_fusion.py
"""
import sys
import tempfile
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

def ok(label, value=None):
    print(f"  ✅ {label}" + (f"  →  {value}" if value is not None else ""))

def fail(label, err):
    print(f"  ❌ {label}  →  [{type(err).__name__}] {err}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n── FEATURE DATASET ────────────────────────────────────────────────────")

try:
    from data.feature_dataset import FeatureDataset

    AUDIO_DIM  = 768   # wav2vec2
    VISUAL_DIM = 2048  # xception
    N_SAMPLES  = 20

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_dir  = tmp / "audio"
        visual_dir = tmp / "visual"
        audio_dir.mkdir(); visual_dir.mkdir()

        audio_rows  = []
        visual_rows = []

        for i in range(N_SAMPLES):
            # fake mp4 path with a known category
            cat = "FakeVideo-FakeAudio" if i % 2 == 0 else "RealVideo-RealAudio"
            mp4 = f"/data/{cat}/video_{i:03d}.mp4"

            # save synthetic .npy features
            a_path = audio_dir  / f"video_{i:03d}.npy"
            v_path = visual_dir / f"video_{i:03d}_xception.npy"
            np.save(a_path, np.random.randn(AUDIO_DIM).astype(np.float32))
            np.save(v_path, np.random.randn(VISUAL_DIM).astype(np.float32))

            label = 1 if "Fake" in cat else 0
            audio_rows.append({"feature_path": str(a_path), "label": label, "mp4_path": mp4})
            visual_rows.append({"feature_path": str(v_path), "label": label, "mp4_path": mp4})

        # write manifest CSVs
        def write_csv(path, rows):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["feature_path", "label", "mp4_path"])
                w.writeheader(); w.writerows(rows)

        audio_csv  = tmp / "audio_manifest.csv"
        visual_csv = tmp / "visual_manifest.csv"
        write_csv(audio_csv, audio_rows)
        write_csv(visual_csv, visual_rows)

        ds = FeatureDataset(str(audio_csv), str(visual_csv))

        # length
        assert len(ds) == N_SAMPLES, f"expected {N_SAMPLES}, got {len(ds)}"
        ok("FeatureDataset.__len__", len(ds))

        # __getitem__ shapes & types
        audio_vec, visual_vec, label, category = ds[0]
        assert isinstance(audio_vec,  torch.Tensor) and audio_vec.shape  == (AUDIO_DIM,)
        assert isinstance(visual_vec, torch.Tensor) and visual_vec.shape == (VISUAL_DIM,)
        assert isinstance(label, torch.Tensor) and label.dtype == torch.float32
        assert isinstance(category, str) and category in ("FakeVideo-FakeAudio", "RealVideo-RealAudio")
        ok("FeatureDataset.__getitem__", f"audio={tuple(audio_vec.shape)}, visual={tuple(visual_vec.shape)}, label={label.item()}, category={category!r}")

        # inner-join: missing visual sample dropped
        visual_rows_partial = visual_rows[:-3]  # αφαιρούμε 3 samples
        visual_csv2 = tmp / "visual_partial.csv"
        write_csv(visual_csv2, visual_rows_partial)
        ds2 = FeatureDataset(str(audio_csv), str(visual_csv2))
        assert len(ds2) == N_SAMPLES - 3
        ok("FeatureDataset inner-join (drop missing)", f"kept {len(ds2)}/{N_SAMPLES}")

except Exception as e:
    fail("FeatureDataset", e)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VISUAL PCA REDUCER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VISUAL PCA REDUCER ─────────────────────────────────────────────────")

try:
    from models.pca_reducer import VisualPCAReducer, get_n_components

    # get_n_components table
    assert get_n_components("handcrafted", "landmarks") == 536
    assert get_n_components("handcrafted", "xception")  == 536
    assert get_n_components("wav2vec2",    "landmarks") is None
    assert get_n_components("wav2vec2",    "xception")  == 768
    assert get_n_components("hubert",      "landmarks") is None
    assert get_n_components("hubert",      "xception")  == 768
    ok("get_n_components table")

    # PCA needed: 2048 → 768  (n_samples > n_components required by sklearn)
    X_train = np.random.randn(800, 2048).astype(np.float32)
    X_val   = np.random.randn(50,  2048).astype(np.float32)

    reducer = VisualPCAReducer(n_components=768)
    assert reducer.needed is True

    X_train_reduced = reducer.fit_transform(X_train)
    assert X_train_reduced.shape == (800, 768)
    ok("fit_transform (2048→768)", f"shape={X_train_reduced.shape}")

    X_val_reduced = reducer.transform(X_val)
    assert X_val_reduced.shape == (50, 768)
    ok("transform val (no refit)", f"shape={X_val_reduced.shape}")

    # save / load
    with tempfile.TemporaryDirectory() as tmp:
        pkl = Path(tmp) / "pca.pkl"
        reducer.save(pkl)
        loaded = VisualPCAReducer.load(pkl)
        X_check = loaded.transform(X_val)
        assert np.allclose(X_check, X_val_reduced)
        ok("save / load", f"pkl={pkl.name}, output matches")

    # PCA not needed: None
    reducer_none = VisualPCAReducer(n_components=None)
    assert reducer_none.needed is False
    X_pass = np.random.randn(10, 731).astype(np.float32)
    out = reducer_none.fit_transform(X_pass)
    assert np.array_equal(out, X_pass)
    ok("n_components=None (passthrough)", f"shape unchanged {out.shape}")

    # transform before fit → RuntimeError
    try:
        VisualPCAReducer(n_components=128).transform(X_val)
        fail("RuntimeError on unfit transform", "no error raised")
    except RuntimeError:
        ok("RuntimeError on unfit transform")

except Exception as e:
    fail("VisualPCAReducer", e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EARLY FUSION MLP
# ─────────────────────────────────────────────────────────────────────────────
print("\n── EARLY FUSION MLP ───────────────────────────────────────────────────")

try:
    from models.early_fusion import EarlyFusionMLP

    BATCH = 16

    for audio_dim, visual_dim, label in [
        (536, 536,  "handcrafted + landmarks/xception (PCA→536)"),
        (768, 731,  "wav2vec2 + landmarks (no PCA)"),
        (768, 768,  "wav2vec2/HuBERT + xception (PCA→768)"),
    ]:
        model = EarlyFusionMLP(audio_dim=audio_dim, visual_dim=visual_dim)
        audio_t  = torch.randn(BATCH, audio_dim)
        visual_t = torch.randn(BATCH, visual_dim)

        with torch.no_grad():
            logits = model(audio_t, visual_t)

        assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in logits"
        ok(label, f"logits shape={tuple(logits.shape)}, range=[{logits.min():.2f}, {logits.max():.2f}]")

    # BCEWithLogitsLoss συμβατότητα
    model  = EarlyFusionMLP(audio_dim=768, visual_dim=768)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logits  = model(torch.randn(8, 768), torch.randn(8, 768))
    labels  = torch.randint(0, 2, (8,)).float()
    loss    = loss_fn(logits, labels)
    assert not torch.isnan(loss)
    ok("BCEWithLogitsLoss compatibility", f"loss={loss.item():.4f}")

except Exception as e:
    fail("EarlyFusionMLP", e)


print("\n── DONE ────────────────────────────────────────────────────────────────\n")
