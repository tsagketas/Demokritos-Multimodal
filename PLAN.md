# Multimodal Deepfake Speech Detection — Project Plan

## Dataset
**FakeAVCeleb_v1.2** — 4 categories:
- `RealVideo-RealAudio` → label 0 (real)
- `RealVideo-FakeAudio` → label 1 (fake)
- `FakeVideo-RealAudio` → label 1 (fake)
- `FakeVideo-FakeAudio` → label 1 (fake)

Split: 70% train / 15% val / 15% test (stratified, seed=42)

---

## File Structure

```
src/
├── data/
│   ├── dataset.py
│   └── preprocessing.py
├── features/
│   ├── audio/
│   │   ├── handcrafted.py
│   │   ├── wav2vec2.py
│   │   └── hubert.py
│   └── visual/
│       └── lip_features.py
├── models/
│   ├── audio_classifier.py
│   ├── early_fusion.py
│   └── late_fusion.py
├── utils/
│   ├── config.py
│   └── results.py
├── train.py
├── eval.py
└── explain.py

configs/
├── experiment.yaml   ← master, εδώ βάζεις όνομα run + overrides
├── data.yaml
├── features.yaml
├── train.yaml
├── fusion.yaml
└── eval.yaml

outputs/
├── features/         ← cached embeddings (wav2vec2, hubert, lip)
└── experiments/
    └── <timestamp_name>/
        ├── config_snapshot.yaml
        ├── metrics/
        │   ├── eval_TIMESTAMP.txt
        │   ├── eval_TIMESTAMP.json
        │   └── all_runs.csv
        ├── plots/
        └── checkpoints/
```

---

## Modules

### Module 1 — Data Pipeline
**Files:** `src/data/dataset.py`, `src/data/preprocessing.py`
**Libraries:** pandas, sklearn, torchaudio, opencv-python

- `dataset.py`: διαβάζει `meta_data.csv`, assigns labels, κάνει train/val/test split, επιστρέφει paths (όχι τα data)
- `preprocessing.py`: mp4 path → waveform tensor (16kHz) + lip frames tensor

---

### Module 2 — Audio Feature Extraction
**Files:** `src/features/audio/`
**Libraries:** librosa, transformers

- `handcrafted.py`: MFCCs (40) + Mel-Spectrogram (128 mels) + deltas → pooling → fixed-size vector
- `wav2vec2.py`: `facebook/wav2vec2-base` → extract last transformer layer → mean pooling
- `hubert.py`: `facebook/hubert-base-ls960` → extract last transformer layer → mean pooling

Όλα τα features cached στο `outputs/features/<method>/`.

---

### Module 3 — Visual Feature Extraction
**Files:** `src/features/visual/lip_features.py`
**Libraries:** mediapipe, torchvision (mobilenet_v2)

- MediaPipe για lip landmark detection → crop lip region per frame
- MobileNetV2 pretrained → frame embedding
- Temporal mean pooling → fixed-size vector
- Cached στο `outputs/features/visual/`

---

### Module 4 — Classifiers
**Files:** `src/models/`
**Libraries:** torch.nn, sklearn (PCA)

- `audio_classifier.py`: MLP [hidden: 256→128] + dropout 0.3, binary output — τρέχει ×3 (handcrafted, wav2vec2, hubert)
- `early_fusion.py`: concat(audio_vec, visual_vec) → PCA (n=128) → MLP
- `late_fusion.py`: audio_posterior + visual_posterior → average ή learned weights → decision

---

### Module 5 — Train / Eval
**Files:** `src/train.py`, `src/eval.py`
**Libraries:** torch, tqdm, sklearn.metrics

- `train.py`: training loop, early stopping (patience=5), checkpointing best model by val_AUC
- `eval.py`: accuracy, AUC, EER, F1, precision, recall — per category breakdown — αποθήκευση σε .txt/.csv/.json

---

### Module 6 — Explainability
**Files:** `src/explain.py`
**Libraries:** torch, matplotlib

- Sliding window (0.5s window, 0.1s step) πάνω στο audio
- Για κάθε window → model score → plot temporal heatmap
- Εντοπισμός ποιο segment trigger το fake prediction

---

## Experiments (σύγκριση)

| Run | Audio FE | Visual | Fusion | Expected |
|-----|----------|--------|--------|----------|
| 1 | Handcrafted | — | audio-only | baseline |
| 2 | wav2vec2 | — | audio-only | better |
| 3 | HuBERT | — | audio-only | better |
| 4 | wav2vec2 | Lip/MobileNet | Early (PCA) | multimodal |
| 5 | wav2vec2 | Lip/MobileNet | Late (avg) | multimodal |
| 6 | HuBERT | Lip/MobileNet | Late (avg) | best? |

---

## How to Run an Experiment

1. Αλλάζεις `configs/experiment.yaml` (όνομα + overrides)
2. `docker exec -it multimodal python train.py`
3. Αποτελέσματα στο `outputs/experiments/<timestamp_name>/`

---

## Professor Guidance (από email)
- 2-3 audio FE methods: handcrafted + wav2vec2 + HuBERT
- Visual: μόνο 1 απλή μέθοδος (MobileNet lip), όχι state-of-the-art
- Early fusion: PCA για να μην κάνει dominate το μεγαλύτερο modality
- Late fusion: combine posteriors
- Explainability: temporal localization αρκεί
- Generalization: small custom test set (ASVspoof δεν έχει video)
