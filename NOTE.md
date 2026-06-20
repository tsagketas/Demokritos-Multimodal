# Αποτελέσματα πριν το rerun

![AUC Σύγκριση](outputs/comparison/comparison_auc.png)

```
Experiment                        AUC     EER     F1      ACC
--------------------------------------------------------------
06_early_handcrafted_landmarks    0.9322  0.1616  0.9786  0.9586
01_audio_only_handcrafted         0.9274  0.1496  0.9645  0.9323
12_late_handcrafted_landmarks     0.9255  0.1573  0.9881  0.9765
02_audio_only_wav2vec2            0.9122  0.1999  0.9802  0.9617
03_audio_only_hubert              0.9074  0.2001  0.9750  0.9518
14_late_wav2vec2_landmarks        0.8896  0.1895  0.9878  0.9759
16_late_hubert_landmarks          0.8724  0.1849  0.9880  0.9765
08_early_wav2vec2_landmarks       0.7697  0.2902  0.9883  0.9768
10_early_hubert_landmarks         0.6983  0.3985  0.9883  0.9768
04_video_only_landmarks           0.5712  0.4295  0.9881  0.9765
```

---

## Αλλαγές για rerun

**1. pos_weight (`src/train.py`)**
Προστέθηκε `BCEWithLogitsLoss(pos_weight=0.2)` — εξισορροπεί το residual 5:1 imbalance που μένει μετά τον WeightedRandomSampler. Διαβάζεται δυναμικά από `loader.sampler_target_ratio`.

**2. Late fusion weight (`configs/fusion.yaml`)**
`weight: 0.5 → 0.7` — επηρεάζει ΜΟΝΟ τα late fusion experiments (12, 14, 16), όχι audio-only/video-only/early fusion.

Αιτιολόγηση (data-driven, από τα αποτελέσματα):
- `score = weight × audio_score + (1-weight) × video_score`
- Με `weight=0.5`: ίσο βάρος σε audio (AUC 0.93) και video (AUC 0.57 ≈ τυχαίο) → το video "μολύνει" το audio
- Με `weight=0.7`: το audio κυριαρχεί (70%), το video συνεισφέρει ελάχιστα (30%)
- Το 0.7 δεν είναι arbitrary — δικαιολογείται από το ίδιο το AUC gap μεταξύ των δύο modalities

**3. AUC heatmap (`src/utils/plots.py`)**
Skip όταν όλες οι τιμές NaN — αποφεύγει κενό plot.
