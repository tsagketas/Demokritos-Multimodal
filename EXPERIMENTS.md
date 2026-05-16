# Experiments Overview

## Feature Extraction
- **Audio:** Handcrafted (536d) | wav2vec2 sequence (T×768) | HuBERT sequence (T×768)
- **Visual:** MediaPipe lip landmarks → temporal stats

## Classifiers
- **MLP** — για handcrafted features + fusion
- **LSTM** — για wav2vec2/HuBERT sequences
- **Transformer** — για wav2vec2/HuBERT sequences (attention = explainability)

## 22 Experiments

### Audio-only (7)
| # | FE | Classifier |
|---|---|---|
| 1 | Handcrafted | MLP |
| 2 | wav2vec2 | MLP |
| 3 | wav2vec2 | LSTM |
| 4 | wav2vec2 | Transformer |
| 5 | HuBERT | MLP |
| 6 | HuBERT | LSTM |
| 7 | HuBERT | Transformer |

### Visual-only (1)
| # | FE | Classifier |
|---|---|---|
| 8 | Lip Landmarks | MLP |

### Multimodal (14)
| # | Audio FE | Classifier | Fusion |
|---|---|---|---|
| 9  | Handcrafted | MLP | Early |
| 10 | Handcrafted | MLP | Late |
| 11 | wav2vec2 | MLP | Early |
| 12 | wav2vec2 | MLP | Late |
| 13 | wav2vec2 | LSTM | Early |
| 14 | wav2vec2 | LSTM | Late |
| 15 | wav2vec2 | Transformer | Early |
| 16 | wav2vec2 | Transformer | Late |
| 17 | HuBERT | MLP | Early |
| 18 | HuBERT | MLP | Late |
| 19 | HuBERT | LSTM | Early |
| 20 | HuBERT | LSTM | Late |
| 21 | HuBERT | Transformer | Early |
| 22 | HuBERT | Transformer | Late |

## Evaluation Metrics
Accuracy, AUC, EER, F1 — per experiment + σύγκριση όλων

## Explainability
Transformer attention weights → temporal localization του deepfake
