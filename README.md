# Multimodal Deepfake Speech Detection

This repository contains the code and resources for our research project on **Multimodal Deepfake Speech Detection**, developed as part of the Multimodal Information Processing coursework for the MSc in Artificial Intelligence at the National Centre for Scientific Research "Demokritos" & University of Piraeus.

## 📌 Project Overview
The primary goal of this project is to detect AI-generated speech and deepfake videos by leveraging both audio and visual modalities. We aim to evaluate and compare the effectiveness of audio-only, visual-only, and multimodal fusion techniques for binary classification (Real vs. Fake).

## 🚀 Key Features & Methodology
* **Audio Feature Extraction:** Exploring both traditional hand-crafted features (MFCCs/Mel-Spectrograms) and pre-trained Self-Supervised Learning (SSL) models (`wav2vec2`, `HuBERT`).
* **Visual Feature Extraction:** Utilizing lightweight pre-trained CNNs focused on lip movements to ensure computational efficiency.
* **Fusion Strategies:** 
  * Baseline Early Fusion
  * Dimensionality-reduced Early Fusion (using PCA to prevent modality domination)
  * Late Fusion of modality-specific posteriors
* **Explainability:** Implementing temporal localization to identify specific time segments (windows) containing manipulated content.

## 📊 Datasets
* **Training/Validation:** [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
* **Testing:** Cross-dataset evaluation, including a small custom "in-the-wild" dataset for generalization assessment.

## 👥 Authors
* Garo Aristotelis 
* Plessias Giorgos 
* Tsagketas Orestis 

---
*Supervised by: Dr. Theodoros Giannakopoulos*