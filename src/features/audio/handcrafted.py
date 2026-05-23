import numpy as np
import torch
import librosa
import parselmouth

# MFCCs(40×3) + spectral(1+1+1+1+7) + ZCR(1) + RMS(1) + F0(1) = 134 rows × 4 stats
FEATURE_DIM = 536


def extract(waveform: torch.Tensor, sample_rate: int, cfg: dict, device: str = "cpu") -> np.ndarray:
    y = waveform.squeeze().numpy()

    n_mfcc     = cfg.get("n_mfcc", 40)
    hop_length = cfg.get("hop_length", 512)
    n_fft      = cfg.get("n_fft", 2048)

    feature_matrices = []

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    feature_matrices.append(mfcc)
    feature_matrices.append(librosa.feature.delta(mfcc))
    feature_matrices.append(librosa.feature.delta(mfcc, order=2))

    # Spectral features
    feature_matrices.append(librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))
    feature_matrices.append(librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))
    S    = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0, keepdims=True))
    flux = np.pad(flux, ((0, 0), (1, 0)))
    feature_matrices.append(flux)
    feature_matrices.append(librosa.feature.spectral_rolloff(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))
    feature_matrices.append(librosa.feature.spectral_contrast(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))

    # ZCR + RMS energy
    feature_matrices.append(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))
    feature_matrices.append(librosa.feature.rms(y=y, hop_length=hop_length))

    # F0 via Praat (parselmouth) — fast C++ backend, 0 for unvoiced frames
    snd   = parselmouth.Sound(y, sampling_frequency=sample_rate)
    pitch = snd.to_pitch(time_step=hop_length / sample_rate)
    f0    = pitch.selected_array['frequency'].reshape(1, -1)   # (1, T')
    feature_matrices.append(f0)

    # Aggregate each matrix: mean, min, max, std across time
    parts = []
    for mat in feature_matrices:
        parts.append(mat.mean(axis=1))
        parts.append(mat.min(axis=1))
        parts.append(mat.max(axis=1))
        parts.append(mat.std(axis=1))

    return np.concatenate(parts).astype(np.float32)
