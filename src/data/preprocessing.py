import cv2
import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T


def extract_audio(mp4_path: str, sample_rate: int = 16000,
                  trim_silence: bool = True, normalize: bool = True) -> torch.Tensor:
    waveform, sr = torchaudio.load(mp4_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    y = waveform.squeeze().numpy()

    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)

    if normalize and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return torch.from_numpy(y).unsqueeze(0)  # shape: (1, T)


def extract_frames(mp4_path: str, max_frames: int = 100) -> list[np.ndarray]:
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = _sample_indices(total, max_frames)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # (H, W, 3) RGB

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {mp4_path}")

    return frames  # list of np.ndarray (H, W, 3)


def _sample_indices(total: int, max_frames: int) -> list[int]:
    if total <= max_frames:
        return list(range(total))
    step = total / max_frames
    return [int(i * step) for i in range(max_frames)]
