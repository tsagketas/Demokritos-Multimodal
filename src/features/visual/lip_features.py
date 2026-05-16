import numpy as np
import torch
import face_alignment

# 68-point model: lip landmarks are indices 48-67
LIP_INDICES   = list(range(48, 68))   # 20 points
UPPER_CENTER  = 62   # top of inner lip
LOWER_CENTER  = 66   # bottom of inner lip

_fa = None


def _get_fa(device: str = "cpu"):
    global _fa
    if _fa is None:
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
        )
    return _fa


def extract(frames: torch.Tensor, cfg: dict, device: str = "cpu") -> np.ndarray:
    """
    frames: (N, 3, H, W) uint8
    returns: fixed-size 244-dim feature vector
    """
    fa = _get_fa(device)
    landmarks_seq = []
    apertures     = []

    for frame in frames:
        img_np = frame.permute(1, 2, 0).numpy()
        preds  = fa.get_landmarks(img_np)

        if preds is None:
            continue

        lm = preds[0]  # (68, 2)
        lip_coords = lm[LIP_INDICES]   # (20, 2)
        landmarks_seq.append(lip_coords)

        aperture = abs(lm[UPPER_CENTER][1] - lm[LOWER_CENTER][1])
        apertures.append(aperture)

    if len(landmarks_seq) < 2:
        return np.zeros(244, dtype=np.float32)

    seq  = np.array(landmarks_seq)       # (T, 20, 2)
    apt  = np.array(apertures)           # (T,)
    flat = seq.reshape(len(seq), -1)     # (T, 40)

    pos_stats = np.concatenate([
        flat.mean(axis=0), flat.std(axis=0),
        flat.min(axis=0),  flat.max(axis=0),
    ])  # 160 dims

    velocity  = np.diff(flat, axis=0)   # (T-1, 40)
    vel_stats = np.concatenate([
        velocity.mean(axis=0), velocity.std(axis=0),
    ])  # 80 dims

    apt_stats = np.array([apt.mean(), apt.std(), apt.min(), apt.max()])  # 4 dims

    return np.concatenate([pos_stats, vel_stats, apt_stats]).astype(np.float32)  # 244 dims
