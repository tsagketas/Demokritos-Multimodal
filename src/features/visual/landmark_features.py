import numpy as np
import face_alignment

# 68-point landmark groups
MOUTH      = list(range(48, 68))   # 20 points
LEFT_EYE   = list(range(36, 42))   # 6 points
RIGHT_EYE  = list(range(42, 48))   # 6 points
LEFT_BROW  = list(range(17, 22))   # 5 points
RIGHT_BROW = list(range(22, 27))   # 5 points
JAW        = list(range(0, 17))    # 17 points
JAW_LEFT   = list(range(0, 8))     # 8 points (left half, without chin)
JAW_RIGHT  = list(range(9, 17))    # 8 points (right half, without chin)

# Specific mouth points for MAR / aperture (relative to MOUTH index 48)
MOUTH_TOP    = 62   # top inner lip    → mouth[14]
MOUTH_BOTTOM = 66   # bottom inner lip → mouth[18]
MOUTH_LEFT   = 60   # left inner corner → mouth[12]
MOUTH_RIGHT  = 64   # right inner corner → mouth[16]

EAR_BLINK_THRESHOLD = 0.2

# Eyes(155) + Jaw(206) + Brows(122) + Mouth(248) = 731
FEATURE_DIM = 731

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


def _ear(eye: np.ndarray) -> float:
    """Eye Aspect Ratio — eye: (6, 2)"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(mouth: np.ndarray) -> float:
    """Mouth Aspect Ratio — mouth: (20, 2), indices relative to landmark 48"""
    vert  = np.linalg.norm(mouth[14] - mouth[18])   # 62→14, 66→18
    horiz = np.linalg.norm(mouth[12] - mouth[16])   # 60→12, 64→16
    return vert / (horiz + 1e-6)


def _region_stats(seq: np.ndarray) -> np.ndarray:
    """seq: (T, N, 2) → position stats + velocity stats → (N*12,)"""
    flat  = seq.reshape(len(seq), -1)                                 # (T, N*2)
    pos   = np.concatenate([flat.mean(0), flat.std(0),
                             flat.min(0),  flat.max(0)])              # N*8
    vel   = np.diff(flat, axis=0)                                     # (T-1, N*2)
    vel_s = np.concatenate([vel.mean(0), vel.std(0)])                 # N*4
    return np.concatenate([pos, vel_s])                               # N*12


def _metric_stats(arr: np.ndarray) -> np.ndarray:
    """arr: (T,) → mean, std, min, max"""
    return np.array([arr.mean(), arr.std(), arr.min(), arr.max()])


def _asymmetry_stats(left_seq: np.ndarray, right_seq: np.ndarray) -> np.ndarray:
    """
    left_seq, right_seq: (T, N, 2)
    returns mean and std of centroid-distance over time
    """
    dist = np.linalg.norm(left_seq.mean(1) - right_seq.mean(1), axis=1)  # (T,)
    return np.array([dist.mean(), dist.std()])


def extract(frames: list, cfg: dict, device: str = "cpu") -> np.ndarray:
    """
    frames : list of np.ndarray (H, W, 3) RGB
    returns: FEATURE_DIM-dimensional vector
    """
    fa = _get_fa(device)

    lm_list, ear_l_list, ear_r_list, mar_list, apt_list = [], [], [], [], []

    for frame in frames:
        preds = fa.get_landmarks(frame)
        if preds is None:
            continue
        lm = preds[0]  # (68, 2)
        lm_list.append(lm)
        ear_l_list.append(_ear(lm[LEFT_EYE]))
        ear_r_list.append(_ear(lm[RIGHT_EYE]))
        mar_list.append(_mar(lm[MOUTH]))
        apt_list.append(abs(lm[MOUTH_TOP][1] - lm[MOUTH_BOTTOM][1]))

    if len(lm_list) < 2:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    lm    = np.array(lm_list)       # (T, 68, 2)
    ear_l = np.array(ear_l_list)    # (T,)
    ear_r = np.array(ear_r_list)    # (T,)
    mar   = np.array(mar_list)      # (T,)
    apt   = np.array(apt_list)      # (T,)

    # ── Eyes (155 dims) ──────────────────────────────────────────────────────
    eye_feat = np.concatenate([
        _region_stats(lm[:, LEFT_EYE]),                                # 72
        _region_stats(lm[:, RIGHT_EYE]),                               # 72
        _metric_stats(ear_l),                                          # 4
        _metric_stats(ear_r),                                          # 4
        _asymmetry_stats(lm[:, LEFT_EYE], lm[:, RIGHT_EYE]),          # 2
        np.array([(ear_l < EAR_BLINK_THRESHOLD).sum() / len(ear_l)]), # 1 — blink rate
    ])

    # ── Jaw (206 dims) ───────────────────────────────────────────────────────
    jaw_feat = np.concatenate([
        _region_stats(lm[:, JAW]),                                     # 204
        _asymmetry_stats(lm[:, JAW_LEFT], lm[:, JAW_RIGHT]),          # 2
    ])

    # ── Eyebrows (122 dims) ──────────────────────────────────────────────────
    brow_feat = np.concatenate([
        _region_stats(lm[:, LEFT_BROW]),                               # 60
        _region_stats(lm[:, RIGHT_BROW]),                              # 60
        _asymmetry_stats(lm[:, LEFT_BROW], lm[:, RIGHT_BROW]),        # 2
    ])

    # ── Mouth (248 dims) ─────────────────────────────────────────────────────
    mouth_feat = np.concatenate([
        _region_stats(lm[:, MOUTH]),                                   # 240
        _metric_stats(mar),                                            # 4
        _metric_stats(apt),                                            # 4
    ])

    return np.concatenate([eye_feat, jaw_feat, brow_feat, mouth_feat]).astype(np.float32)
