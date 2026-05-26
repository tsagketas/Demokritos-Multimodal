import numpy as np
import torch
import torchvision.transforms as T
import timm
import face_alignment

FEATURE_DIM = 2048

_model = None
_fa    = None

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _get_model(device: str = "cpu"):
    global _model
    if _model is None:
        _model = timm.create_model("xception", pretrained=True, num_classes=0)
        _model.eval()
        _model = _model.to(device)
    return _model


def _get_fa(device: str = "cpu"):
    global _fa
    if _fa is None:
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
        )
    return _fa


def _crop_face(frame: np.ndarray, fa) -> np.ndarray | None:
    preds = fa.get_landmarks(frame)
    if preds is None:
        return None
    lm    = preds[0]
    x_min, y_min = lm.min(0).astype(int)
    x_max, y_max = lm.max(0).astype(int)
    h, w  = frame.shape[:2]
    pad   = int(0.15 * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)
    crop  = frame[y_min:y_max, x_min:x_max]
    return crop if crop.size > 0 else None


def extract(frames: list, cfg: dict, device: str = "cpu") -> np.ndarray:
    model = _get_model(device)
    fa    = _get_fa(device)
    embeddings = []

    for frame in frames:
        crop = _crop_face(frame, fa)
        if crop is None:
            continue
        tensor = _transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor).squeeze()
        embeddings.append(feat.cpu().numpy())

    if not embeddings:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    return np.mean(embeddings, axis=0).astype(np.float32)
