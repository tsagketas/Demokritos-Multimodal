import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

FEATURE_DIM = 768

_model     = None
_processor = None


def _load(model_name: str, device: str):
    global _model, _processor
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(model_name)
        _model     = Wav2Vec2Model.from_pretrained(model_name).to(device)
        _model.eval()


def extract(waveform: torch.Tensor, sample_rate: int, cfg: dict, device: str = "cpu") -> np.ndarray:
    _load(cfg.get("model_name", "facebook/wav2vec2-base"), device)

    y = waveform.squeeze().numpy()
    inputs = _processor(y, sampling_rate=sample_rate, return_tensors="pt",
                        padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = _model(input_values, output_hidden_states=True)

    layer = cfg.get("layer", -1)
    hidden = outputs.hidden_states[layer]  # (1, T, 768)
    pooled = hidden.mean(dim=1).squeeze(0)  # (768,)

    return pooled.cpu().numpy().astype(np.float32)
