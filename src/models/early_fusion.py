import torch
import torch.nn as nn
from models.unimodal_classifier import _build_mlp


class EarlyFusionMLP(nn.Module):
    """
    Early Fusion classifier for multimodal deepfake detection.

    Concatenates audio and (PCA-reduced) visual feature vectors,
    then passes them through an MLP whose depth is controlled by hidden_dims.

    Returns a raw logit (no Sigmoid) — use BCEWithLogitsLoss during training
    and apply Sigmoid manually at inference time.
    """

    def __init__(self, audio_dim: int, visual_dim: int,
                 hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        self.hidden_dims = hidden_dims or [256, 128]
        self.classifier  = _build_mlp(audio_dim + visual_dim, self.hidden_dims, dropout)

    def forward(self, audio_vec: torch.Tensor, visual_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio_vec, visual_vec], dim=-1)
        return self.classifier(x).squeeze(-1)
