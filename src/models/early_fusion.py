import torch
import torch.nn as nn


class EarlyFusionMLP(nn.Module):
    """
    Early Fusion classifier for multimodal deepfake detection.

    Concatenates audio and (PCA-reduced) visual feature vectors,
    then passes them through a 3-layer MLP.

    Returns a raw logit (no Sigmoid) — use BCEWithLogitsLoss during training
    and apply Sigmoid manually at inference time.
    """

    def __init__(self, audio_dim: int, visual_dim: int, dropout: float = 0.3):
        super().__init__()
        input_dim = audio_dim + visual_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, audio_vec: torch.Tensor, visual_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio_vec, visual_vec], dim=-1)
        return self.classifier(x).squeeze(-1)
