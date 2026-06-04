import torch
import torch.nn as nn


class UnimodalClassifier(nn.Module):
    """
    MLP classifier for a single modality (audio or visual).

    Returns a raw logit — use BCEWithLogitsLoss during training
    and Sigmoid manually at inference.
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)