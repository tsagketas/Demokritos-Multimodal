import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list, dropout: float) -> nn.Sequential:
    layers, prev = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class UnimodalClassifier(nn.Module):
    """
    MLP classifier for a single modality (audio or visual).

    Returns a raw logit — use BCEWithLogitsLoss during training
    and Sigmoid manually at inference.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        self.hidden_dims = hidden_dims or [256, 128]
        self.classifier  = _build_mlp(input_dim, self.hidden_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)