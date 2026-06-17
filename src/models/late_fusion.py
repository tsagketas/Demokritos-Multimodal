import torch
import torch.nn as nn


class LateFusionCombiner(nn.Module):
    """
    Combines audio and visual posterior scores for late fusion.

    strategy="average":
        final_score = weight * score_a + (1 - weight) * score_v
        Returns a probability in [0, 1]. No training needed.

    strategy="learned":
        Trains Linear(2 -> 1) on [score_a, score_v].
        Returns a raw logit — use BCEWithLogitsLoss during training
        and Sigmoid at inference.
    """

    def __init__(self, strategy: str = "average", weight: float = 0.5):
        super().__init__()
        if strategy not in ("average", "learned"):
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'average' or 'learned'.")
        self.strategy = strategy
        self.weight   = weight
        if strategy == "learned":
            self.linear = nn.Linear(2, 1, bias=True)

    def forward(self, score_a: torch.Tensor, score_v: torch.Tensor) -> torch.Tensor:
        """
        score_a, score_v: sigmoid outputs in [0, 1], shape (batch,)
        Returns:
            average  → probability in [0, 1]
            learned  → raw logit (apply Sigmoid at inference)
        """
        if self.strategy == "average":
            return self.weight * score_a + (1.0 - self.weight) * score_v
        x = torch.stack([score_a, score_v], dim=-1)
        return self.linear(x).squeeze(-1)