import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dimension: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dimension, model_dimension * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension * 4, model_dimension)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)
