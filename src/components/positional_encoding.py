import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, dropout: float, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.model_dimension = model_dimension

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO - Add positional encoding to X
        return X