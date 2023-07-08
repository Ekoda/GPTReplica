import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimensions: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimensions, 2) * (-np.log(10000.0) / model_dimensions))
        pe = torch.zeros(1, max_len, model_dimensions)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)