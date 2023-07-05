import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, model_dimension: int, include_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(model_dimension))
        self.bias = nn.Parameter(torch.zeros(model_dimension)) if include_bias else None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(X, self.weight.shape, self.weight, self.bias)