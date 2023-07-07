import torch
from torch import nn
from src.components.attention import MultiHeadSelfAttention, MultiQueryAttention
from src.components.feed_forward import FeedForwardNetwork
from src.components.layer_norm import LayerNorm

class Decoder(nn.Module):
    def __init__(self, model_dimension: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiQueryAttention(model_dimension, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(model_dimension, dropout)
        self.attention_pre_norm = LayerNorm(model_dimension)
        self.feed_forward_pre_norm = LayerNorm(model_dimension)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.attention_pre_norm.forward(X)
        X = self.attention.forward(X) + X
        X = self.feed_forward_pre_norm.forward(X)
        X = self.feed_forward.forward(X) + X
        return X