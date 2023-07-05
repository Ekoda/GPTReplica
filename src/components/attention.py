import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension: int, n_heads: int, dropout: float, mask: bool = True):
        super().__init__()
        self.mask = mask
        self.multi_head_attention = nn.MultiheadAttention(model_dimension, n_heads, dropout=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.mask:
            mask = torch.triu(torch.ones((X.size(0), X.size(0))), diagonal=1) == 1
            mask = mask.to(X.device)
            return self.multi_head_attention(X, X, X, attn_mask=mask)[0]
        return self.multi_head_attention(X, X, X)[0]