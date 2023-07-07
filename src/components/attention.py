import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    '''
    Standard Multi Head Attention Layer as described in the paper "Attention Is All You Need" (Vaswani, 2017).
    Utilizing the native PyTorch implementation, good stuff!

    https://arxiv.org/abs/1706.03762v5
    '''
    def __init__(self, model_dimension: int, n_heads: int, dropout: float, mask: bool = True):
        super().__init__()
        self.mask = mask
        self.multi_head_attention = nn.MultiheadAttention(model_dimension, n_heads, dropout=dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.mask:
            mask = torch.triu(torch.ones((X.shape[0], X.shape[0])), diagonal=1) == 1
            mask = mask.to(X.device)
            return self.multi_head_attention(X, X, X, attn_mask=mask)[0]
        return self.multi_head_attention(X, X, X)[0]


class MultiQueryAttention(nn.Module):
    '''
    Multi Query Attention Layer as described in the paper "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)

    In Multi Query Attention, the keys and values are shared across attention heads, reducing the memory required at inference time at the cost of a small decrease in performance as compared to multi-head attention.
    The aim being to reduce the memory bandwidth requirements.

    Notes on the implementation can be found in exploration/attention.ipynb

    https://arxiv.org/abs/1911.02150

    '''
    def __init__(self, model_dimension: int, n_heads: int, dropout: float, mask: bool = True):
        super().__init__()
        self.head_dimension = model_dimension // n_heads
        self.n_heads = n_heads
        self.mask = mask
        self.queries = nn.Linear(model_dimension, model_dimension, bias=False)
        self.kv_projection = nn.Linear(model_dimension, self.head_dimension * 2, bias=False)
        self.linear = nn.Linear(model_dimension, model_dimension, bias=False)
        self.dropout_p = dropout
        self.r_dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_len, seq_len, embd_dim = X.shape
        Q = self.queries(X).view(batch_len, seq_len, self.n_heads, self.head_dimension).transpose(1, 2)
        K, V = self.kv_projection(X).unsqueeze(1).expand(-1, self.n_heads, -1, -1).split(self.head_dimension, dim=-1)
        heads = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=self.mask)
        concat = heads.transpose(1, 2).contiguous().view(batch_len, seq_len, embd_dim)
        linear = self.linear(concat)
        return self.r_dropout(linear)
