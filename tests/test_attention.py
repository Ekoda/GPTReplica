import pytest
import torch
from src.components.attention import MultiHeadSelfAttention

def test_multi_head_attention_output_shape():
    batch_size = 32
    seq_len = 10
    model_dimension = 64
    n_heads = 8

    attention = MultiHeadSelfAttention(model_dimension, n_heads, dropout=0.1)
    X = torch.rand(batch_size, seq_len, model_dimension)
    out = attention(X)

    assert out.shape == X.shape


def test_multi_head_attention_zero_input():
    batch_size = 32
    seq_len = 10
    model_dimension = 64
    n_heads = 8

    attention = MultiHeadSelfAttention(model_dimension, n_heads, dropout=0.1)
    X = torch.zeros(batch_size, seq_len, model_dimension)
    out = attention(X)

    assert torch.allclose(out, X, atol=1e-7)  # The output should be close to zero, as the input is zero


@pytest.mark.parametrize("model_dimension,n_heads", [(64, 8), (128, 4), (256, 16)])
def test_multi_head_attention_random_input(model_dimension, n_heads):
    batch_size = 32
    seq_len = 10

    attention = MultiHeadSelfAttention(model_dimension, n_heads, dropout=0.1)
    X = torch.rand(batch_size, seq_len, model_dimension)
    out = attention(X)

    assert out.shape == X.shape

def test_masked_self_attention():
    model_dimension = 64
    n_heads = 8
    dropout = 0.1
    model = MultiHeadSelfAttention(model_dimension, n_heads, dropout, mask=True)

    X = torch.rand(10, 32, model_dimension)

    output = model(X)
    assert output.shape == X.shape

    model_no_mask = MultiHeadSelfAttention(model_dimension, n_heads, dropout, mask=False)
    output_no_mask = model_no_mask(X)
    assert not torch.allclose(output, output_no_mask)
