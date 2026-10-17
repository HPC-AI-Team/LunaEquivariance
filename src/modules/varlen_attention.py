"""CSR attention modules with optional atom aggregation."""

from torch import nn

from ..nn.varlen_attention.interface import varlen_attention
from ..nn.varlen_attention_conv.interface import varlen_attention_conv


class VarlenAttention(nn.Module):
    """Attention configuration shared by different CSR batches."""

    def __init__(self, *, num_heads=1, softmax_scale=None, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(self, q, k, v, csr, *, bias=None):
        return varlen_attention(
            q,
            k,
            v,
            csr,
            num_heads=self.num_heads,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            bias=bias,
        )


class VarlenAttentionConv(nn.Module):
    def __init__(self, *, num_heads=1, softmax_scale=None, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(self, q, k, v, csr, *, envelope=None, return_attn_out=False, bias=None):
        return varlen_attention_conv(
            q,
            k,
            v,
            csr,
            envelope=envelope,
            return_attn_out=return_attn_out,
            num_heads=self.num_heads,
            bias=bias,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )


__all__ = ["VarlenAttention", "VarlenAttentionConv"]
