"""Fused variable-length attention and CSR aggregation."""
from .interface import varlen_attention_conv

__all__ = ["VarlenAttentionConv", "varlen_attention_conv"]


def __getattr__(name):
    if name == "VarlenAttentionConv":
        from ...modules.varlen_attention import VarlenAttentionConv

        return VarlenAttentionConv
    raise AttributeError(f"{__name__} has no attribute {name!r}")
