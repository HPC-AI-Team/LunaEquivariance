"""FP32/FP16/BF16 attention over CSR variable-length sequences."""

from .interface import varlen_attention

__all__ = ["VarlenAttention", "varlen_attention"]


def __getattr__(name):
    if name == "VarlenAttention":
        from ...modules.varlen_attention import VarlenAttention

        return VarlenAttention
    raise AttributeError(f"{__name__} has no attribute {name!r}")
