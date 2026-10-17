"""Fused softmax over CSR graph segments."""

from .interface import build_graph_softmax_row_ptr, graph_softmax

__all__ = ["GraphSoftmax", "build_graph_softmax_row_ptr", "graph_softmax"]


def __getattr__(name):
    if name == "GraphSoftmax":
        from ...modules.graph_softmax import GraphSoftmax

        return GraphSoftmax
    raise AttributeError(f"{__name__} has no attribute {name!r}")
